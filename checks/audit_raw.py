"""Stream completed simulator raw ADC logs and independently rebuild saved results.

No project imports, hardware access, unpickle, or simulator execution. Paths passed
as --run are relative to --repo (default current directory).
"""
import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

from audit_results import npy_weights

if not __debug__:
    raise RuntimeError('Audit assertions require normal Python; do not use -O or -OO')


def streamed_sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def raw_lines(repo,run,storage):
    raw=run/'raw_tasks.jsonl'
    if raw.is_file():
        with raw.open('rb') as stream:
            yield from stream
        return
    assert storage is not None, 'Neither raw_tasks.jsonl nor trace_storage.json exists'
    parts=storage['parts']
    assert parts, 'Empty trace archive'
    assert storage['format']=='ordered_gzip_jsonl_parts'
    pending=b''
    seen=set()
    for part in parts:
        relative=Path(part['path'])
        assert not relative.is_absolute() and '..' not in relative.parts
        path=repo/relative
        path.resolve().relative_to(run.resolve())
        assert path not in seen
        seen.add(path)
        assert streamed_sha(path)==part['sha256'], 'Compressed part hash mismatch: '+str(relative)
        assert path.stat().st_size==part['compressed_bytes']
        part_digest=hashlib.sha256()
        part_bytes,part_records=0,0
        with gzip.open(path,'rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):
                part_digest.update(block)
                part_bytes+=len(block)
                part_records+=block.count(b'\n')
                lines=(pending+block).split(b'\n')
                pending=lines.pop()
                for line in lines:
                    yield line+b'\n'
        assert not pending, 'Part must end at a full JSONL record'
        assert part_digest.hexdigest()==part['uncompressed_sha256'], 'Uncompressed part hash mismatch'
        assert part_bytes==part['uncompressed_bytes'] and part_records==part['records']
    if pending:
        yield pending


def audit(repo, run):
    manifest = json.loads((run/'manifest.json').read_text())
    assert manifest['status'] == 'completed', 'Only a fully completed run can be audited'
    assert manifest['source'] == 'simulator'
    assert set(manifest['completed_phases']) == {'baseline','phase1','phase2','phase3','phase4'}
    protocol = json.loads((run/'protocol.json').read_text())
    storage_path=run/'trace_storage.json'
    storage=json.loads(storage_path.read_text()) if storage_path.exists() else None
    board = protocol['board']
    weights = npy_weights(repo/protocol['suite']['weights_file'])
    scales = [max(abs(x) for x in w) for w in weights]
    nominal_v = protocol['suite']['input_voltage']
    dac_code = round(nominal_v*(2**board['dac_bit']-1)/board['vol_ref_dac'])
    decoded_v = round(dac_code/((2**board['dac_bit']-1)/board['vol_ref_dac']),3)
    assert decoded_v > 0
    errors,checks,max_difference = [],0,0.0
    mismatches = 0

    def same(actual,expected,label):
        nonlocal checks,max_difference,mismatches
        checks += 1
        if actual is None or expected is None:
            equal = actual == expected
        else:
            max_difference = max(max_difference,abs(actual-expected))
            equal = math.isclose(actual,expected,rel_tol=1e-10,abs_tol=1e-12)
        if not equal:
            mismatches += 1
            if len(errors) < 20:
                errors.append(dict(field=label,recomputed=actual,saved=expected))

    def stats(values):
        mean = math.fsum(values)/len(values)
        sd = math.sqrt(math.fsum((x-mean)**2 for x in values)/(len(values)-1)) if len(values)>1 else None
        return mean,sd

    mapping,characterization,baseline = defaultdict(list),defaultdict(list),defaultdict(list)
    first,mvm_parts = {},defaultdict(dict)
    raw_count,raw_bytes,mode_counts = 0,0,defaultdict(int)
    digest = hashlib.sha256()
    stream=raw_lines(repo,run,storage)
    try:
        for line in stream:
            digest.update(line)
            raw_bytes+=len(line)
            row = json.loads(line)
            raw_count += 1
            assert row['source'] == 'simulator' and row['status'] == 'ok'
            assert row['task_id'] == raw_count
            adc,mode = row['raw_adc'],row['task']['mode_flag']
            assert type(adc) is int
            mode_counts[mode] += 1
            if mode == 7:
                assert 1 <= adc <= 2**board['adc_bit']-2
                resistance = (board['gain']*board['res_load']*board['vol_read']*2**board['adc_bit']
                              /(adc*board['vol_ref_adc'])-board['res_switches']-board['res_load'])
                conductance = 1/resistance
                q = board['sum_gain']/resistance
                for name,value in (('resistance_ohm',resistance),('conductance_siemens',conductance),('q',q)):
                    same(value,row[name],'raw '+str(raw_count)+' '+name)
                if row['phase'] == 'mapping_readback':
                    key=(row['stage'],row['filter_id'],row['repetition'],row['tile'],row['tap'],row['wl'],row['bl'])
                    mapping[key].append(q)
                elif row['phase'] == 'characterization_readback':
                    characterization[row['wl'],row['bl'],row['target_q'],row['cycle']].append(conductance)
                elif row['phase'] == 'repeatability':
                    baseline[row['wl'],row['bl']].append(conductance)
                elif row['phase'] == 'first_check':
                    assert (row['wl'],row['bl']) not in first
                    first[row['wl'],row['bl']] = q
            elif mode == 10:
                assert 0 <= adc <= 2**board['adc_bit']-2
                output = adc*board['vol_ref_adc']/2**board['adc_bit']
                same(output,row['output_voltage'],'raw '+str(raw_count)+' output_voltage')
                key=(row['stage'],row['filter_id'],row['repetition'],row['input_id'])
                subkey=(row['tile'],row['wl'])
                assert subkey not in mvm_parts[key], 'Duplicate MVM tile/WL'
                mvm_parts[key][subkey] = output
            else:
                raise AssertionError('Unexpected raw mode '+str(mode))
            if raw_count % 100000 == 0:
                print('Audited raw tasks:',raw_count,flush=True)
    finally:
        stream.close()
    if storage is not None:
        assert digest.hexdigest()==storage['uncompressed_sha256'], 'Uncompressed trace hash mismatch'
        assert raw_bytes==storage['uncompressed_bytes'], 'Uncompressed trace length mismatch'
        assert raw_count==storage['records'], 'Uncompressed trace count mismatch'
    same(raw_count,manifest['raw_task_count'],'raw count')
    assert all(len(v)==3 for v in mapping.values())
    assert len(characterization)==75 and all(len(v)==5 for v in characterization.values())
    assert len(baseline)==3 and all(len(v)==20 for v in baseline.values())
    assert len(first)==3
    assert all(set(v)=={(t,w) for t in range(3) for w in (0,2)} for v in mvm_parts.values())
    phase_counts = {}
    for phase in ('phase2','phase3','phase4'):
        consumed_mapping,consumed_mvm = set(),set()
        with (run/(phase+'_programming.csv')).open(newline='') as stream:
            for row in csv.DictReader(stream):
                key=(phase,int(row['filter_id']),int(row['repetition']),int(row['tile']),int(row['tap']),int(row['wl']),int(row['bl']))
                assert key not in consumed_mapping
                consumed_mapping.add(key)
                same(math.fsum(mapping[key])/3,float(row['observed_q']),'programmed q '+str(key))
        with (run/(phase+'_mvm_outputs.csv')).open(newline='') as stream:
            for row in csv.DictReader(stream):
                key=(phase,int(row['filter_id']),int(row['repetition']),int(row['input_id']))
                assert key not in consumed_mvm
                consumed_mvm.add(key)
                parts=mvm_parts[key]
                reconstructed=math.fsum(parts[t,0]-parts[t,2] for t in range(3))*scales[key[1]]/(.04*decoded_v)
                same(reconstructed,float(row['measured']),'measured MVM '+str(key))
        assert consumed_mapping == {k for k in mapping if k[0]==phase}
        assert consumed_mvm == {k for k in mvm_parts if k[0]==phase}
        phase_counts[phase]=dict(programming_rows=len(consumed_mapping),mvm_outputs=len(consumed_mvm))
    with (run/'characterization_attempts.csv').open(newline='') as stream:
        consumed=set()
        for row in csv.DictReader(stream):
            key=(int(row['wl']),int(row['bl']),float(row['target_q']),int(row['cycle']))
            assert key not in consumed
            consumed.add(key)
            mean,sd=stats(characterization[key])
            same(mean,float(row['mean_conductance_siemens']),'characterization mean '+str(key))
            same(sd,float(row['read_std_s_ddof1']),'characterization SD '+str(key))
        assert consumed==set(characterization)
    for cell in json.loads((run/'baseline.json').read_text())['cells']:
        key=(cell['wl'],cell['bl'])
        values=baseline[key]
        mean,sd=stats(values)
        for field,value in dict(first_q=first[key],mean=mean,std_ddof1=sd,min=min(values),max=max(values),count=len(values)).items():
            same(value,cell[field],'baseline '+str(key)+' '+field)
    return dict(passed=not mismatches,source='simulator',hardware_qualification=False,
                method='Streaming raw ADC conversion without project imports; raw readback means and native MVM tile aggregation rebuilt into CSV.',
                raw_sha256=digest.hexdigest(),raw_tasks=raw_count,mode_counts=dict(mode_counts),
                raw_bytes=raw_bytes,trace_storage_verified=storage is not None,
                compressed_parts_verified=storage is not None and not (run/'raw_tasks.jsonl').is_file(),
                comparisons=checks,mismatch_count=mismatches,first_errors=errors,max_absolute_difference=max_difference,
                mapping_readback_groups=len(mapping),characterization_groups=len(characterization),
                baseline_cells=len(baseline),phase_counts=phase_counts,
                mvm_input=dict(nominal_voltage=nominal_v,dac_code=dac_code,native_decoded_voltage=decoded_v),
                limits='Confirms recorded ADC arithmetic and aggregation. Does not establish physical calibration or independent model correctness.')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--repo',type=Path,default=Path.cwd())
    parser.add_argument('--run',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert not Path(args.run).is_absolute(), '--run must be relative to repository'
    result=audit(args.repo,args.repo/args.run)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('passed','raw_tasks','comparisons','mismatch_count','max_absolute_difference')}))
    raise SystemExit(0 if result['passed'] else 1)
