"""Independent offline audit using only the standard library; no project imports."""
import argparse
import ast
import csv
import hashlib
import json
import math
import pickletools
import struct
import zipfile
from pathlib import Path

if not __debug__:
    raise RuntimeError('Audit assertions require normal Python; do not use -O or -OO')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def npy_payload(path):
    with path.open('rb') as stream:
        assert stream.read(6) == b'\x93NUMPY'
        version = tuple(stream.read(2))
        width = 2 if version[0] == 1 else 4
        header_size = int.from_bytes(stream.read(width), 'little')
        header = ast.literal_eval(stream.read(header_size).decode('latin1'))
        assert header['shape'] == (64, 3, 5, 5)
        assert header['descr'] == '<f4' and not header['fortran_order']
        return stream.read()


def npy_weights(path):
    values = struct.unpack('<4800f', npy_payload(path))
    return [list(values[i*75:(i+1)*75]) for i in range(64)]


def checkpoint_provenance(repo):
    checkpoint = repo/'weights/best_fs.pt'
    npy = repo/'weights/conv1_weights.npy'
    with zipfile.ZipFile(checkpoint) as archive:
        metadata_files = [n for n in archive.namelist() if n.endswith('/data.pkl')]
        assert len(metadata_files) == 1
        prefix = metadata_files[0][:-len('data.pkl')]
        assert archive.read(prefix+'byteorder') == b'little'
        operations = [(op.name,arg) for op,arg,pos in pickletools.genops(archive.read(metadata_files[0]))]
        # Inspect exact first tensor metadata. No opcode or unpickled object is executed.
        meaningful = [(name,arg) for name,arg in operations if name != 'BINPUT']
        expected = [
            ('PROTO',2),('GLOBAL','collections OrderedDict'),('EMPTY_TUPLE',None),
            ('REDUCE',None),('MARK',None),('BINUNICODE','conv_layers.0.weight'),
            ('GLOBAL','torch._utils _rebuild_tensor_v2'),('MARK',None),('MARK',None),
            ('BINUNICODE','storage'),('GLOBAL','torch FloatStorage'),('BINUNICODE','0'),
            ('BINUNICODE','cuda:0'),('BININT2',4800),('TUPLE',None),('BINPERSID',None),
            ('BININT1',0),('MARK',None),('BININT1',64),('BININT1',3),('BININT1',5),
            ('BININT1',5),('TUPLE',None),('MARK',None),('BININT1',75),('BININT1',25),
            ('BININT1',5),('BININT1',1),('TUPLE',None),('NEWFALSE',None),
            ('BINGET',0),('EMPTY_TUPLE',None),('REDUCE',None),('TUPLE',None),('REDUCE',None),
            ('BINUNICODE','conv_layers.1.weight')]
        assert meaningful[:len(expected)] == expected, 'Checkpoint first-tensor metadata changed'
        payload = archive.read(prefix+'data/0')
        assert len(payload) == 4800*4 and payload == npy_payload(npy)
    return dict(passed=True,method='zipfile + pickletools opcode inspection; no unpickle or torch.load',
                tensor='conv_layers.0.weight',storage='data/0',dtype='float32',byteorder='little',
                shape=[64,3,5,5],stride=[75,25,5,1],offset=0,storage_count=4800,
                payload_bytes=len(payload),payload_sha256=hashlib.sha256(payload).hexdigest(),
                checkpoint_sha256=sha(checkpoint),npy_sha256=sha(npy),payload_identical=True)


def metrics(actual, expected):
    n = len(actual)
    assert n == len(expected) and n > 0
    a_mean, e_mean = math.fsum(actual)/n, math.fsum(expected)/n
    centered_a = [x-a_mean for x in actual]
    centered_e = [x-e_mean for x in expected]
    ss_a = math.fsum(x*x for x in centered_a)
    ss_e = math.fsum(x*x for x in centered_e)
    covariance = math.fsum(a*b for a,b in zip(centered_a,centered_e))
    correlation = covariance/math.sqrt(ss_a*ss_e) if ss_a*ss_e else None
    errors = [a-e for a,e in zip(actual,expected)]
    rmse = math.sqrt(math.fsum(x*x for x in errors)/n)
    std = math.sqrt(ss_e/n)
    return dict(count=n, bias=math.fsum(errors)/n, rmse=rmse,
                expected_std_ddof0=std, pearson=correlation,
                historical_gate_pass=bool(correlation is not None and correlation >= .9
                                          and std > 0 and rmse <= .6*std))


def audit(repo, run, phase):
    paths = [run/(phase+'.json'), run/(phase+'_programming.csv'),
             run/(phase+'_mvm_outputs.csv'),run/(phase+'_mvm_inputs.json'),
             repo/'weights/conv1_weights.npy']
    saved = json.loads(paths[0].read_text())
    with paths[1].open(newline='') as stream:
        programming = list(csv.DictReader(stream))
    with paths[2].open(newline='') as stream:
        outputs = list(csv.DictReader(stream))
    inputs = json.loads(paths[3].read_text())['inputs']
    weights = npy_weights(paths[4])
    assert len(inputs) == 92 and all(len(x) == 75 for x in inputs)
    assert inputs[0] == [0.0]*75
    for i in range(75):
        assert inputs[i+1] == [float(j == i) for j in range(75)]
    assert all(sum(x) == 8 and set(x) <= {0,1} for x in inputs[76:])
    errors, max_difference = [], [0.0]
    checks = [0]

    def same(actual, expected, label):
        checks[0] += 1
        if isinstance(actual, dict):
            for key, value in actual.items():
                same(value, expected[key], label+'.'+key)
        elif isinstance(actual, list):
            assert len(actual) == len(expected), label
            for i,(a,e) in enumerate(zip(actual,expected)):
                same(a,e,label+'['+str(i)+']')
        elif isinstance(actual, bool) or actual is None:
            if actual != expected:
                errors.append(dict(field=label, recalculated=actual, saved=expected))
        else:
            difference = abs(actual-expected)
            max_difference[0] = max(max_difference[0], difference)
            if not math.isclose(actual,expected,rel_tol=1e-10,abs_tol=1e-12):
                errors.append(dict(field=label, recalculated=actual, saved=expected))

    reconstructed, measured, details = {}, {}, []
    expected_keys = {(r['filter_id'],r['repetition']) for r in saved['results']}
    assert len(expected_keys) == len(saved['results'])
    assert {(int(r['filter_id']),int(r['repetition'])) for r in programming} == expected_keys
    assert {(int(r['filter_id']),int(r['repetition'])) for r in outputs} == expected_keys
    for stored in saved['results']:
        fid, rep = stored['filter_id'], stored['repetition']
        key = (fid,rep)
        ideal_weights = weights[fid]
        scale = max(abs(x) for x in ideal_weights)
        rows = [r for r in programming if (int(r['filter_id']),int(r['repetition'])) == key]
        assert len(rows) == 150
        pair = {(int(r['tap']),int(r['wl'])):r for r in rows}
        assert len(pair) == 150 and set(pair) == {(t,w) for t in range(75) for w in (0,2)}
        for r in rows:
            tap,wl,tile = int(r['tap']),int(r['wl']),int(r['tile'])
            expected_tile = 0 if tap < 32 else 1 if tap < 64 else 2
            assert tile == expected_tile
            assert r['array_id'] == ('B' if tile == 1 else 'A')
            assert int(r['bl']) == tap-(0,32,64)[tile]
            same(float(r['target_q']), .03 + (1 if wl == 0 else -1)*.02*ideal_weights[tap]/scale,
                 str(key)+'.target_q.'+str((tap,wl)))
        reconstructed[key] = [(float(pair[t,0]['observed_q'])-float(pair[t,2]['observed_q']))*scale/.04
                              for t in range(75)]
        out = [r for r in outputs if (int(r['filter_id']),int(r['repetition'])) == key]
        indexed = {int(r['input_id']):r for r in out}
        assert len(out) == 92 and set(indexed) == set(range(92))
        measured[key] = [float(indexed[i]['measured']) for i in range(92)]
        ideal = [math.fsum(a*b for a,b in zip(v,ideal_weights)) for v in inputs]
        predicted = [math.fsum(a*b for a,b in zip(v,reconstructed[key])) for v in inputs]
        same(ideal,[float(indexed[i]['ideal']) for i in range(92)],str(key)+'.ideal_outputs')
        same(predicted,[float(indexed[i]['programmed_prediction']) for i in range(92)],str(key)+'.predicted_outputs')
        result = dict(filter_id=fid,repetition=rep,programmed_cell_count=150,
                      target_reached_count=sum(r['target_reached']=='True' for r in rows),
                      reset_success_count=sum(r['reset_ok']=='True' for r in rows),input_count=92,
                      weights=metrics(reconstructed[key],ideal_weights),
                      mvm_vs_ideal=metrics(measured[key],ideal),
                      mvm_vs_programmed=metrics(measured[key],predicted))
        same(result,stored,str(key)+'.metrics')
        same(reconstructed[key],saved['reconstructed_weights'][str(fid)][rep],str(key)+'.weights')
        details.append(result)

    prefix_details = []
    for item in saved['prefix_means']:
        fid,k = item['filter_id'],item['K']
        mean_w = [math.fsum(reconstructed[fid,r][i] for r in range(k))/k for i in range(75)]
        mean_y = [math.fsum(measured[fid,r][i] for r in range(k))/k for i in range(92)]
        ideal = [math.fsum(a*b for a,b in zip(v,weights[fid])) for v in inputs]
        result = dict(filter_id=fid,K=k,readback_weights=mean_w,
                      weights=metrics(mean_w,weights[fid]),mvm_vs_ideal=metrics(mean_y,ideal),
                      all_resets_succeeded=all(d['reset_success_count']==150 for d in details
                                              if d['filter_id']==fid and d['repetition']<k))
        same(result,item,'prefix.'+str((fid,k)))
        prefix_details.append(result)
    return dict(phase=phase,passed=not errors,source='simulator',hardware_qualification=False,
                checkpoint_provenance=checkpoint_provenance(repo),
                method='Independent standard-library CSV reconstruction and metrics; original NPY parsed directly. No suite/project imports.',
                limits='Checks saved CSV consistency, not independent acquisition or physical accuracy; raw ADC audit is separate.',
                input_hashes={str(p.relative_to(repo)).replace('\\','/'):sha(p) for p in paths},
                comparisons=checks[0],max_absolute_difference=max_difference[0],errors=errors,
                filters=saved['filter_count'],programming_rows=len(programming),output_rows=len(outputs),
                results=details,prefix_means=prefix_details)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--run',required=True)
    parser.add_argument('--phase',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=audit(args.repo,args.repo/args.run,args.phase)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('phase','passed','comparisons','max_absolute_difference','filters','programming_rows','output_rows')}))
    raise SystemExit(0 if result['passed'] else 1)
