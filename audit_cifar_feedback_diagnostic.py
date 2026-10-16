"""Independent full diagnostic CIFAR audit. No acquisition-module imports."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
from collections import Counter
from configparser import ConfigParser
import statistics
from contextlib import contextmanager
import re

FEEDBACK_AUDIT_SHA='cde63401e787770523e002bc9f4a970dea1d528c3367d41ef7d0d05b2c13a5ee'
if hashlib.sha256(Path(__file__).with_name('audit_cifar_feedback.py').read_bytes()).hexdigest()!=FEEDBACK_AUDIT_SHA:
    raise RuntimeError('Frozen independent feedback helper changed')
from audit_cifar_feedback import (CompactChecks,reset_replay,pair_replay,counts_between,
    fidelity,pools_for,expected_counts,clean,ledger,FEEDBACK_PINS)
from audit_cifar_memristor import (Replay,sha,local,load,verify_inputs,calibration,output_voltage,
    multiply,audit_matrix,replay_block,aggregate_independent,audit_git_status,PINS,PRIVATE,BOARD,resistance,dac)


def fixed_decode(corrected_rows,gain):
    if gain not in (1.,.03) or len(corrected_rows)!=2 or not all(math.isfinite(v) for v in corrected_rows):
        raise ValueError('Expected two finite measured channels and a reviewed backend-fixed gain')
    return (corrected_rows[0]-corrected_rows[1])/gain


def fixed_effective_weights(matrix,anchors,gain):
    """Calibration-based reference only; never substituted for actual commands."""
    n=len(anchors[0][0]['singles']);differences=[]
    for repeat in anchors:
        row=[]
        for i in range(n):
            raw=[repeat[s]['singles'][i]['voltage_v']-repeat[s]['zero']['voltage_v'] for s in range(2)]
            row.append(fixed_decode(multiply(matrix,raw),gain))
        differences.append(row)
    return [statistics.mean(r[i] for r in differences) for i in range(n)]


DIAGNOSTIC_PINS={
    **{k:v for k,v in FEEDBACK_PINS.items() if k!='experiment.py'},
    'cifar_feedback_diagnostic.py':'d4bcc3c00d9fc37552af3c2f3b205fa9bb795ce057cafedd93c864ee33069e62',
    'configs/cifar_feedback_diagnostic.json':'c798a58227d25cc5ca7fe8a6f8b3472efb28fe90dde8759d86840e8b42e92246',
    'experiment.py':'252332367886083b2609d222361fe17846ccc33011bc16d36610f941b4391124'}
PROTOCOL='cifar-mode10-feedback-full-fixed-gain-diagnostic-v1'


def fixed_calibration(weights,differences,gain,floor_adc=5):
    fitted=calibration(weights,differences,floor_adc)
    effective=[v/gain for v in fitted['measured_difference_mean_v']]
    residuals=[v-w for v,w in zip(effective,weights)]
    return dict(decoder_gain_v_per_weight=gain,decoder_gain_is_fixed_before_acquisition=True,
        fitted_alpha_diagnostic=fitted,decode_usable=True,
        decode_usable_means_finite_arithmetic_not_weight_quality=True,effective_weights=effective,
        weight_residuals=residuals,weight_rmse=math.sqrt(statistics.mean(v*v for v in residuals)),
        measured_difference_mean_v=fitted['measured_difference_mean_v'],calibration_uses_cifar=False)


def raw_lines(path):
    """Verify every byte before yielding it; preserve pairing across part boundaries."""
    folder=Path(path).parent;index=load(folder/'raw_parts.json');parts=index['parts']
    expected=['raw.jsonl']+[f'raw.{i:05d}.jsonl' for i in range(1,len(parts))]
    if not parts or [p['path'] for p in parts]!=expected:raise ValueError('Invalid raw part order/names')
    actual=sorted(p.name for p in folder.iterdir() if re.fullmatch(r'raw(?:\.\d+)?\.jsonl',p.name))
    if actual!=sorted(expected):raise ValueError('Unindexed or missing raw part')
    if index['maximum_part_bytes']!=33554432:raise ValueError('Unreviewed raw part size')
    total=hashlib.sha256();size=count=0
    for part in parts:
        payload=local(folder,part['path']).read_bytes()
        if len(payload)>33554432:raise ValueError('Raw part exceeds limit')
        if len(payload)!=part['bytes'] or hashlib.sha256(payload).hexdigest()!=part['sha256']:
            raise ValueError('Raw part hash/size mismatch')
        lines=payload.splitlines(keepends=True)
        if any(not line.endswith(b'\n') or line.endswith(b'\r\n') or not line.strip() for line in lines):
            raise ValueError('Raw part does not contain complete canonical JSON lines')
        if len(lines)!=part['records']:raise ValueError('Raw part record count mismatch')
        total.update(payload);size+=len(payload);count+=len(lines)
        for line in lines:yield line.decode('utf-8')
    if (total.hexdigest(),size,count)!=(index['sha256'],index['bytes'],index['records']):
        raise ValueError('Raw stream commitment mismatch')


@contextmanager
def joined_raw(path):
    class Reader:
        def __init__(self):self.lines=iter(raw_lines(path))
        def readline(self):return next(self.lines,'')
    yield Reader()


def read_raw(path, source, ck, session_id=None):
    events=[]; latest={}; sequence=decisions=pulses=skips=0; previous_time=-math.inf
    with joined_raw(path) as stream:
        while line:=stream.readline():
            request=json.loads(line); kind=request['event']; label='raw/'+str(sequence+1)
            if session_id is not None:ck.equal(request.get('session_id'),session_id,label+'/session')
            if kind in ('calibration_frozen','route_frozen'):
                ck.equal(request['before_task_id'],sequence+1,label+'/freeze_sequence')
                events.append(request);continue
            ck.require(kind in ('request','skipped'),label+'/request_or_skip')
            ck.require(request['source']==source,label+'/source')
            ck.require(request['monotonic_time_s']>=previous_time,label+'/time')
            previous_time=request['monotonic_time_s']
            command=request['planned_command_si'] if kind=='skipped' else request['command_si']
            cell=(request['wl'],request['bl']);mode=command['mode_flag'];vol=command['vol']
            ck.require(0<=cell[0]<8 and 0<=cell[1]<32,label+'/address')
            ck.require(mode in (7,10),label+'/mode')
            if mode==7:
                ck.require(type(vol) in (int,float) and 0<=vol<=3,label+'/program_voltage')
                ck.equal(command['wl'],cell[0],label+'/wl');ck.equal(command['bl'],cell[1],label+'/bl')
                ck.require(command['sign'] in (0,1),label+'/polarity')
                ck.equal(request['native_voltage_conversion'],dac(vol),label+'/dac')
                ck.equal(command['pulse_width'],.000125 if source=='simulator' and vol else 0.,label+'/width')
                if vol:
                    decisions+=1;ck.require(cell in latest,label+'/previous_R_exists')
                    if cell in latest:
                        resistance_value,prior=latest[cell];limit=.04 if command['sign']==0 else .002
                        ck.equal(request,dict(programming_decision_id=decisions,previous_native_r_ohm=resistance_value,
                                 resistance_source_task_id=prior,predicted_current_a=vol/resistance_value,current_limit_a=limit),label+'/guard')
                        ck.require((vol/resistance_value>limit)==(kind=='skipped'),label+'/guard_decision')
                    if kind=='skipped':
                        skips+=1;ck.require(not any(k in request for k in ('task_id','raw_adc','q','native_result')),label+'/no_fabrication')
                        events.append(request);continue
                    pulses+=1
            else:
                ck.require(kind=='request',label+'/mvm_not_skipped')
                ck.require(isinstance(vol,list) and len(vol)==32 and all(v in (0,.1) for v in vol) and sum(v!=0 for v in vol)<=1,label+'/onehot')
                ck.equal(request['native_voltage_conversion'],[dac(v) for v in vol],label+'/dac')
                if any(vol):
                    ck.require(cell in latest,label+'/model_previous_R')
                    if cell in latest:ck.require(.1/latest[cell][0]<=.005 and .1/latest[cell][0]*330<=4,label+'/model_guard')
            sequence+=1;ck.equal(request['task_id'],sequence,label+'/sequence')
            ck.equal(command['id'],sequence if mode==7 else 0,label+'/command_id')
            next_line=stream.readline()
            if not next_line:raise ValueError('Unpaired request at '+label)
            result=json.loads(next_line);ck.equal(result['event'],'result',label+'/result')
            ck.equal(result,{k:v for k,v in request.items() if k!='event'},label+'/request_retained')
            ck.equal(result['status'],'ok',label+'/status')
            if result['status']!='ok':raise ValueError('Recorded hardware/transport error; partial audit required')
            ck.equal(result['returned_id'],command['id'],label+'/returned_id')
            ck.equal(result['native_result'],[result['native_r'],result['returned_id'],result['raw_adc']],label+'/native_tuple')
            ck.require(result['latency_s']>=0 and math.isfinite(result['latency_s']),label+'/latency')
            if mode==7:
                r=resistance(result['raw_adc']);ck.equal(result,dict(native_r=r,resistance_ohm=r,conductance_siemens=1/r,q=330/r,
                                      read_voltage_parameter_v=.3),label+'/readback')
                latest[cell]=(r,sequence)
            else:
                v=output_voltage(result['raw_adc']);ck.equal(result['output_voltage_v'],v,label+'/output_voltage')
                ck.require(v<=4,label+'/actual_headroom')
            ck.equal(result['physical_pulse_width_s'],None,label+'/unknown_physical_width')
            ck.equal(result['native_adc_samples'],10 if mode==7 and source=='hardware' else 1,label+'/native_samples')
            ck.equal(result['simulator_duration_steps'],125 if mode==7 and source=='simulator' and vol else None,label+'/simulator_steps')
            events.append(result)
    return events,dict(impact_attempts=sequence,programming_pulse_attempts=pulses,
                      programming_decisions=decisions,programming_skipped_current=skips)


def audit_diagnostic_tile(directory,filt,tile,fixture,plan,source,ck,session_id):
    saved=load(directory/'tile.json');fid=filt['filter_id'];tid=tile['tile_id'];meta=dict(filter_id=fid,tile_id=tid)
    active=copy.deepcopy([s for s in tile['slots'] if not s['padding']]);pools=pools_for(tile)
    events,counters=read_raw(directory/'raw.jsonl',source,ck,session_id);replay=Replay(events,ck)
    candidates=[];reset_rows=[];candidate_ledger=[];chosen={}
    for side,cells in pools.items():
        qualifying=[]
        for index,cell in enumerate(cells):
            md=dict(**meta,side=side,candidate_index=index);candidate=reset_replay(replay,cell,md,reset_rows)
            candidates.append(dict(side=side,candidate_index=index,**candidate));candidate_ledger.append(dict(**md,**candidate))
            if candidate['eligible']:qualifying.append(cell)
        chosen[side]=qualifying[:len(active)]
    ck.equal(saved['reset_candidates'],candidates,'reset/all_candidates')
    ck.equal(ledger(directory/'reset_candidates.jsonl'),candidate_ledger,'reset/candidate_ledger')
    ck.equal(ledger(directory/'reset_feedback.jsonl'),reset_rows,'reset/feedback_ledger')
    if any(len(cells)!=len(active) for cells in chosen.values()):
        ck.equal(saved['status'],'failed','reset/insufficient_status')
        ck.equal(saved['stop_reason'],'insufficient_quiet_reset_candidates_for_fixed_tile','reset/insufficient_reason')
        ck.equal(replay.position,len(events),'reset/no_measurement_after_insufficient_candidates')
        return dict(**meta,technical_completed=False,physical_weight_quality_pass=False,counters=counters,
                    reset_candidates=candidates,partials=[],acquisition_stop='insufficient_quiet_reset_candidates')
    for i,slot in enumerate(active):
        for side in chosen:slot[side]=chosen[side][i]
    ck.equal(saved['effective_route'],active,'routing/deterministic_weight_independent')
    route=load(directory/'frozen_route.json');freeze=events[replay.position];replay.position+=1
    ck.equal(route,dict(candidate_pools=pools,slots=active,mapping_uses_trained_weight_values=False,
                        frozen_before_task_id=events[replay.position]['task_id']),'routing/frozen_values')
    ck.equal(freeze,dict(event='route_frozen',session_id=session_id,**meta,sha256=sha(directory/'frozen_route.json'),
                        before_task_id=route['frozen_before_task_id']),'routing/freeze_commitment')
    gain=1. if source=='hardware' else .03;pairs=[];set_rows=[];pair_ledger=[];programming=[]
    for slot in active:
        md=dict(**meta,pair_tap=slot['tap_index']);pair=pair_replay(replay,slot,md,gain,set_rows)
        pairs.append(pair);pair_ledger.append(dict(**md,**pair))
        for side in ('positive_cell','negative_cell'):
            programming.append(dict(cell=slot[side],status='completed',tap=slot['tap_index'],side=side,
                feedback_pair_stop=pair['stop'],selected_SET_side=side==pair['active_set_side'],participated_in_SET=side==pair['active_set_side'] and pair['programming_pulse_attempts']>0,reset_eligible=True))
    ck.equal(saved['pairs'],pairs,'pairs/results');ck.equal(saved['programming'],programming,'pairs/assignments')
    ck.equal(ledger(directory/'pair_programming.jsonl'),pair_ledger,'pairs/ledger')
    ck.equal(ledger(directory/'set_feedback.jsonl'),set_rows,'pairs/feedback_ledger')
    wmax=max(abs(s['trained_weight']) for t in filt['tiles'] for s in t['slots'] if not s['padding'])

    prereads=[]
    for slot in active:
        for side in ('positive_cell','negative_cell'):
            prereads.append(replay.reads('tile_pre_readback',slot[side],3,tap=slot['tap_index'],side=side,**meta))
    ck.equal(saved['pre_readback'],[[{k:v for k,v in r.items() if k not in ('event','latency_s')} for r in group] for group in prereads],'tile/pre_readback')
    def anchors(phase):
        records=[]
        for repeat in range(5):
            pair=[]
            for side in ('positive_cell','negative_cell'):
                z=replay.mvm(phase,active[0][side],False,repeat=repeat,side=side,role='zero',**meta)
                singles=[replay.mvm(phase,s[side],True,repeat=repeat,side=side,role='single',tap=s['tap_index'],**meta) for s in active]
                pair.append(dict(zero=z,singles=singles))
            records.append(pair)
        return records
    calanchors=anchors('calibration');ck.equal(load(directory/'calibration_anchors.json'),calanchors,'tile/anchors')
    correction=audit_matrix(directory,replay,active,calanchors,meta,ck)
    ck.equal(saved['matrix'],correction,'tile/matrix',1e-9)
    differences=[]
    for plus,minus in calanchors:
        corrected=[multiply(correction['matrix'],[p['voltage_v']-plus['zero']['voltage_v'],m['voltage_v']-minus['zero']['voltage_v']])
                   for p,m in zip(plus['singles'],minus['singles'])]
        differences.append([pair[0]-pair[1] for pair in corrected])
    cal=fixed_calibration([s['trained_weight'] for s in active],differences,gain,plan['alpha_absolute_floor_adc'])
    ck.equal(cal['effective_weights'],fixed_effective_weights(correction['matrix'],calanchors,gain),'calibration/independent_fixed_units')
    freeze=events[replay.position];replay.position+=1
    ck.equal(freeze,dict(event='calibration_frozen',**meta,sha256=sha(directory/'frozen_calibration.json'),
                        matrix_sha256=sha(directory/'frozen_matrix.json'),
                        before_task_id=events[replay.position]['task_id']),'tile/freeze_commitment')
    cal.update(frozen_before_task_id=events[replay.position]['task_id'],trained_weights=[s['trained_weight'] for s in active],wmax_per_filter=wmax)
    ck.require(set(saved['calibration'])==set(cal),'tile/calibration_schema_supported')
    ck.equal(saved['calibration'],cal,'tile/calibration',1e-9);ck.equal(load(directory/'frozen_calibration.json'),cal,'tile/frozen_calibration',1e-9)
    partials=[]
    for ci,case in enumerate(fixture['cases']):
        for repeat in range(5):
            for t,spikes in enumerate(case['spikes_t12x75']):
                selected=[s for s in active if spikes[s['tap_index']]]
                event=dict(**meta,case=ci,original_test_index=case['original_test_index'],repeat=repeat,t=t)
                values,rows=replay_block(replay,active,selected,'cifar',event)
                corrected=multiply(correction['matrix'],values);difference=corrected[0]-corrected[1]
                partials.append(dict(**event,active_taps=[s['tap_index'] for s in selected],measured_rows=rows,
                                     raw_row_sums_v=values,corrected_row_sums_v=corrected,raw_signed_voltage_v=values[0]-values[1],
                                     signed_voltage_v=difference,trained_unit_contribution=fixed_decode(corrected,gain)))
    ck.equal(saved['partials'],partials,'tile/partials',1e-9)
    ck.equal([json.loads(line) for line in (directory/'partials.jsonl').read_text(encoding='utf-8').splitlines()],partials,'tile/partial_ledger',1e-9)
    finalanchors=anchors('final_anchors')
    ck.equal(saved['final_anchors'],finalanchors,'tile/final_anchors')
    postreads=[]
    for slot in active:
        for side in ('positive_cell','negative_cell'):
            postreads.append(replay.reads('tile_post_readback',slot[side],3,tap=slot['tap_index'],side=side,**meta))
    ck.equal(saved['post_readback'],[[{k:v for k,v in r.items() if k not in ('event','latency_s')} for r in group] for group in postreads],'tile/post_readback')
    ck.equal(replay.position,len(events),'tile/no_unconsumed_events')
    ck.equal(saved,counters,'tile/counters');ck.equal(saved['status'],'completed','tile/completed')
    def raw_differences(samples):
        return [[(row[0]['singles'][i]['voltage_v']-row[0]['zero']['voltage_v'])-(row[1]['singles'][i]['voltage_v']-row[1]['zero']['voltage_v']) for i in range(len(active))] for row in samples]
    weights=[s['trained_weight'] for s in active]
    before=fidelity(weights,raw_differences(calanchors),gain);after=fidelity(weights,raw_differences(finalanchors),gain)
    ck.equal(saved['physical_fidelity'],before,'fidelity/fixed_gain_after_all_writes')
    ck.equal(saved['final_physical_fidelity'],after,'fidelity/post_inference')
    eligible=before['physical_fidelity_pass'] and after['physical_fidelity_pass']
    ck.equal(saved['physical_weight_quality_pass'],eligible,'fidelity/reported_quality_not_collection_gate')
    n=len(active);candidate_count=2*n+6
    ck.require(counters['programming_decisions']<=55*candidate_count+109*n,'feedback/decision_bound')
    ck.require(counters['programming_pulse_attempts']<=counters['programming_decisions'],'feedback/pulse_bound')
    return dict(**meta,technical_completed=True,physical_weight_quality_pass=eligible,calibration=cal,matrix=correction,
        reset_candidates=candidates,pairs=pairs,effective_route=active,physical_fidelity=before,final_physical_fidelity=after,partials=partials,counters=counters,
        candidate_eligible_counts={side:sum(c['eligible'] for c in candidates if c['side']==side) for side in pools},
        pair_stop_counts=dict(Counter(p['stop'] for p in pairs)))


def verify_manifest(repo,directory,ck):
    m=load(directory/'manifest.json');git=m['invocation']['git']
    provenance=audit_git_status(git,m['source_sha256'],ck)
    ck.require(m['source'] in ('hardware','simulator'),'manifest/source')
    ck.equal(m['hardware_access'],m['source']=='hardware','manifest/hardware_access')
    ck.equal(m['historical_pilot_imported'],False,'manifest/no_pilot_import')
    ck.equal(m['all_tiles_new_measurements'],True,'manifest/fresh_tiles')
    ck.require('parent_pilot' not in m and 'stage' not in m,'manifest/no_legacy_resume')
    for name,digest in m['result_sha256'].items():ck.equal(sha(local(directory,name)),digest,'result_sha/'+name)
    for name,digest in m['source_sha256'].items():
        path=local(repo,name)
        if name in PRIVATE:
            ck.equal(digest,PRIVATE[name],'source/private_pin/'+name)
            if path.exists():ck.equal(sha(path),digest,'source/private_local/'+name)
        else:
            proc=subprocess.run(['git','-C',str(repo),'show',git['commit']+':'+name],capture_output=True)
            ck.equal(proc.returncode,0,'source/git_exists/'+name)
            ck.equal(hashlib.sha256(proc.stdout).hexdigest(),digest,'source/git_blob/'+name)
    for name,digest in {**PINS,**DIAGNOSTIC_PINS}.items():
        ck.equal(m['source_sha256'].get(name),digest,'source/reviewed_pin/'+name)
    plan=load(directory/'protocol.json')
    config=subprocess.run(['git','-C',str(repo),'show',git['commit']+':configs/cifar_feedback_diagnostic.json'],capture_output=True,check=True).stdout
    ck.equal(plan,json.loads(config),'protocol/snapshot_git_config')
    ck.equal(m['protocol_id'],PROTOCOL,'protocol/manifest');ck.equal(plan['protocol_id'],PROTOCOL,'protocol/id')
    ck.equal(plan,dict(target_gain_v_per_weight={'hardware':1.,'simulator':.03},
        maximum_impacts=580210,maximum_programming_pulses=74940,maximum_programming_decisions=74940,
        acquisition_completion_is_accuracy_claim=False,raw_part_maximum_bytes=33554432),'protocol/fixed_contract')
    ck.equal(m['fixed_decoder_gain_v_per_weight'],1. if m['source']=='hardware' else .03,'manifest/fixed_gain')
    board=ConfigParser();board.read(directory/'board.ini',encoding='utf-8')
    for key,value in BOARD.items():ck.equal(board.getfloat('board',key),value,'INI/'+key)
    start=load(directory/'start_manifest.json')
    for key in ('session_id','source_sha256','protocol_id','source','invocation'):
        ck.equal(m[key],start[key],'manifest/start_consistency/'+key)
    return m,plan,provenance


def coverage(results):
    sets={name:set() for name in ('reset_probed','reset_eligible','selected','set_attempted','represented')}
    for r in results:
        for candidate in r['reset_candidates']:
            addr=tuple(candidate['cell']);sets['reset_probed'].add(addr)
            if candidate['eligible']:sets['reset_eligible'].add(addr)
        for slot in r.get('effective_route',[]):
            for side in ('positive_cell','negative_cell'):
                addr=tuple(slot[side]);sets['selected'].add(addr)
                if r['technical_completed']:sets['represented'].add(addr)
        for pair in r.get('pairs',[]):
            if pair['programming_pulse_attempts']>0:sets['set_attempted'].add(tuple(pair[pair['active_set_side']]))
    out={key:dict(count=len(values),addresses=[list(v) for v in sorted(values)]) for key,values in sets.items()}
    out['probed_but_not_selected']=[list(v) for v in sorted(sets['reset_probed']-sets['selected'])]
    out['coverage_does_not_prove_programmability']=True
    return out


def audit_session(repo,directory,fixture,routing,ck):
    manifest,plan,provenance=verify_manifest(repo,directory,ck)
    selected=[(f,t) for f in routing['filters'] for t in f['tiles']];results=[]
    ck.equal(len(selected),28,'session/all_28_tiles');bound=expected_counts(selected,fixture)
    ck.equal(manifest,bound,'manifest/recomputed_budget')
    ck.equal(bound,dict(trained_coefficients=300,maximum_impacts=580210,maximum_programming_pulses=74940),'budget/reviewed')
    permitted={f"f{f['filter_id']}_t{t['tile_id']}" for f,t in selected}
    actual={p.name for p in (directory/'tiles').iterdir() if p.is_dir()}
    ck.require(actual<=permitted,'session/no_unexpected_tiles')
    if manifest['status']=='completed':ck.equal(sorted(actual),sorted(permitted),'session/all_tiles_present')
    visited=set()
    for filt,tile in selected:
        name=f"tiles/f{filt['filter_id']}_t{tile['tile_id']}";folder=directory/name
        if not folder.exists():
            ck.require(manifest['status']!='completed','session/missing_tile');break
        visited.add(folder.name)
        for required in ('raw.jsonl','raw_parts.json','tile.json','reset_candidates.jsonl'):
            ck.require(name+'/'+required in manifest['result_sha256'],'manifest/required/'+required)
        for artifact in folder.iterdir():
            if artifact.is_file():ck.require(name+'/'+artifact.name in manifest['result_sha256'],'manifest/tile_artifact/'+artifact.name)
        result=audit_diagnostic_tile(folder,filt,tile,fixture,plan,manifest['source'],ck,manifest['session_id'])
        results.append(result)
        if not result['technical_completed']:break
    ck.equal(sorted(visited),sorted(actual),'session/no_unreplayed_tiles')
    complete=len(results)==28 and all(r['technical_completed'] for r in results)
    trajectories=[];quality=[];physical=complete and all(r['physical_weight_quality_pass'] for r in results)
    counts={k:sum(r['counters'][k] for r in results) for k in ('impact_attempts','programming_pulse_attempts','programming_decisions','programming_skipped_current')}
    addresses=coverage(results)
    if complete:
        saved=load(directory/'summary.json');trajectories,quality=aggregate_independent(results,routing,fixture,ck)
        ck.equal(saved['trajectories'],trajectories,'summary/measured_sum_before_BN_LIF_two_refs',1e-8)
        ck.equal(saved,dict(status='completed',technical_completed=True,filter_tiles_completed=28,
            acquisition_complete=True,decoded_conv_bn_lif_complete=True,physical_weight_quality_pass=physical,
            fixed_decoder_gain_v_per_weight=1. if manifest['source']=='hardware' else .03,
            fitted_alpha_used_for_decoding=False,diagnostic_acquisition_not_accuracy_qualification=True,
            actual_branch_uses_software_dot_product=False,full_network_accuracy_claim=False,
            total_impacts=counts['impact_attempts'],total_pulses=counts['programming_pulse_attempts'],address_coverage=addresses),'summary/flags_counts_coverage')
        ck.equal(saved['tile_status'],[dict(filter_id=r['filter_id'],tile_id=r['tile_id'],
            physical_weight_quality_pass=r['physical_weight_quality_pass'],
            fitted_alpha_v_per_weight=r['calibration']['fitted_alpha_diagnostic']['alpha_v_per_weight']) for r in results],'summary/tile_status')
        ck.equal(len(trajectories)*12,2400,'summary/all_2400_time_outputs')
        ck.equal(sum(len(r['partials']) for r in results),16800,'summary/all_fresh_partial_points')
        ck.equal(manifest['status'],'completed','manifest/complete');ck.equal(manifest['acquisition_complete'],True,'manifest/acquired')
    else:ck.equal(manifest['status'],'failed','manifest/partial_scientific_stop_preserved')
    ck.require(counts['impact_attempts']<=580210,'session/impact_bound')
    ck.require(counts['programming_decisions']<=74940,'session/decision_bound')
    compact=[]
    for r in results:
        item={k:v for k,v in r.items() if k not in ('partials','reset_candidates','pairs','matrix')}
        if 'matrix' in r:item['matrix']={k:r['matrix'][k] for k in ('matrix','selected_model','selection_pass','training_design_rank','training_design_condition')}
        compact.append(item)
    return dict(session_id=manifest['session_id'],source=manifest['source'],acquisition_status=manifest['status'],
        acquisition_complete=complete,physical_weight_quality_pass=physical,provenance=provenance,counters=counts,
        manifest_sha256=sha(directory/'manifest.json'),partial_points=sum(len(r['partials']) for r in results),
        full_time_outputs=len(trajectories)*12,tiles=compact,address_coverage=addresses,quality=quality,
        limitations=['Consistency PASS does not mean weight fidelity or network accuracy PASS.',
            'Fixed HW gain 1 and SIM gain 0.03 are reviewed provisional scales, not independently calibrated hardware gains.',
            'Fitted alpha is diagnostic only. No CPU weight substitution enters measured outputs.',
            'All 28 tiles are freshly acquired; sequential physical-cell reuse is not 300 simultaneous stored weights.',
            'Raw-to-derived tolerance 1e-9; aggregate 1e-8; float32 golden comparison 1e-5; none is a physical-quality threshold.'])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--repo',default='.')
    parser.add_argument('--session',required=True);parser.add_argument('--output',required=True);args=parser.parse_args(argv)
    repo=Path(args.repo).resolve();directory=local(repo,args.session);output=local(repo,args.output)
    if output==directory or output.is_relative_to(directory):raise ValueError('Audit output must not change acquired data')
    ck=CompactChecks();report={}
    try:
        fixture,routing,golden=verify_inputs(repo,ck);report=audit_session(repo,directory,fixture,routing,ck)
        report['golden_validation']=golden
    except Exception as exc:ck.require(False,'exception/'+type(exc).__name__+': '+str(exc))
    report.update(audit_status='passed' if not ck.errors else 'failed',checks=ck.count,errors=ck.errors,
        auditor_sha256=sha(Path(__file__)),independent_feedback_helper_sha256=FEEDBACK_AUDIT_SHA,
        independent_mathematical_helper_sha256=sha(Path(__file__).with_name('audit_cifar_memristor.py')),hardware_io=False)
    output.mkdir(parents=True,exist_ok=False)
    (output/'audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    (output/'REPORT.md').write_text(f"# Независимый аудит полного диагностического CIFAR\n\n{report['audit_status']}; проверок {ck.count}; ошибок {len(ck.errors)}.\n\n"
        f"Полнота сбора: {report.get('acquisition_complete')}. Физическое качество весов: {report.get('physical_weight_quality_pass')}.\n\n"
        'Фиксированный масштаб применяется к свежим ADC после C; fitted alpha используется только как диагностика. '
        'Сумма тайлов предшествует обученному BNTT и единственному LIF. Эталон обученных весов и W_eff отделены от измеренной ветви. '
        'PASS аудита не означает точность сети.\n',encoding='utf-8')
    print(json.dumps(dict(status=report['audit_status'],checks=ck.count,errors=ck.errors[:15],output=args.output)))
    return int(bool(ck.errors))


if __name__=='__main__':raise SystemExit(main())
