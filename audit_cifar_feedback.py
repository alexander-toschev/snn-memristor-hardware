"""Independent offline mode10-feedback audit; no collector or device imports."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
from configparser import ConfigParser
from collections import Counter

HELPER_SHA='4d9ff520d22a5142ef040c2c50f3c0d61f1558579cbf0411803f606865110a94'
FEEDBACK_PINS={
    'cifar_feedback.py':'0cd7b6dbbb789afa6a9b3fe71e1f50a9dd9c66715bfd1675112b69f9eb8cb5b4',
    'configs/cifar_feedback.json':'c0978558dda13ebb28ca25e93c52237802907a7b6d2ce3c552374e2b71e5350c',
    'experiment.py':'f7b7f3ff20c288f9ab3077832a6142285c75f31046fc0d718179e83dd80febd0',
    'cifar_memristor.py':'2410b67088f39b6d415ed560bff6d06aeae6b8712ab2906bfb69d7191773eefe',
    'cifar_memristor_analysis.py':'8522b922d9d6cf7d98e4e6f6f3c578453cc3d50d688af1ca4e322c67217785de',
    'configs/cifar_memristor.json':'be859fd8b07c8c7fac3caadcd6eb44fc0ae572ccf99ba96b28e7702d35bb966d',
}
if hashlib.sha256(Path(__file__).with_name('audit_cifar_memristor.py').read_bytes()).hexdigest()!=HELPER_SHA:
    raise RuntimeError('Independent audit helper fingerprint changed')
from audit_cifar_memristor import (Checks,Replay,sha,local,load,output_voltage,resistance,dac,
    verify_inputs,calibration,pair_values,multiply,audit_matrix,replay_block,aggregate_independent,
    audit_git_status,PINS,PRIVATE,BOARD)


class CompactChecks(Checks):
    def equal(self,a,b,label,tolerance=1e-11):
        def same(a,b):
            if isinstance(b,dict):return isinstance(a,dict) and all(k in a and same(a[k],v) for k,v in b.items())
            if isinstance(b,(list,tuple)):return isinstance(a,(list,tuple)) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
            if type(b) in (float,int):return type(a) in (float,int) and math.isfinite(a) and math.isclose(a,b,rel_tol=tolerance,abs_tol=tolerance)
            return a==b
        self.require(same(a,b),label)


def clean(row):return {k:v for k,v in row.items() if k not in ('event','latency_s')}
def ledger(path):return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()] if path.is_file() else []


def fidelity(weights,observations,gain):
    if gain not in (1.,.03) or len(observations)!=5 or any(len(r)!=len(weights) for r in observations):
        raise ValueError('Unreviewed scale or incomplete fidelity samples')
    means=[statistics.mean(r[i] for r in observations) for i in range(len(weights))]
    sds=[statistics.stdev(r[i] for r in observations) for i in range(len(weights))]
    unit=output_voltage(1);normal=math.sqrt(statistics.mean(w*w for w in weights))
    errors=[m/gain-w for m,w in zip(means,weights)];rmse=math.sqrt(statistics.mean(e*e for e in errors))
    significant=[i for i,w in enumerate(weights) if abs(w)>2*unit/gain]
    resolved=[abs(m)>2*(unit+s) for m,s in zip(means,sds)]
    wrong=[i for i in significant if resolved[i] and means[i]*weights[i]<0]
    unresolved=[i for i in significant if not resolved[i]]
    uncertainties=[math.hypot(sd,unit)/gain for sd in sds]
    unrmse=math.sqrt(statistics.mean(u*u for u in uncertainties))/normal
    return dict(target_gain_v_per_weight=gain,raw_measured_weight=[m/gain for m in means],raw_target_errors=errors,
        raw_target_rmse=rmse,raw_target_nrmse=rmse/normal,one_adc_weight_resolution=unit/gain,
        quantization_floor_not_reduced_by_averaging=True,observed_weight_sd=[sd/gain for sd in sds],
        resolved_sign_error_slots=wrong,target_significant_slots=significant,
        measured_sign_unresolved_significant_slots=unresolved,
        unresolved_quantization_slots=[i for i in range(len(weights)) if i not in significant],
        engineering_weight_uncertainty=uncertainties,uncertainty_nrms=unrmse,uncertainty_is_worstcase_bound=False,
        physical_fidelity_pass=rmse/normal<=.20 and unrmse<=.20 and not wrong and not unresolved,
        uses_fitted_alpha_or_matrix=False)


def pools_for(tile):
    active=[s for s in tile['slots'] if not s['padding']];pools={}
    for side in ('positive_cell','negative_cell'):
        original=[s[side][1] for s in active]
        spares=[b for b in [*range(16),31] if b not in original][:3]
        if len(spares)!=3 or len(set(original))!=len(original):raise ValueError('Candidate pool contract')
        pools[side]=[[active[0][side][0],b] for b in original+spares]
    return pools


def counts_between(events,start,stop):
    rows=events[start:stop]
    pulse=lambda r:r['event']!='skipped' and r.get('command_si',{}).get('mode_flag')==7 and r['command_si']['vol']!=0
    pulses=sum(bool(pulse(r)) for r in rows);skips=sum(r['event']=='skipped' for r in rows)
    return dict(impact_attempts=sum(r['event']=='result' for r in rows),programming_pulse_attempts=pulses,
                programming_decisions=pulses+skips,programming_skipped_current=skips)


def triple(replay,cell,phase,**meta):
    before=replay.mvm(phase,cell,False,role='zero_before',**meta)
    signal=replay.mvm(phase,cell,True,role='signal',**meta)
    after=replay.mvm(phase,cell,False,role='zero_after',**meta)
    return dict(zero_before=before,signal=signal,zero_after=after,
                voltage_v=signal['voltage_v']-(before['voltage_v']+after['voltage_v'])/2)


def reset_replay(replay,cell,meta,feedback_rows):
    start=replay.position;current=replay.reads('candidate_initial',cell,1,**meta)[0]
    initial=triple(replay,cell,'candidate_initial_mvm',**meta);feedback=initial;limit=2*output_voltage(1)
    stop='reset_ramp_exhausted'
    for step in range(55):
        if abs(feedback['voltage_v'])<=limit:stop='mode10_quiet_crossed';break
        row=replay.take('candidate_reset',cell,round(.3+.05*step,3),step=step,**meta)
        replay.ck.equal((row.get('command_si') or row['planned_command_si'])['sign'],0,'reset/sign')
        if row['event']=='skipped':stop='current_guard';break
        current=row;feedback=triple(replay,cell,'candidate_reset_feedback',step=step,**meta)
        feedback_rows.append(dict(**meta,cell=cell,step=step,write_task_id=row['task_id'],measurement=feedback))
    crossed=abs(feedback['voltage_v'])<=limit
    if crossed:stop='mode10_quiet_crossed'
    confirmation=replay.reads('candidate_reset_confirmation',cell,5,**meta)
    observed=[triple(replay,cell,'candidate_reset_confirmation_mvm',repeat=i,**meta) for i in range(5)]
    mean=statistics.mean(r['voltage_v'] for r in observed);sd=statistics.stdev(r['voltage_v'] for r in observed)
    return dict(cell=cell,status='completed',eligibility_scope='bounded_reset_policy_only',initial_mvm=initial,
        reset_stop=stop,reset_crossed=crossed,q_confirmation=[dict(task_id=r['task_id'],q=r['q']) for r in confirmation],
        mvm_confirmation=observed,baseline_mvm_mean_v=mean,baseline_mvm_sd_v=sd,mode10_quiet_limit_v=limit,
        native_q_diagnostic_crossed=current['q']<=.005,eligible=crossed and abs(mean)<=limit and sd<=limit,
        physical_failure_cause='unknown; not inferred from this screening',**counts_between(replay.events,start,replay.position))


def pair_replay(replay,slot,meta,gain,feedback_rows):
    start=replay.position;target=gain*slot['trained_weight'];tol=max(.5*output_voltage(1),.05*abs(target))
    refresh=[clean(replay.reads('pair_guard_refresh',slot[side],1,side=side,**meta)[0]) for side in ('positive_cell','negative_cell')]
    values,initial=replay_block(replay,[slot],[slot],'pair_initial_feedback',meta)
    current=values[0]-values[1];initial_signed=current;direction=1 if target>=current else -1
    side='positive_cell' if direction==1 else 'negative_cell';stop='set_ramp_exhausted'
    for step in range(109):
        if abs(current-target)<=tol:stop='feedback_hit';break
        if direction*(current-target)>tol:stop='feedback_overshoot';break
        row=replay.take('pair_set',slot[side],round(.3+.025*step,3),step=step,target_signed_v=target,**meta)
        replay.ck.equal((row.get('command_si') or row['planned_command_si'])['sign'],1,'pair/sign')
        if row['event']=='skipped':stop='current_guard';break
        values,rows=replay_block(replay,[slot],[slot],'pair_set_feedback',dict(**meta,step=step));current=values[0]-values[1]
        feedback_rows.append(dict(**meta,step=step,write_task_id=row['task_id'],measured_rows=rows,signed_voltage_v=current))
    if abs(current-target)<=tol:stop='feedback_hit'
    elif direction*(current-target)>tol:stop='feedback_overshoot'
    reads=[[clean(r) for r in replay.reads('pair_post_mode7',slot[s],5,side=s,**meta)] for s in ('positive_cell','negative_cell')]
    post=[]
    for repeat in range(5):
        values,rows=replay_block(replay,[slot],[slot],'pair_post_feedback',dict(**meta,repeat=repeat))
        post.append(dict(measured_rows=rows,signed_voltage_v=values[0]-values[1]))
    return dict(tap=slot['tap_index'],trained_weight=slot['trained_weight'],target_signed_v=target,feedback_stop_tolerance_v=tol,
        status='completed',positive_cell=slot['positive_cell'],negative_cell=slot['negative_cell'],guard_refresh=refresh,
        initial_mvm=initial,initial_signed_v=initial_signed,active_set_side=side,stop=stop,endpoint_signed_v=current,
        post_mode7=reads,post_feedback=post,post_mean_signed_v=statistics.mean(r['signed_voltage_v'] for r in post),
        post_sd_signed_v=statistics.stdev(r['signed_voltage_v'] for r in post),
        stop_band_confirmed_all_five=all(abs(r['signed_voltage_v']-target)<=tol for r in post),
        stop_band_does_not_imply_weight_fidelity=True,**counts_between(replay.events,start,replay.position))


def read_raw(path, source, ck, session_id=None):
    events=[]; latest={}; sequence=decisions=pulses=skips=0; previous_time=-math.inf
    with Path(path).open(encoding='utf-8') as stream:
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


def audit_feedback_tile(directory,filt,tile,fixture,plan,source,ck,session_id):
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
        return dict(**meta,technical_completed=False,continuation_eligible=False,counters=counters,
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
                feedback_pair_stop=pair['stop'],participated_in_SET=side==pair['active_set_side'] and pair['programming_pulse_attempts']>0,reset_eligible=True))
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
    cal=calibration([s['trained_weight'] for s in active],differences,plan['alpha_absolute_floor_adc'])
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
                                     signed_voltage_v=difference,trained_unit_contribution=difference/cal['alpha_v_per_weight'] if cal['decode_usable'] else None))
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
    eligible=cal['decode_usable'] and before['physical_fidelity_pass'] and after['physical_fidelity_pass']
    ck.equal(saved['continuation_eligible'],eligible,'fidelity/continuation_gate')
    n=len(active);candidate_count=2*n+6
    ck.require(counters['programming_decisions']<=55*candidate_count+109*n,'feedback/decision_bound')
    ck.require(counters['programming_pulse_attempts']<=counters['programming_decisions'],'feedback/pulse_bound')
    return dict(**meta,technical_completed=True,continuation_eligible=eligible,calibration=cal,matrix=correction,
        effective_route=active,physical_fidelity=before,final_physical_fidelity=after,partials=partials,counters=counters,
        candidate_eligible_counts={side:sum(c['eligible'] for c in candidates if c['side']==side) for side in pools},
        pair_stop_counts=dict(Counter(p['stop'] for p in pairs)))


def expected_counts(selected,fixture):
    n=sum(sum(not s['padding'] for s in t['slots']) for _,t in selected);candidates=2*n+6*len(selected)
    tail=0
    for _,tile in selected:
        active=[s for s in tile['slots'] if not s['padding']];size=len(active)
        masks=[[i] for i in range(size)]+[list(range(size)),list(range(0,size,2)),list(range(1,size,2))]
        masks += [[]]+[[i] for i in range(size)]+[[i,i+1] for i in range(0,size-1,2)]
        tail+=12*size+20*(size+1)+sum(3*(4+2*max(1,len(mask))) for mask in masks)
        for case in fixture['cases']:
            tail+=sum(5*(4+2*max(1,sum(row[s['tap_index']] for s in active))) for row in case['spikes_t12x75'])
    return dict(maximum_impacts=244*candidates+811*n+tail,maximum_programming_pulses=55*candidates+109*n,
                reset_candidates=candidates,trained_coefficients=n,physical_assignments=2*n,tiles=len(selected))


def verify_feedback_manifest(repo,directory,ck):
    m=load(directory/'manifest.json');git=m['invocation']['git'];policy_git=copy.deepcopy(git)
    standalone={'audit_cifar_feedback.py','test_audit_cifar_feedback.py'}
    exempt=[name for name in git['untracked'] if name in standalone]
    for name in exempt:ck.require(name not in m['source_sha256'],'provenance/offline_auditor_not_runtime')
    policy_git['untracked']=[n for n in git['untracked'] if n not in exempt]
    policy_git['status_porcelain']='\n'.join(line for line in git['status_porcelain'].splitlines() if line not in ['?? '+n for n in exempt])
    provenance=audit_git_status(policy_git,m['source_sha256'],ck)
    provenance['additional_known_offline_audit_untracked']=exempt
    ck.require(m['source'] in ('hardware','simulator'),'manifest/source')
    ck.equal(m['hardware_access'],m['source']=='hardware','manifest/hardware_access')
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
    for name,digest in PINS.items():ck.equal(m['source_sha256'].get(name),digest,'source/known_pin/'+name)
    for name,digest in FEEDBACK_PINS.items():ck.equal(m['source_sha256'].get(name),digest,'source/frozen_feedback_pin/'+name)
    config_bytes=subprocess.run(['git','-C',str(repo),'show',git['commit']+':configs/cifar_feedback.json'],capture_output=True,check=True).stdout
    plan=load(directory/'protocol.json');ck.equal(plan,json.loads(config_bytes),'protocol/snapshot_git_config')
    ck.equal(m['protocol_id'],plan['protocol_id'],'protocol/manifest')
    ck.equal(plan['protocol_id'],'cifar-mode10-feedback-v1','protocol/frozen_id')
    ck.equal(plan,dict(target_gain_v_per_weight={'hardware':1.,'simulator':.03},
        reset_ramp_v=[round(.3+.05*i,3) for i in range(55)],set_ramp_v=[round(.3+.025*i,3) for i in range(109)],
        feedback_relative_tolerance=.05,feedback_absolute_adc_tolerance=.5,spares_per_wl=3,
        candidate_bl_pool=[*range(16),31],reset_signal_limit_adc=2,maximum_programming_pulses=74940,
        maximum_programming_decisions=74940,maximum_impacts=580210,pilot_maximum_impacts=25612,
        pilot_maximum_programming_pulses=3396),'protocol/reviewed_feedback_constants')
    ck.equal(plan['physical_fidelity_gate'],dict(review_pending=False,raw_target_nrmse_max=.20,uncertainty_nrms_max=.20,
             resolved_sign_errors_max=0,target_significant_adc=2,
             uncertainty_formula='sqrt(sampleSD(D)^2+ADC_LSB^2)/target_gain; no sqrt(repeats); engineering not worstcase',
             measurement_sign_resolution='abs(meanD)>2ADC+2sampleSD(D)'),'protocol/fidelity_gate')
    old_config=subprocess.run(['git','-C',str(repo),'show',git['commit']+':configs/cifar_memristor.json'],capture_output=True,check=True).stdout
    ck.equal(plan['matrix'],json.loads(old_config)['matrix'],'protocol/unchanged_matrix_contract')
    board=ConfigParser();board.read(directory/'board.ini',encoding='utf-8')
    for key,value in BOARD.items():ck.equal(board.getfloat('board',key),value,'INI/'+key)
    return m,plan,provenance


def audit_session(repo,directory,fixture,routing,ck,allow_parent=True):
    manifest,plan,provenance=verify_feedback_manifest(repo,directory,ck)
    all_tiles=[(f,t) for f in routing['filters'] for t in f['tiles']];results=[]
    if manifest['stage']=='pilot':selected=all_tiles[:1];ck.equal(manifest['parent_pilot'],None,'pilot/no_parent')
    elif manifest['stage']=='remaining' and allow_parent:
        selected=all_tiles[1:];parent=manifest['parent_pilot'];parent_dir=local(repo,parent['session']);pm=load(parent_dir/'manifest.json')
        ck.equal(sha(parent_dir/'manifest.json'),parent['manifest_sha256'],'parent/manifest_sha')
        ck.equal(pm['source_sha256'],manifest['source_sha256'],'parent/source_snapshot')
        ck.equal(pm['source'],manifest['source'],'parent/same_backend')
        ck.equal(pm['result_sha256'],parent['result_sha256'],'parent/result_commitments')
        ck.equal(parent['measured_data_reused'],True,'parent/reused')
        _,results=audit_session(repo,parent_dir,fixture,routing,ck,False)
        ck.require(all(r['continuation_eligible'] for r in results),'parent/physical_fidelity_gate')
        if manifest['source']=='simulator':ck.equal(manifest['array_initial_sha256'],parent['result_sha256']['array.cb'],'parent/persisted_array_copy')
    else:raise ValueError('Unsupported stage or nested parent')
    ck.equal(manifest,expected_counts(selected,fixture),'manifest/recomputed_budget')
    permitted={f"f{f['filter_id']}_t{t['tile_id']}" for f,t in selected}
    actual_folders={p.name for p in (directory/'tiles').iterdir() if p.is_dir()}
    ck.require(actual_folders<=permitted,'session/no_unexpected_tiles')
    if manifest['status']=='completed':ck.equal(sorted(actual_folders),sorted(permitted),'session/all_selected_tiles')
    for filt,tile in selected:
        name=f"tiles/f{filt['filter_id']}_t{tile['tile_id']}";folder=directory/name
        if not folder.exists():
            ck.require(manifest['status']!='completed','session/missing_tile');break
        for required in ('raw.jsonl','tile.json','reset_candidates.jsonl'):
            ck.require(name+'/'+required in manifest['result_sha256'],'manifest/required_result/'+required)
        for artifact in folder.iterdir():
            if artifact.is_file():ck.require(name+'/'+artifact.name in manifest['result_sha256'],'manifest/all_tile_artifacts_hashed/'+artifact.name)
        results.append(audit_feedback_tile(folder,filt,tile,fixture,plan,manifest['source'],ck,manifest['session_id']))
        if not results[-1]['technical_completed']:break
    acquired=manifest['stage']=='remaining' and len(results)==28 and all(r['technical_completed'] for r in results)
    trajectories=[];quality=[]
    if all(r['technical_completed'] for r in results):
        trajectories,quality=aggregate_independent(results,routing,fixture,ck)
        saved=load(directory/'summary.json')
        ck.equal(saved['trajectories'],trajectories,'summary/actual_BN_LIF_and_two_refs',1e-8)
        ck.equal(saved['acquisition_complete'],acquired,'summary/acquisition')
        ck.equal(saved['decoded_conv_bn_lif_complete'],acquired and all(r['calibration']['decode_usable'] for r in results),'summary/decoded')
        ck.equal(saved['continuation_eligible'],all(r['continuation_eligible'] for r in results),'summary/continuation')
    else:ck.equal(manifest['status'],'failed','summary/partial_status_preserved')
    counts={name:sum(r['counters'][name] for r in results) for name in ('impact_attempts','programming_pulse_attempts','programming_decisions','programming_skipped_current')}
    if all(r['technical_completed'] for r in results):
        ck.equal(saved['total_impacts_including_pilot'],counts['impact_attempts'],'summary/total_impacts')
        ck.equal(saved['total_pulses_including_pilot'],counts['programming_pulse_attempts'],'summary/total_pulses')
    bound=expected_counts(all_tiles[:1] if manifest['stage']=='pilot' else all_tiles,fixture)
    ck.require(counts['impact_attempts']<=bound['maximum_impacts'],'session/impact_bound')
    ck.require(counts['programming_decisions']<=bound['maximum_programming_pulses'],'session/decision_bound')
    compact=[]
    for result in results:
        r={k:v for k,v in result.items() if k not in ('partials','matrix','reset_candidates')}
        if 'matrix' in result:r['matrix']={k:result['matrix'][k] for k in ('matrix','selected_model','selection_pass','training_design_rank','training_design_condition')}
        compact.append(r)
    report=dict(session_id=manifest['session_id'],source=manifest['source'],stage=manifest['stage'],acquisition_status=manifest['status'],
        acquisition_complete=acquired,provenance=provenance,counters=counts,tiles=compact,
        partial_points=sum(len(r['partials']) for r in results),full_time_outputs=len(trajectories)*12,quality=quality,
        continuation_eligible=all(r['continuation_eligible'] for r in results),manifest_sha256=sha(directory/'manifest.json'),
        limitations=['Consistency PASS is not physical-fidelity PASS.','Quiet reset is a mode10 screening result, not a proven conductance state.',
                    'Source-specific fixed gain is provisional; HW and SIM target voltages differ.','Sample SD and one-ADC floor are engineering uncertainty, not a confidence interval.'])
    return report,results


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--repo',default='.')
    parser.add_argument('--session',required=True);parser.add_argument('--output',required=True);args=parser.parse_args(argv)
    repo=Path(args.repo).resolve();directory=local(repo,args.session);output=local(repo,args.output)
    if output==directory or output.is_relative_to(directory):raise ValueError('Output must not change acquired session')
    ck=CompactChecks();report={}
    try:
        fixture,routing,golden=verify_inputs(repo,ck);report,_=audit_session(repo,directory,fixture,routing,ck)
        report['golden_validation']=golden
    except Exception as exc:ck.require(False,'exception/'+type(exc).__name__+': '+str(exc))
    report.update(audit_status='passed' if not ck.errors else 'failed',checks=ck.count,errors=ck.errors,
                  auditor_sha256=sha(Path(__file__)),independent_helper_sha256=HELPER_SHA,hardware_io=False)
    output.mkdir(parents=True,exist_ok=False)
    (output/'audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    (output/'REPORT.md').write_text(f"# Независимый feedback-аудит\n\n{report['audit_status']}; проверок {ck.count}; ошибок {len(ck.errors)}.\n\n"
        'Согласованность журнала, полнота сбора и физическое качество весов проверяются раздельно. '
        'Нет импорта коллектора или драйвера; только сохранённые ADC, provenance и арифметика.\n',encoding='utf-8')
    print(json.dumps(dict(status=report['audit_status'],checks=ck.count,errors=ck.errors[:15],output=args.output)))
    return int(bool(ck.errors))


if __name__=='__main__':raise SystemExit(main())
