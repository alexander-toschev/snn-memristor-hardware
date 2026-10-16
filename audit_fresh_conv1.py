"""Independent offline fresh300 replay. No collector, driver or native imports."""
import argparse
from collections import Counter
import copy
from configparser import ConfigParser
from datetime import datetime
import hashlib
import importlib
import json
import math
from pathlib import Path
import statistics as st
import subprocess
import sys

HELPER_SHA='09d8ac17e57e8a2df12c26d12fe187c52ceeb25f007ed93baeebe67262b4c92f'
FINAL_PINS={
    'fresh_conv1.py':'031150dbb5cfe285e5c3f8edc18757978f145ecb0b8528126ca4957371d60cfb',
    'fresh_conv1_plan.py':'465d9520acea8aecbb175e29aa18719b1787d17b6e4a54ba8c5577648a877aab',
    'configs/fresh_conv1.json':'5b47aff5d26245f8ab8aa742473c5b097d7722fe929d7c8eeaa558c935260922',
    'weights/fresh_conv1_route.json':'f3bc1279ec53ef76f364c2cb02ecbc0b32b2f1329bba869cf308e0edf91377f9',
    'weights/fresh_conv1_capacity.json':'f1c109655557e8186d7356250c80194facd7f2e5ab209fdacfe5ab0cb38c2d72',
    'experiment.py':'2e57a87e6d7197ff9c8ac13624b9e112c53c8a1a4345db5099ce76f5a927043a',
    'cifar_feedback_adaptive.py':'a46eb7b94891d1adf48d10f126227e10ecfa696aba61a98f293199a8dcb9f2ba',
    'experiments/unn_rev3_20261014/paths.py':'2c841b4a446a767595c68aad243abd966cee94ffd91f74c3c2823998fdc9ec1c',
}
PROTOCOL='fresh300-jit-capacity-fixed-gain-v1'
SIDES=('positive_cell','negative_cell')
A=M=F=None

def require(v,s):
    if not v:raise ValueError(s)

def initialize(repo):
    global A,M,F
    p=repo/'audit_cifar_feedback_adaptive.py'
    require(hashlib.sha256(p.read_bytes()).hexdigest()==HELPER_SHA,'Frozen independent helper changed')
    sys.path.insert(0,str(repo))
    A=importlib.import_module('audit_cifar_feedback_adaptive')
    M=importlib.import_module('audit_cifar_memristor');F=importlib.import_module('audit_cifar_feedback')
    require(Path(A.__file__).resolve()==p.resolve(),'Wrong independent helper location')

def quiet(v):return len(v)==5 and abs(st.mean(v))<=2*M.output_voltage(1) and st.stdev(v)<=2*M.output_voltage(1)

def order(slots):
    """Stable precedence for the one measured WL3/BL3 -> WL3/BL8 interaction."""
    ids=sorted(s['tap_index'] for s in slots);by={s['tap_index']:s for s in slots}
    predecessor=set();successor=set()
    for tap,s in by.items():
        side=SIDES[int(s['trained_weight']<0)]
        if any(c[side]==[3,3] for c in s['candidate_pairs']):predecessor.add(tap)
        if any(c[k]==[3,8] for c in s['candidate_pairs'] for k in SIDES):successor.add(tap)
    edges={(x,y) for x in predecessor for y in successor if x!=y};answer=[]
    while ids:
        ready=[t for t in ids if not any(y==t and x in ids for x,y in edges)]
        require(ready,'Programming precedence cycle');t=ready[0];answer.append(by[t]);ids.remove(t)
    return answer

def replay_reset_pair(r,slot,meta,ledgers):
    start=r.position;states={};gates=[]
    def advance(state,md,forced=False):
        while True:
            if not forced and abs(state['last']['voltage_v'])<=2*M.output_voltage(1):
                confirmation=state['confirmations'];state['confirmations']+=1
                g=r.reads('jit_reset_confirm_mode7',state['cell'],1,confirmation=confirmation,**md)[0]
                values=[F.triple(r,state['cell'],'jit_reset_confirm',confirmation=confirmation,confirmation_repeat=k,**md) for k in range(5)]
                v=[x['voltage_v'] for x in values];state['eligible']=quiet(v);state['last']=values[-1]
                ledgers['reset_confirmations'].append(dict(**md,cell=state['cell'],confirmation=confirmation,
                    next_step=state['next_step'],guard_task_id=g['task_id'],observations=values,
                    mean_v=st.mean(v),sd_v=st.stdev(v),quiet_limit_v=2*M.output_voltage(1),eligible=state['eligible']))
                if state['eligible']:return
            state['eligible']=False
            if state['next_step']==55:return
            step=state['next_step'];state['next_step']+=1
            write=r.take('jit_reset',state['cell'],round(.3+.05*step,3),reset_step=step,**md)
            r.ck.equal(write['command_si']['sign'],0,'reset/native_sign')
            state['last']=F.triple(r,state['cell'],'jit_reset_feedback',reset_step=step,**md)
            ledgers['reset_feedback'].append(dict(**md,cell=state['cell'],step=step,write_task_id=write['task_id'],measurement=state['last']))
            forced=False
    for side in SIDES:
        md=dict(**meta,side=side);g=r.reads('jit_reset_initial_mode7',slot[side],1,**md)[0]
        initial=F.triple(r,slot[side],'jit_reset_initial',**md)
        state=dict(cell=slot[side],next_step=0,confirmations=0,initial_guard_task_id=g['task_id'],last=initial,eligible=False)
        states[side]=state;advance(state,md)
    eligible=False
    for round_index in range(3):
        md=dict(**meta,joint_round=round_index)
        guards=[r.reads('jit_joint_gate_mode7',slot[side],1,guard_side=side,**md)[0]['task_id'] for side in SIDES]
        samples=[]
        for k in range(5):
            values,rows=A.replay_block(r,[slot],[slot],'jit_joint_gate',dict(**md,bracket_repeat=k))
            samples.append(dict(row_sums_v=values,measured_rows=rows,signed_voltage_v=values[0]-values[1]))
        flags={s:quiet([v['row_sums_v'][j] for v in samples]) for j,s in enumerate(SIDES)}
        gate=dict(**md,phase='jit_joint_gate',guard_task_ids=guards,samples=samples,
            quiet_by_side=flags,both_legs_quiet=all(flags.values()),quiet_limit_v=2*M.output_voltage(1))
        gates.append(gate);ledgers['joint_gates'].append(gate)
        if gate['both_legs_quiet']:eligible=True;break
        if round_index==2:break
        progressed=False
        for side in SIDES:
            if not flags[side] and states[side]['next_step']<55:
                advance(states[side],dict(**meta,side=side),True);progressed=True
        if not progressed:break
    count=F.counts_between(r.events,start,r.position)
    result=dict(**meta,positive_cell=slot[SIDES[0]],negative_cell=slot[SIDES[1]],status='completed',eligible=eligible,
        joint_gates=gates,legs=states,stop='fresh_joint_quiet' if eligible else 'bounded_quiet_failure',
        impact_attempts=count['impact_attempts'],programming_pulse_attempts=count['programming_pulse_attempts'])
    r.ck.require(count['impact_attempts']<=2336 and count['programming_pulse_attempts']<=110,'reset/attempt_bound')
    ledgers['reset_pairs'].append(result);return result

def replay_program(r,slot,meta,gain,ledgers):
    start=r.position;target=gain*slot['trained_weight'];tol=max(.5*M.output_voltage(1),.05*abs(target))
    direction=1 if slot['trained_weight']>=0 else -1;side=SIDES[int(direction<0)]
    guards=[F.clean(r.reads('pair_guard_refresh',slot[s],1,side=s,**meta)[0]) for s in SIDES]
    values,initial=A.replay_block(r,[slot],[slot],'pair_initial_feedback',meta);current=values[0]-values[1];d0=current
    stop='set_ramp_exhausted'
    for step in range(109):
        if abs(current-target)<=tol:stop='feedback_hit';break
        if direction*(current-target)>tol:
            stop='initial_overshoot_no_SET' if step==0 else 'feedback_overshoot';break
        write=r.take('pair_set',slot[side],round(.3+.025*step,3),step=step,**meta)
        r.ck.equal(write['command_si']['sign'],1,'program/native_sign')
        values,rows=A.replay_block(r,[slot],[slot],'pair_set_feedback',dict(**meta,step=step));current=values[0]-values[1]
        ledgers['set_feedback'].append(dict(**meta,step=step,write_task_id=write['task_id'],measured_rows=rows,signed_voltage_v=current))
    if abs(current-target)<=tol:stop='feedback_hit'
    elif direction*(current-target)>tol and stop!='initial_overshoot_no_SET':stop='feedback_overshoot'
    postreads=[[F.clean(x) for x in r.reads('pair_post_mode7',slot[s],5,side=s,**meta)] for s in SIDES]
    samples=[]
    for k in range(5):
        values,rows=A.replay_block(r,[slot],[slot],'pair_post_feedback',dict(**meta,repeat=k))
        samples.append(dict(measured_rows=rows,signed_voltage_v=values[0]-values[1]))
    counts=F.counts_between(r.events,start,r.position)
    result=dict(**meta,tap=slot['tap_index'],trained_weight=slot['trained_weight'],positive_cell=slot[SIDES[0]],negative_cell=slot[SIDES[1]],
        target_signed_v=target,active_set_side=side,feedback_stop_tolerance_v=tol,status='completed',guard_refresh=guards,
        initial_mvm=initial,initial_signed_v=d0,post_mode7=postreads,stop=stop,endpoint_signed_v=current,post_feedback=samples,
        post_mean_signed_v=st.mean(s['signed_voltage_v'] for s in samples),post_sd_signed_v=st.stdev(s['signed_voltage_v'] for s in samples),
        impact_attempts=counts['impact_attempts'],programming_pulse_attempts=counts['programming_pulse_attempts'])
    r.ck.require(counts['impact_attempts']<=811 and counts['programming_pulse_attempts']<=109,'program/bound')
    if stop=='initial_overshoot_no_SET':r.ck.equal(counts['programming_pulse_attempts'],0,'program/no_opposite_leg')
    ledgers['pair_programming'].append(result);return result

def fidelity(weights,rows,gain):
    require(gain in (.25,.03),'Unsupported fixed gain');require(len(rows)==5 and all(len(v)==len(weights) for v in rows),'Incomplete physical fidelity')
    means=[st.mean(r[i] for r in rows) for i in range(len(weights))];sd=[st.stdev(r[i] for r in rows) for i in range(len(weights))]
    adc=M.output_voltage(1);normal=math.sqrt(st.mean(w*w for w in weights));errors=[v/gain-w for v,w in zip(means,weights)]
    significant=[i for i,w in enumerate(weights) if abs(w)>2*adc/gain];resolved=[abs(v)>2*(adc+s) for v,s in zip(means,sd)]
    wrong=[i for i in significant if resolved[i] and means[i]*weights[i]<0];unresolved=[i for i in significant if not resolved[i]]
    uncertainties=[math.hypot(s,adc)/gain for s in sd];unrmse=math.sqrt(st.mean(u*u for u in uncertainties))/normal
    rmse=math.sqrt(st.mean(e*e for e in errors))
    return dict(target_gain_v_per_weight=gain,raw_measured_weight=[v/gain for v in means],raw_target_errors=errors,
        raw_target_rmse=rmse,raw_target_nrmse=rmse/normal,one_adc_weight_resolution=adc/gain,quantization_floor_not_reduced_by_averaging=True,
        observed_weight_sd=[s/gain for s in sd],resolved_sign_error_slots=wrong,target_significant_slots=significant,
        measured_sign_unresolved_significant_slots=unresolved,unresolved_quantization_slots=[i for i in range(len(weights)) if i not in significant],
        engineering_weight_uncertainty=uncertainties,uncertainty_nrms=unrmse,uncertainty_is_worstcase_bound=False,
        physical_fidelity_pass=rmse/normal<=.20 and unrmse<=.20 and not wrong and not unresolved,uses_fitted_alpha_or_matrix=False)

def fixed_cal(weights,rows,gain):
    fitted=M.calibration(weights,rows,5);means=fitted['measured_difference_mean_v'];eff=[x/gain for x in means]
    residuals=[a-b for a,b in zip(eff,weights)]
    return dict(decoder_gain_v_per_weight=gain,decoder_gain_is_fixed_before_acquisition=True,fitted_alpha_diagnostic=fitted,decode_usable=True,
        decode_usable_means_finite_arithmetic_not_weight_quality=True,effective_weights=eff,weight_residuals=residuals,
        weight_rmse=math.sqrt(st.mean(x*x for x in residuals)),measured_difference_mean_v=means,calibration_uses_cifar=False)

def measurement_tail(directory,r,active,filt,fixture,meta,gain):
    saved=A.load(directory/'tile.json');ck=r.ck
    def reads(phase):return [[F.clean(x) for x in r.reads(phase,s[side],3,tap=s['tap_index'],side=side,**meta)] for s in active for side in SIDES]
    ck.equal(saved['pre_readback'],reads('tile_pre_readback'),'tail/pre_reads')
    def anchors(phase):
        return [[dict(zero=r.mvm(phase,active[0][side],False,repeat=k,side=side,role='zero',**meta),
            singles=[r.mvm(phase,s[side],True,repeat=k,side=side,role='single',tap=s['tap_index'],**meta) for s in active]) for side in SIDES] for k in range(5)]
    before=anchors('calibration');ck.equal(A.load(directory/'calibration_anchors.json'),before,'tail/fresh_anchors')
    matrix=A.audit_adaptive_matrix(directory,r,active,before,meta,ck);ck.equal(saved['matrix'],matrix,'tail/matrix')
    def differences(samples,correct=False):
        result=[]
        for row in samples:
            pairs=[[row[j]['singles'][i]['voltage_v']-row[j]['zero']['voltage_v'] for j in range(2)] for i in range(len(active))]
            if correct:pairs=[M.multiply(matrix['matrix'],x) for x in pairs]
            result.append([x[0]-x[1] for x in pairs])
        return result
    weights=[s['trained_weight'] for s in active];cal=fixed_cal(weights,differences(before,True),gain)
    freeze=r.events[r.position];r.position+=1;nextid=r.events[r.position]['task_id']
    ck.equal(freeze,dict(event='calibration_frozen',**meta,before_task_id=nextid,
        sha256=A.sha(directory/'frozen_calibration.json'),matrix_sha256=A.sha(directory/'frozen_matrix.json')),'tail/freeze_before_CIFAR')
    cal.update(frozen_before_task_id=nextid,trained_weights=weights,wmax_per_filter=max(abs(s['trained_weight']) for t in filt['tiles'] for s in t['slots']))
    ck.equal(saved['calibration'],cal,'tail/fixed_calibration',1e-9);ck.equal(A.load(directory/'frozen_calibration.json'),cal,'tail/cal_file',1e-9)
    partials=[]
    for ci,case in enumerate(fixture['cases']):
        for repeat in range(5):
            for t,spikes in enumerate(case['spikes_t12x75']):
                selected=[s for s in active if spikes[s['tap_index']]]
                md=dict(**meta,case=ci,original_test_index=case['original_test_index'],repeat=repeat,t=t)
                values,rows=A.replay_block(r,active,selected,'cifar',md);corrected=M.multiply(matrix['matrix'],values);dv=corrected[0]-corrected[1]
                partials.append(dict(**md,active_taps=[s['tap_index'] for s in selected],measured_rows=rows,raw_row_sums_v=values,
                    corrected_row_sums_v=corrected,raw_signed_voltage_v=values[0]-values[1],signed_voltage_v=dv,trained_unit_contribution=dv/gain))
    ck.equal(saved['partials'],partials,'tail/fresh_CIFAR',1e-9);ck.equal(A.ledger(directory/'partials.jsonl'),partials,'tail/CIFAR_ledger',1e-9)
    final=anchors('final_anchors');ck.equal(saved['final_anchors'],final,'tail/final_anchors')
    f0=fidelity(weights,differences(before),gain);f1=fidelity(weights,differences(final),gain)
    ck.equal(saved['physical_fidelity'],f0,'tail/physical_before');ck.equal(saved['final_physical_fidelity'],f1,'tail/physical_after')
    quality=f0['physical_fidelity_pass'] and f1['physical_fidelity_pass'];ck.equal(saved['physical_weight_quality_pass'],quality,'tail/quality')
    ck.equal(saved['post_readback'],reads('tile_post_readback'),'tail/post_reads')
    return dict(calibration=cal,matrix=matrix,partials=partials,physical_fidelity=f0,final_physical_fidelity=f1,physical_weight_quality_pass=quality)

def audit_tile(directory,filt,tile,fixture,plan,source,sid,ck):
    saved=A.load(directory/'tile.json');events,counts=A.read_raw(directory/'raw.jsonl',source,ck,sid);r=A.Replay(events,ck)
    require(counts['programming_skipped_current']==0,'Current-stop session unsupported, fail closed')
    ledgers={k:[] for k in ('reset_confirmations','reset_feedback','joint_gates','reset_pairs','set_feedback','pair_programming')}
    meta=dict(filter_id=filt['filter_id'],tile_id=tile['tile_id']);pairs=[];resets=[];programming=[];actual={};failed=None
    for original in order(tile['slots']):
        chosen=None
        for attempt,candidate in enumerate(original['candidate_pairs']):
            slot={**original,**candidate};md=dict(**meta,pair_tap=slot['tap_index'],candidate_attempt=attempt)
            reset=replay_reset_pair(r,slot,md,ledgers);resets.append(reset)
            if reset['eligible']:chosen=slot;break
        if chosen is None:failed=original['tap_index'];break
        pair=replay_program(r,chosen,md,.25 if source=='hardware' else .03,ledgers);pairs.append(pair);actual[chosen['tap_index']]=chosen
        for side in SIDES:programming.append(dict(cell=chosen[side],tap=chosen['tap_index'],side=side,
            participated_in_SET=side==pair['active_set_side'] and pair['programming_pulse_attempts']>0))
    for name,expected in ledgers.items():ck.equal(A.ledger(directory/(name+'.jsonl')),expected,'ledger/'+name)
    ck.equal(saved,dict(**meta,pairs=pairs,reset_attempts=resets,programming=programming,no_imported_measurements=True,**counts),'tile/common')
    if failed is not None:
        ck.equal(saved['technical_completed'],False,'tile/quietfailure_not_complete');ck.equal(saved['partials'],[],'tile/no_fake_partials')
        ck.require('QuietRouteExhausted' in saved.get('error',''),'tile/quietfailure_reason')
        ck.equal(r.position,len(events),'tile/no_commands_after_failure')
        return dict(**meta,technical_completed=False,failed_tap=failed,counters=counts,reset_attempts=resets,pairs=pairs,partials=[])
    active=[actual[s['tap_index']] for s in tile['slots']];ck.equal(saved['effective_route'],active,'tile/chosen_route')
    freeze=events[r.position];r.position+=1;nextid=events[r.position]['task_id']
    ck.equal(A.load(directory/'frozen_route.json'),dict(slots=active,planned_slots=tile['slots'],programming_tap_order=[p['tap'] for p in pairs],frozen_before_task_id=nextid),'tile/route_file')
    ck.equal(freeze,dict(event='route_frozen',**meta,sha256=A.sha(directory/'frozen_route.json'),before_task_id=nextid),'tile/route_freeze')
    tail=measurement_tail(directory,r,active,filt,fixture,meta,.25 if source=='hardware' else .03)
    ck.equal(r.position,len(events),'tile/exact_raw_consumption');ck.equal(saved,dict(status='completed',technical_completed=True,decode_usable=True),'tile/completion')
    return dict(**meta,technical_completed=True,counters=counts,reset_attempts=resets,pairs=pairs,effective_route=active,**tail)

def validate_route(route,original,evidence,ck):
    truth={(f['filter_id'],s['tap_index']):s['trained_weight'] for f in original['filters'] for t in f['tiles'] for s in t['slots'] if not s['padding']}
    cells={tuple(c['cell']):c for c in evidence['cells']};seen=set();primary=set();attempts=0
    ck.equal(route['trained_coefficients'],300,'route/declared300')
    for f in route['filters']:
        ordered=[];ck.equal(f['wl_pair'],[2*f['filter_id'],2*f['filter_id']+1],'route/WLpair')
        ck.equal([t['tile_id'] for t in f['tiles']],list(range(len(f['tiles']))),'route/tileIDs')
        for tile in f['tiles']:
            ck.require(1<=len(tile['slots'])<=7,'route/width');reserved=set()
            for s in tile['slots']:
                key=f['filter_id'],s['tap_index'];ck.require(key in truth and key not in seen,'route/unique_original_target');seen.add(key)
                ck.equal(s['trained_weight'],truth[key],'route/unchanged_weight');ck.require(not s['padding'],'route/no_padding')
                ordered.append(s['tap_index']);side=SIDES[int(s['trained_weight']<0)];other=SIDES[1-SIDES.index(side)]
                require(1<=len(s['candidate_pairs'])<=2,'Unbounded candidates');attempts+=len(s['candidate_pairs'])
                for c in s['candidate_pairs']:
                    for j,k in enumerate(SIDES):
                        addr=tuple(c[k]);require(addr in cells and addr not in reserved,'Unknown/colliding candidate address');reserved.add(addr)
                        ck.equal(addr[0],2*f['filter_id']+j,'route/fixed_WL')
                    active=cells[tuple(c[side])];ck.require(active['active_candidate_allowed'],'route/active_evidence')
                    ck.require(.25*abs(s['trained_weight'])<=active['screening_amplitude_v'],'route/active_amplitude_screen')
                    ck.require(cells[tuple(c[other])]['inactive_candidate_allowed_only_after_fresh_joint_quiet'],'route/inactive_eligibility')
                ck.equal({k:s[k] for k in SIDES},s['candidate_pairs'][0],'route/primary_first')
                primary.update(tuple(s[k]) for k in SIDES)
            order(tile['slots'])
        ck.equal(ordered,list(range(75)),'route/exact75_in_order')
    ck.equal(seen,set(truth),'route/exact300_coverage');ck.equal(route['candidate_pair_attempts'],attempts,'route/attempt_count')
    ck.equal(route['primary_distinct_addresses'],[list(x) for x in sorted(primary)],'route/primary_union')
    ck.equal(route['primary_distinct_count'],len(primary),'route/primary_count')
    return dict(trained_coefficients=len(seen),candidate_pair_attempts=attempts,primary_distinct_count=len(primary))

def bounds(route,fixture):
    selected=[(f,t) for f in route['filters'] for t in f['tiles']]
    n=sum(len(t['slots']) for _,t in selected);attempts=sum(len(s['candidate_pairs']) for _,t in selected for s in t['slots'])
    tail=sum(A.measurement_tail(t['slots'],fixture) for _,t in selected)
    return dict(tiles=len(selected),trained_coefficients=n,candidate_pair_attempts=attempts,
        maximum_impacts=attempts*2336+n*811+tail,maximum_programming_pulses=attempts*110+n*109,measurement_tail_impacts=tail)

def verify_manifest(repo,directory,ck):
    require(bool(FINAL_PINS) and PROTOCOL is not None,'Actual-session audit is not frozen')
    m=A.load(directory/'manifest.json');plan=A.load(directory/'protocol.json');route=A.load(directory/'planned_route.json')
    require(m['source'] in ('hardware','simulator'),'Unknown source')
    provenance=M.audit_git_status(m['invocation']['git'],m['source_sha256'],ck)
    for name,digest in m['result_sha256'].items():ck.equal(A.sha(A.local(directory,name)),digest,'result/hash/'+name)
    actual={p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file() and p.name!='manifest.json'}
    ck.equal(sorted(actual),sorted(m['result_sha256']),'result/exact_inventory')
    source_blobs={}
    for name,digest in m['source_sha256'].items():
        if name in M.PRIVATE:ck.equal(digest,M.PRIVATE[name],'source/private/'+name)
        else:
            proc=subprocess.run(['git','-C',str(repo),'show',m['invocation']['git']['commit']+':'+name],capture_output=True)
            ck.equal(proc.returncode,0,'source/Git_exists/'+name);ck.equal(hashlib.sha256(proc.stdout).hexdigest(),digest,'source/Git_SHA/'+name)
            source_blobs[name]=proc.stdout
    for name,digest in {**M.PINS,**A.DIAGNOSTIC_PINS,**FINAL_PINS}.items():ck.equal(m['source_sha256'].get(name),digest,'source/frozen/'+name)
    for name,digest in M.PRIVATE.items():
        if m['source']=='hardware' or '/MemriCORE/' not in name:ck.equal(m['source_sha256'].get(name),digest,'source/native_required/'+name)
    config_key='configs/fresh_conv1.json'
    route_key='weights/fresh_conv1_route.json'
    evidence_key='weights/fresh_conv1_capacity.json'
    require(plan==json.loads(source_blobs[config_key]),'Protocol differs from exact pinned source JSON')
    require(route==json.loads(source_blobs[route_key]),'Planned route differs from pinned JSON')
    evidence=json.loads(source_blobs[evidence_key]);ck.equal(plan['capacity_evidence_sha256'],FINAL_PINS[evidence_key],'capacity/pin')
    for name,digest in evidence['historical_leaf_sha256'].items():ck.equal(A.sha(A.local(repo,name)),digest,'capacity/leaf/'+name)
    for name,digest in evidence['source_manifest_sha256'].items():ck.equal(A.sha(A.local(repo,name)/'manifest.json'),digest,'capacity/manifest/'+name)
    ck.equal(plan,dict(protocol_id=PROTOCOL,review_pending=False,target_gain_v_per_weight={'hardware':.25,'simulator':.03},
        max_joint_repair_rounds=2,max_candidate_pairs_per_tap=2,max_microtile_width=7,
        reset_ramp_v=[round(.3+.05*i,3) for i in range(55)],set_ramp_v=[round(.3+.025*i,3) for i in range(109)],
        feedback_relative_tolerance=.05,feedback_absolute_adc_tolerance=.5,cutoff_utc='2026-10-16T18:50:00+00:00'),
        'protocol/frozen_contract')
    ck.equal(m,dict(protocol_id=PROTOCOL,no_imported_measurements=True,hardware_access=m['source']=='hardware',
        fixed_decoder_gain_v_per_weight=.25 if m['source']=='hardware' else .03,physical_pulse_width_s=None),'manifest/contract')
    board=ConfigParser();board.read(directory/'board.ini',encoding='utf-8')
    for k,v in M.BOARD.items():ck.equal(board.getfloat('board',k),v,'INI/'+k)
    start=A.load(directory/'start_manifest.json')
    for k in ('session_id','source','source_sha256','protocol_id','invocation','cutoff_utc'):ck.equal(m[k],start[k],'start/'+k)
    ck.require(datetime.fromisoformat(m['started_utc'])<datetime.fromisoformat(plan['cutoff_utc']),'deadline/start')
    ck.require(datetime.fromisoformat(m['finished_utc'])<datetime.fromisoformat(plan['cutoff_utc']),'deadline/close')
    return m,plan,route,evidence,provenance

def coverage(results):
    sets={k:set() for k in ('reset_probed','joint_quiet','selected','SET_attempted','represented')}
    for r in results:
        for a in r['reset_attempts']:
            for side in SIDES:
                cell=tuple(a[side]);sets['reset_probed'].add(cell)
                if a['eligible']:sets['joint_quiet'].add(cell)
        for p in r['pairs']:
            for side in SIDES:sets['selected'].add(tuple(p[side]))
            if p['programming_pulse_attempts']:sets['SET_attempted'].add(tuple(p[p['active_set_side']]))
        if r['technical_completed']:
            for s in r['effective_route']:
                for side in SIDES:sets['represented'].add(tuple(s[side]))
    return {k:dict(count=len(s),addresses=[list(x) for x in sorted(s)]) for k,s in sets.items()}

def audit_session(repo,directory,fixture,original,ck):
    m,plan,route,evidence,provenance=verify_manifest(repo,directory,ck)
    routecheck=validate_route(route,original,evidence,ck);bound=bounds(route,fixture)
    ck.equal(m,bound,'manifest/independent_budget');ck.equal(plan,{k:bound[k] for k in ('maximum_impacts','maximum_programming_pulses')},'plan/independent_budget')
    results=[];failed=False
    for f in route['filters']:
        if failed:break
        for tile in f['tiles']:
            folder=directory/'tiles'/f"f{f['filter_id']}_t{tile['tile_id']}"
            result=audit_tile(folder,f,tile,fixture,plan,m['source'],m['session_id'],ck);results.append(result)
            if not result['technical_completed']:failed=True;break
    ck.equal(sorted(str(p.relative_to(directory)) for p in (directory/'tiles').iterdir() if p.is_dir()),
        sorted(str(Path('tiles')/f"f{r['filter_id']}_t{r['tile_id']}") for r in results),'session/exact_attempted_tiles')
    total={k:sum(r['counters'][k] for r in results) for k in ('impact_attempts','programming_pulse_attempts','programming_decisions','programming_skipped_current')}
    complete=not failed;trajectories=[];quality=[];cells=coverage(results)
    ck.equal(m,dict(total_impacts=total['impact_attempts'],total_pulses=total['programming_pulse_attempts'],
        completed_microtiles=sum(r['technical_completed'] for r in results)),'manifest/all_attempts')
    if complete:
        trajectories,quality=M.aggregate_independent(results,route,fixture,ck)
        saved=A.load(directory/'summary.json');ck.equal(saved['trajectories'],trajectories,'full/independent_sum75_BN_LIF',1e-8)
        ck.equal(len(trajectories)*12,2400,'full/exact2400')
        ck.equal(saved,dict(status='completed',technical_completed=True,acquisition_complete=True,decoded_conv_bn_lif_complete=True,
            fresh_trained_coefficients=300,completed_microtiles=len(results),no_imported_measurements=True,
            physical_weight_quality_pass=all(r['physical_weight_quality_pass'] for r in results),
            represented_addresses=cells['represented']['addresses'],represented_address_count=cells['represented']['count'],
            address_coverage_is_not_weight_qualification=True,total_impacts=total['impact_attempts'],total_pulses=total['programming_pulse_attempts'],
            actual_branch_uses_software_dot_product=False,full_network_accuracy_claim=False),'summary/flags')
        ck.equal(m,dict(status='completed',acquisition_complete=True),'manifest/complete')
    else:
        ck.equal(m['status'],'failed','manifest/quietfailure');ck.require(not m.get('acquisition_complete',False),'manifest/no_false_completion')
        ck.require(not (directory/'summary.json').exists(),'failure/no_summary_or_old_outputs')
    ck.require(total['impact_attempts']<=bound['maximum_impacts'] and total['programming_pulse_attempts']<=bound['maximum_programming_pulses'],'budget/global')
    return dict(session_id=m['session_id'],source=m['source'],manifest_sha256=A.sha(directory/'manifest.json'),provenance=provenance,
        acquisition_complete=complete,full_time_outputs=len(trajectories)*12,counters=total,planned_coverage=routecheck,
        actual_coverage=cells,quality=quality,tiles=[{k:v for k,v in r.items() if k not in ('partials','matrix','pairs','reset_attempts')} for r in results])

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--repo',default='.');p.add_argument('--session',required=True);p.add_argument('--output',required=True)
    args=p.parse_args(argv);repo=Path(args.repo).resolve();initialize(repo);ck=A.CompactChecks();report={}
    directory=A.local(repo,args.session);output=A.local(repo,args.output);require(not output.is_relative_to(directory),'Output modifies source')
    try:
        fixture,original,golden=M.verify_inputs(repo,ck)
        report=audit_session(repo,directory,fixture,original,ck);report['golden_validation']=golden
    except Exception as exc:ck.require(False,type(exc).__name__+': '+str(exc))
    report.update(audit_status='passed' if not ck.errors else 'failed',checks=ck.count,errors=ck.errors,
        auditor_sha256=A.sha(Path(__file__)),helper_sha256=HELPER_SHA,hardware_io=False,
        limitations=['Consistency PASS is not weight fidelity or full classifier accuracy.',
            'Quiet exhaustion is replayed; transport/current/deadline interruptions fail closed.',
            'Historical amplitude screen does not establish a continuous programmable range.'])
    output.mkdir(parents=True,exist_ok=False)
    (output/'audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=report['audit_status'],checks=ck.count,errors=ck.errors[:15])))
    return int(bool(ck.errors))

if __name__=='__main__':raise SystemExit(main())
