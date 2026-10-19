"""Independent bounded-range replay derived statically from the frozen full auditor.

No Connector, synthetic footer, scope patch or runtime source mutation. A prefix
is accepted only at a real stage_complete after final_basis. This is a separate
audit contract; it never reports the frozen full48 auditor passed.
Base source SHA: d3cd8bf2cf11512f81aabff0300c5fde391dad240fd9008834150e698910e717.
"""
from array import array
import hashlib
import json
import math
import statistics
import sys
import random

from experiments.cifar_full_case0_20261019.reconstruct import expected_signals


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def same(a,b):
    if a!=b:
        raise ValueError('Independent replay mismatch')


def numerically_same(a,b):
    if isinstance(a,dict) and isinstance(b,dict):
        same(set(a),set(b))
        for k in a:numerically_same(a[k],b[k])
    elif isinstance(a,list) and isinstance(b,list):
        same(len(a),len(b))
        for x,y in zip(a,b):numerically_same(x,y)
    elif type(a) is float and type(b) in (int,float):
        if not math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10):raise ValueError('Independent numeric calibration replay differs')
    else:same(a,b)


def expected_calibration_masks(n,seed):
    rng=random.Random(seed);used=set();result={}
    for phase,count in (('fit',64),('selection',32),('heldout',32)):
        masks=[]
        while len(masks)<count:
            mask=tuple(sorted(rng.sample(range(n),rng.choice((1,2,3,4)))))
            if mask not in used:
                used.add(mask);masks.append(list(mask))
        result[phase]=masks
    return result


def matrix_replay(basis,fit,selection,p):
    import numpy as np
    zero=np.asarray(basis[0]['mean'],dtype=float)
    h=np.asarray([v['mean'] for v in basis[1:]])-zero
    def xy(rows):
        return np.asarray([v['centered_v'] for v in rows]),np.asarray([sum((h[b] for b in v['mask']),np.zeros(8)) for v in rows])
    x,y=xy(fit);u,v=xy(selection)
    ridge=p['calibration_ridge']
    rank=int(np.linalg.matrix_rank(x));yrank=int(np.linalg.matrix_rank(y))
    gram=x.T@x+ridge*np.eye(8)
    norms=np.linalg.norm(x,axis=0)
    unregularized=float(np.linalg.cond(x/np.where(norms>0,norms,1.)))
    condition=unregularized if math.isfinite(unregularized) else None
    gram_condition=float(np.linalg.cond(gram))
    options=[('identity',np.eye(8)),('diagonal',np.diag(np.sum(x*y,axis=0)/(np.sum(x*x,axis=0)+ridge)))]
    if rank==yrank==8 and condition is not None and condition<=p['maximum_matrix_condition']:
        options.append(('full',np.linalg.solve(gram,x.T@y).T))
    candidates=[]
    for name,c in options:
        gain=float(np.max(np.sum(np.abs(c),axis=1)))
        if np.isfinite(c).all() and gain<=p['maximum_matrix_gain']:
            candidates.append(dict(name=name,matrix=c.tolist(),selection_rmse_v=float(np.sqrt(np.mean((u@c.T-v)**2))),operator_gain=gain))
    if not candidates:
        raise ValueError('No independently eligible C')
    chosen=min(candidates,key=lambda c:(round(c['selection_rmse_v'],12),c['name']!='identity',c['name']))
    return dict(**chosen,candidates=candidates,fit_rank=rank,reference_rank=yrank,
                condition=condition,condition_definition='2-norm of unregularized column-L2-normalized fit design',
                regularized_gram_condition=gram_condition,targets='measured_singleton_superposition',trained_target_weights_used=False)


def compare_calibration(local, recorded, basis, fit, portable):
    """Versioned exception ONLY for excluded rank-deficient design condition.

    All decision inputs/outputs other than this unused diagnostic retain the
    original 1e-10 comparison. No pseudoinverse or substituted calibration.
    """
    try:
        numerically_same(local, recorded)
        return dict(strict_numeric_replay='PASS', decision_equivalence='exact_within_original_tolerance')
    except ValueError:
        if not portable:
            raise
    a, b = dict(local), dict(recorded)
    ca, cb = a.pop('condition'), b.pop('condition')
    numerically_same(a, b)
    if not (local['fit_rank'] < 8 or local['reference_rank'] < 8):
        raise ValueError('Portable policy requires rank-deficient X or Y')
    if any(v['name'] == 'full' for v in local['candidates'] + recorded['candidates']):
        raise ValueError('Portable policy forbids a full-matrix candidate')
    for value in (ca, cb):
        if value is not None and not (type(value) in (int, float) and math.isfinite(value) and value > 1e6):
            raise ValueError('Portable condition discrepancy outside approved domain')
    import numpy as np
    h=np.asarray([v['mean'] for v in basis[1:]])-np.asarray(basis[0]['mean'])
    matrices={'fit':np.asarray([v['centered_v'] for v in fit]),
              'reference':np.asarray([sum((h[bl] for bl in v['mask']),np.zeros(8)) for v in fit])}
    spectra={}
    for name,x in matrices.items():
        singular=np.linalg.svd(x,compute_uv=False)
        tol=float(singular.max()*max(x.shape)*np.finfo(singular.dtype).eps)
        stable=bool(np.all((singular >= 16*tol) | (singular <= tol/16)))
        spectra[name]=dict(singular_values=singular.tolist(),numpy_rank_tolerance=tol,margin_factor=16,stable=stable)
        if not stable:
            raise ValueError('review_required: singular value inside portable rank margin band')
    return dict(strict_numeric_replay='FAIL_condition_only', decision_equivalence='PASS',
                policy='excluded-rank-condition-v1', recorded_condition=cb, local_condition=ca,
                unchanged_other_fields_tolerance=dict(absolute=1e-10,relative=1e-10),spectra=spectra)


def audit_range(records,image,weights,protocol_sha,first_stage=0,stop_stage=48,allow_prefix_eof=False,portable_condition=False):
    events=iter(records)
    start=next(events)
    same(start.get('event'),'session_begin')
    p=start['protocol'];same(digest(p),protocol_sha);same(start['protocol_sha256'],protocol_sha)
    same(start['image_sha256'],hashlib.sha256(bytes(image)).hexdigest())
    same(start.get('conv1_weights_file_sha256'),'8be312b9eafb9de23f8340b17f848e7e0e2e7884293c826a6d9f6cbb5125d440')
    same(start.get('checkpoint_sha256'),'ee4fc4f58a3d010218116e9ad4c7ba808787959f9ed7687cd58cbb32d90f62fb')
    source=start['source'];scope=start['scope']
    if source not in ('hardware','simulator') or scope not in ('full','smoke'):
        raise ValueError('Unknown source/scope')
    gain=p[source+'_gain'];stages=48 if scope=='full' else 3
    expected_positions=784 if scope=='full' else 1
    if not (type(first_stage) is int and type(stop_stage) is int and 0<=first_stage<stop_stage<=stages):
        raise ValueError('Invalid explicit stage range')
    if start.get('acquisition_kind')=='stage_segment':
        same(start.get('stage_range'),[first_stage,stop_stage])
        same(stop_stage,first_stage+1)
    elif first_stage!=0 or stop_stage!=stages or 'stage_range' in start:
        raise ValueError('Original acquisition requires its original full range')
    stage_count=first_stage
    sequence=pulses=0
    accepted=[];payloads={};stage_start_sequence=stage_start_pulses=0
    boundary=False;terminal=False
    basis_completions=heldout_completions=0;numeric_diagnostics=[]
    current_stage=-1;coefficient=None;coefficient_count=0
    last_r={};native={};vectors=[];basis=None;matrix=None
    observations={'fit':[],'selection':[],'heldout':[]}
    bracket={};position=None;position_count=0;position_vectors=[];expected=None
    size=12*64*28*28*3
    values=array('f',[0.])*size;coverage=bytearray(size)
    phase_counts={};all_coefficients=0;last_vector_end=0;decisions={};pending_pulse=None;feedback=None
    zero_shift_max=duration_max=0.;retention=[];heldout_metrics=[];last_time=-1.;programmed=False;worst_four=False
    weight_observations=[];stage_weights=[];last_feedback_task=-1;last_write_task=-1
    raw_values=array('f',[0.])*size
    for event in events:
        kind=event.get('event')
        same(event.get('source'),source)
        if kind!='stage_begin':same(event.get('stage'),current_stage)
        if kind=='request':
            if boundary or current_stage<first_stage:raise ValueError('Native operation outside active stage')
            sequence+=1;same(event['task_id'],sequence)
            result=next(events)
            while result.get('event')=='driver_output':
                same(result.get('task_id'),sequence);same(result.get('source'),source);same(result.get('stage'),current_stage)
                if result.get('stream') not in ('stdout','stderr') or not isinstance(result.get('text'),str):raise ValueError('Malformed driver output')
                result=next(events)
            same(result.get('event'),'result')
            for key,value in event.items():
                if key!='event':same(result.get(key),value)
            c=event['command_si'];mode=c['mode_flag'];phase=event['phase']
            timestamp=event.get('monotonic_s')
            if type(timestamp) not in (int,float) or not math.isfinite(timestamp) or timestamp<last_time:raise ValueError('Invalid/nonmonotonic native timestamp')
            last_time=timestamp
            if pending_pulse is not None and not (mode==7 and c['vol']):
                raise ValueError('Unresolved programming guard decision')
            phase_counts[phase]=phase_counts.get(phase,0)+1
            triple=result['native_result']
            if not isinstance(triple,list) or len(triple)!=3 or type(triple[1]) is not int or type(triple[2]) is not int:
                raise ValueError('Invalid native response')
            r,echo,adc=triple
            if mode==7:
                same(echo,sequence);same(c['id'],sequence)
                wl,bl=c['wl'],c['bl']
                if type(wl) is not int or type(bl) is not int or wl not in range(8) or bl not in range(32):
                    raise ValueError('Invalid address')
                if type(r) not in (int,float) or not math.isfinite(r) or r<=1e-8 or not 20<=adc<=16382:
                    raise ValueError('Invalid native R/ADC')
                voltage=c['vol']
                if phase=='pulse_guard':
                    if coefficient is None:raise ValueError('Pulse guard outside coefficient')
                    f,tap,physical_bl,target=coefficient
                    same(bl,physical_bl);same(event['filter'],f);same(event['tap'],tap)
                    purpose=event['intended_phase'];step=event['step']
                    if purpose not in ('program_reset','program_set'):raise ValueError('Unknown programming purpose')
                    levels=p['reset_levels_v'] if purpose=='program_reset' else p['set_levels_v']
                    key=(current_stage,f,tap,purpose,wl)
                    if type(step) is not int or step!=decisions.get(key,-1)+1 or step>=len(levels):
                        raise ValueError('Repeated or out-of-order programming step')
                    decisions[key]=step
                    pending_pulse=dict(wl=wl,bl=bl,level=levels[step],phase=purpose,step=step,filter=f,tap=tap,guard_id=sequence,r=r)
                if voltage:
                    if coefficient is None or phase not in ('program_reset','program_set'):
                        raise ValueError('Unplanned programming')
                    f,tap,physical_bl,target=coefficient
                    same(bl,physical_bl);same(event['filter'],f);same(event['tap'],tap)
                    levels=p['reset_levels_v'] if phase=='program_reset' else p['set_levels_v']
                    step=event['step']
                    if type(step) is not int or not 0<=step<len(levels):raise ValueError('Pulse outside ramp')
                    same(voltage,levels[step]);same(c['sign'],0 if phase=='program_reset' else 1)
                    if phase=='program_reset':
                        if wl not in (2*(f%4),2*(f%4)+1):raise ValueError('Wrong reset leg')
                    else:same(wl,2*(f%4)+(target<0))
                    previous=last_r.get((wl,bl))
                    if previous is None or previous['task_id']!=sequence-1 or previous['phase']!='pulse_guard':
                        raise ValueError('Missing fresh pulse resistance')
                    if voltage/previous['r']>(.04 if c['sign']==0 else .002):
                        raise ValueError('Write current guard exceeded')
                    if pending_pulse is None:raise ValueError('Missing pulse decision')
                    for key,value in dict(wl=wl,bl=bl,level=voltage,phase=phase,step=step,filter=f,tap=tap,guard_id=sequence-1).items():same(pending_pulse[key],value)
                    pending_pulse=None
                    same(c['pulse_width'],.000125 if source=='simulator' else 0.)
                    pulses+=1
                    last_write_task=sequence
                    if pulses>p['maximum_programming_pulses']:raise ValueError('Pulse budget exceeded')
                else:
                    same(c['sign'],0);same(c['pulse_width'],0.)
                last_r[wl,bl]=dict(event,r=r)
            elif mode==10:
                same(echo,0);same(c['id'],0)
                if not 0<=adc<=16382:raise ValueError('Invalid mode10 ADC')
                wl=c['wl'];mask=event['mask']
                if len(mask)>4 or len(set(mask))!=len(mask) or any(type(bl) is not int or bl not in range(32) for bl in mask):
                    raise ValueError('Invalid mask')
                same(c,dict(mode_flag=10,vol=[.05 if bl in mask else 0. for bl in range(32)],wl=wl,id=0))
                current=0.
                for bl in mask:
                    previous=last_r.get((wl,bl))
                    if previous is None or previous['phase']!='vector_guard' or previous.get('vector_phase')!=phase:
                        raise ValueError('Missing current vector guard read')
                    if previous['task_id']<=last_vector_end:raise ValueError('Stale vector guard read')
                    same(previous.get('mask'),mask)
                    for key in ('y','x','t','split','filter','tap','mask_index','repeat','basis_index'):
                        same(previous.get(key),event.get(key))
                    current+=.05/previous['r']
                if current>.005 or 330*current>=3.3:raise ValueError('MVM current guard exceeded')
                if basis is not None and not phase.startswith('program') and phase!='basis':
                    zero=basis[0]
                    upper=zero['mean_adc'][wl]+3*zero['sd_adc'][wl]
                    upper+=2*sum(max(0.,basis[b+1]['mean_adc'][wl]-zero['mean_adc'][wl]+3*math.hypot(basis[b+1]['sd_adc'][wl],zero['sd_adc'][wl])) for b in mask)
                    if upper>.8*16383:raise ValueError('Basis headroom exceeded')
                v=round(adc*5/16383,5)
                if v>4:raise ValueError('Observed ADC headroom exceeded')
                native[sequence]=dict(event,adc=adc,v=v)
            else:raise ValueError('Unexpected native mode')
            if sequence>p['maximum_impacts']:raise ValueError('Impact budget exceeded')
        elif kind=='stage_begin':
            if stage_count>=stop_stage or (current_stage>=first_stage and not boundary):raise ValueError('Overlapping/out-of-range stage')
            same(event['stage'],stage_count);current_stage=stage_count
            boundary=False;stage_start_sequence=sequence;stage_start_pulses=pulses
            basis_completions=heldout_completions=0
            basis=matrix=None;vectors=[];observations={'fit':[],'selection':[],'heldout':[]}
            coefficient_count=position_count=0;bracket={};programmed=False;worst_four=False;stage_weights=[]
            masks=expected_calibration_masks((32,32,11)[current_stage%3],p['seed']+current_stage)
        elif kind=='coefficient_begin':
            if coefficient is not None:raise ValueError('Overlapping coefficients')
            lo,hi=((0,32),(32,64),(64,75))[current_stage%3]
            pair,bl=divmod(coefficient_count,hi-lo);f=(current_stage//3)*4+pair;tap=lo+bl
            for key,value in dict(filter=f,tap=tap,bl=bl,trained_weight=float(weights[f][tap]),target_v=float(weights[f][tap])*gain).items():same(event.get(key),value)
            coefficient=(f,tap,bl,event['target_v'])
            feedback=None;last_feedback_task=-1;last_write_task=-1
        elif kind=='coefficient_complete':
            if coefficient is None:raise ValueError('Unstarted coefficient')
            if pending_pulse is not None:raise ValueError('Unresolved programming decision')
            f,tap,bl,target=coefficient
            same(event['filter'],f);same(event['tap'],tap);same(event['bl'],bl)
            if feedback is None or last_feedback_task<=last_write_task or 'program_feedback' in bracket:raise ValueError('Missing or stale actual programming feedback')
            actual=feedback[2*(f%4)]-feedback[2*(f%4)+1]
            same(event['actual_v'],actual);same(event['target_v'],target)
            tolerance=max(p['weight_absolute_tolerance_v'],abs(target)*p['weight_relative_tolerance'])
            same(event['within_tolerance'],abs(actual-target)<=tolerance)
            stage_weights.append(dict(stage=current_stage,filter=f,tap=tap,bl=bl,positive_cell=[2*(f%4),bl],negative_cell=[2*(f%4)+1,bl],
                                      trained_weight=float(weights[f][tap]),immediate_weight=actual/gain,
                                      immediate_within_tolerance=event['within_tolerance'],programming_stop=event['stop']))
            coefficient=None;coefficient_count+=1;all_coefficients+=1
        elif kind=='programming_complete':
            lo,hi=((0,32),(32,64),(64,75))[current_stage%3]
            same(coefficient_count,4*(hi-lo));same(event['coefficients'],coefficient_count)
            if programmed or coefficient is not None or pending_pulse is not None:raise ValueError('Invalid programming completion')
            programmed=True
        elif kind=='vector_complete':
            rows=event['rows']
            if len(rows)!=8:raise ValueError('Incomplete eight-row vector')
            first=rows[0]['task_id']
            same([r['task_id'] for r in rows],list(range(first,first+8)))
            mask=event['mask']
            guards=sorted(last_r[wl,bl]['task_id'] for wl in range(8) for bl in mask)
            same(guards,list(range(first-8*len(mask),first)))
            for wl,row in enumerate(rows):
                raw=native.pop(row['task_id'])
                same(raw['command_si']['wl'],wl);same(raw['mask'],event['mask']);same(raw['phase'],event['phase'])
                same(row,dict(adc=raw['adc'],v=raw['v'],task_id=raw['task_id'],monotonic_s=raw['monotonic_s']))
            if native:raise ValueError('Unconsumed MVM results')
            last_vector_end=sequence
            if event['phase'] in ('basis','final_basis'):vectors.append(event)
            phase=event['phase']
            if phase in ('program_feedback','program_feedback_zero_before','program_feedback_zero_after'):
                if coefficient is None or matrix is not None:raise ValueError('Feedback outside programming')
                f,tap,bl,target=coefficient
                same(event.get('filter'),f);same(event.get('tap'),tap)
                same(event['mask'],[bl] if phase=='program_feedback' else [])
            if not phase.startswith('program') and not programmed:raise ValueError('Calibration/inference before programming completion')
            for name in ('fit','selection','heldout'):
                if phase in (name,name+'_zero_before',name+'_zero_after'):
                    count=len(observations[name]);index,repeat=divmod(count,3)
                    if index>=len(masks[name]):raise ValueError('Extra calibration record')
                    same(event.get('mask_index'),index);same(event.get('repeat'),repeat)
                    same(event['mask'],masks[name][index] if phase==name else [])
                    if name=='heldout' and matrix is None:raise ValueError('Heldout before freeze')
                    if name!='heldout' and matrix is not None:raise ValueError('Refit after freeze')
            if phase.startswith('inference'):
                if position is None:raise ValueError('Inference outside position')
                wanted=next(expected)
                same((phase,event.get('t'),event.get('split'),event['mask']),wanted)
                same((event['y'],event['x']),position)
                position_vectors.append(event)
            for name in ('fit','selection','heldout','program_feedback'):
                if phase==name+'_zero_before':bracket[name]=[event]
                elif phase==name:
                    if len(bracket.get(name,[]))!=1:raise ValueError('Missing zero before')
                    bracket[name].append(event)
                elif phase==name+'_zero_after':
                    if len(bracket.get(name,[]))!=2:raise ValueError('Missing fit signal')
                    before,signal=bracket.pop(name)
                    centered=[s['v']-(a['v']+b['v'])/2 for a,s,b in zip(before['rows'],signal['rows'],event['rows'])]
                    if name=='program_feedback':feedback=centered;last_feedback_task=sequence
                    else:observations[name].append(dict(mask=signal['mask'],centered_v=centered))
        elif kind=='basis_complete':
            if basis_completions or matrix is not None:raise ValueError('Repeated or late basis completion')
            basis_completions+=1
            n=((32,32,11)[current_stage%3]);computed=[]
            for i in range(n+1):
                selected=[v for v in vectors if v['phase']=='basis' and v['basis_index']==i]
                if len(selected)!=3:raise ValueError('Incomplete basis repeats')
                same([v['repeat'] for v in selected],[0,1,2]);same([v['mask'] for v in selected],[[] if i==0 else [i-1]]*3)
                computed.append(dict(mean=[statistics.mean(v['rows'][wl]['v'] for v in selected) for wl in range(8)],
                                     mean_adc=[statistics.mean(v['rows'][wl]['adc'] for v in selected) for wl in range(8)],
                                     sd_adc=[statistics.stdev(v['rows'][wl]['adc'] for v in selected) for wl in range(8)]))
            same(computed,event['basis']);basis=computed
        elif kind=='worst_four_guard_passed':
            if basis is None:raise ValueError('Worst-four guard before basis')
            same(event['basis_sha256'],digest(basis))
            for wl in range(8):
                zero=basis[0]
                terms=[max(0.,b['mean_adc'][wl]-zero['mean_adc'][wl]+3*math.hypot(b['sd_adc'][wl],zero['sd_adc'][wl])) for b in basis[1:]]
                upper=zero['mean_adc'][wl]+3*zero['sd_adc'][wl]+2*sum(sorted(terms,reverse=True)[:4])
                current=.05*sum(sorted((1/last_r[wl,b]['r'] for b in range(len(basis)-1)),reverse=True)[:4])
                if upper>.8*16383 or current>.005 or 330*current>=3.3:raise ValueError('Worst-four guard fails')
            worst_four=True
        elif kind=='calibration_frozen':
            if matrix is not None or basis is None or not worst_four:raise ValueError('Invalid calibration freeze order')
            if len(observations['fit'])!=192 or len(observations['selection'])!=96:raise ValueError('Incomplete independent calibration')
            matrix=matrix_replay(basis,observations['fit'],observations['selection'],p)
            diagnostic=compare_calibration(matrix,event['matrix'],basis,observations['fit'],portable_condition)
            numeric_diagnostics.append(dict(stage=current_stage,**diagnostic));matrix=event['matrix']
            same(digest(matrix),event['calibration_sha256']);same(digest(basis),event['basis_sha256'])
            same(digest({k:observations[k] for k in ('fit','selection')}),event['fit_sha256']);same(gain,event['gain_v_per_weight'])
        elif kind=='heldout_complete':
            if heldout_completions:raise ValueError('Repeated heldout completion')
            heldout_completions+=1
            if matrix is None or len(observations['heldout'])!=96:raise ValueError('Invalid heldout order/coverage')
            same(event['observations'],observations['heldout']);same(event['refit_performed'],False)
            raw_error=[];corrected_error=[]
            for row in observations['heldout']:
                reference=[sum(basis[b+1]['mean'][wl]-basis[0]['mean'][wl] for b in row['mask']) for wl in range(8)]
                corrected=[sum(a*b for a,b in zip(cr,row['centered_v'])) for cr in matrix['matrix']]
                raw_error.extend((a-b)**2 for a,b in zip(row['centered_v'],reference))
                corrected_error.extend((a-b)**2 for a,b in zip(corrected,reference))
            heldout_metrics.append(dict(stage=current_stage,raw_rmse_v=math.sqrt(statistics.mean(raw_error)),calibrated_rmse_v=math.sqrt(statistics.mean(corrected_error))))
        elif kind=='position_begin':
            if heldout_completions!=1:raise ValueError('Position before completed heldout verification')
            if matrix is None or len(observations['heldout'])!=96 or position is not None:raise ValueError('Uncalibrated or overlapping position')
            position=(event['y'],event['x']);same(position,divmod(position_count,28) if scope=='full' else (0,0))
            same(event['calibration_sha256'],digest(matrix));position_vectors=[]
            schedule=[('inference_zero_before',None,None,[])]
            schedule.extend(('inference_signal',t,split,mask) for t,split,mask in expected_signals(image,current_stage%3,*position))
            schedule.append(('inference_zero_after',None,None,[]));expected=iter(schedule)
        elif kind=='position_complete':
            if position is None or next(expected,None) is not None:raise ValueError('Incomplete physical position')
            before,after=position_vectors[0]['rows'],position_vectors[-1]['rows']
            zero_shift_max=max(zero_shift_max,max(abs(a['v']-b['v']) for a,b in zip(before,after)))
            duration_max=max(duration_max,after[-1]['monotonic_s']-before[0]['monotonic_s'])
            same(event['zero_before_task_ids'],[v['task_id'] for v in before]);same(event['zero_after_task_ids'],[v['task_id'] for v in after])
            output=[[0.]*4 for _ in range(12)]
            raw_output=[[0.]*4 for _ in range(12)]
            for v in position_vectors[1:-1]:
                centered=[s['v']-(a['v']+b['v'])/2 for a,s,b in zip(before,v['rows'],after)]
                changed=[sum(a*b for a,b in zip(row,centered)) for row in matrix['matrix']]
                for pair in range(4):
                    output[v['t']][pair]+=(changed[2*pair]-changed[2*pair+1])/gain
                    raw_output[v['t']][pair]+=(centered[2*pair]-centered[2*pair+1])/gain
            y,x=position
            for t in range(12):
                for pair in range(4):
                    f=(current_stage//3)*4+pair;index=(((t*64+f)*28+y)*28+x)*3+current_stage%3
                    if coverage[index]:raise ValueError('Duplicate tensor element')
                    values[index]=output[t][pair];raw_values[index]=raw_output[t][pair];coverage[index]=1
            position=None;position_count+=1
        elif kind=='stage_complete':
            if boundary or matrix is None:raise ValueError('Repeated or uncalibrated stage completion')
            if basis_completions!=1 or heldout_completions!=1:raise ValueError('Missing unique basis/heldout completion')
            same(position_count,expected_positions)
            n=(32,32,11)[current_stage%3]
            if not programmed or coefficient is not None or pending_pulse is not None or position is not None or coefficient_count!=4*n:
                raise ValueError('Incomplete stage programming/collection')
            final=[v for v in vectors if v['phase']=='final_basis']
            same(len(final),(n+1)*3)
            changes=[];final_means=[]
            for index in range(n+1):
                rows=final[index*3:(index+1)*3]
                same([v['basis_index'] for v in rows],[index]*3);same([v['repeat'] for v in rows],[0,1,2])
                same([v['mask'] for v in rows],[[] if index==0 else [index-1]]*3)
                changes.extend(statistics.mean(v['rows'][wl]['v'] for v in rows)-basis[index]['mean'][wl] for wl in range(8))
                final_means.append([statistics.mean(v['rows'][wl]['v'] for v in rows) for wl in range(8)])
            retention.append(dict(stage=current_stage,final_minus_initial_basis_rmse_v=math.sqrt(statistics.mean(v*v for v in changes)),max_abs_v=max(map(abs,changes))))
            for observation in stage_weights:
                pair=observation['filter']%4;bl=observation['bl']
                post=[basis[bl+1]['mean'][wl]-basis[0]['mean'][wl] for wl in range(8)]
                last=[final_means[bl+1][wl]-final_means[0][wl] for wl in range(8)]
                post_c=[sum(c*v for c,v in zip(row,post)) for row in matrix['matrix']]
                last_c=[sum(c*v for c,v in zip(row,last)) for row in matrix['matrix']]
                observation.update(post_all_raw_weight=(post[2*pair]-post[2*pair+1])/gain,
                                   post_all_calibrated_weight=(post_c[2*pair]-post_c[2*pair+1])/gain,
                                   final_raw_weight=(last[2*pair]-last[2*pair+1])/gain,
                                   final_calibrated_weight=(last_c[2*pair]-last_c[2*pair+1])/gain,
                                   per_coefficient_fitting_used=False)
                weight_observations.append(observation)
            stage_payload={}
            spatial=28 if scope=='full' else 1
            for label,data in (('raw',raw_values),('calibrated',values)):
                out=array('f')
                for t in range(12):
                    for pair in range(4):
                        f=(current_stage//3)*4+pair
                        for y in range(spatial):
                            for x in range(spatial):
                                index=(((t*64+f)*28+y)*28+x)*3+current_stage%3
                                if coverage[index]!=1 or not math.isfinite(data[index]):raise ValueError('Incomplete/nonfinite stage tensor')
                                out.append(data[index])
                if sys.byteorder!='little':out.byteswap()
                stage_payload[label]=out.tobytes()
            payloads[current_stage]=stage_payload
            accepted.append(dict(stage=current_stage,native_impacts=sequence-stage_start_sequence,
                programming_pulses=pulses-stage_start_pulses,task_first=stage_start_sequence+1,task_last=sequence,
                calibration_sha256=digest(matrix),shape=[12,4,spatial,spatial],dtype='<f4',
                payload_sha256={k:hashlib.sha256(v).hexdigest() for k,v in stage_payload.items()}))
            stage_count+=1;boundary=True
        elif kind=='session_complete':
            same(event['status'],'completed');same(stage_count,stop_stage)
            if not boundary:raise ValueError('Terminal event outside completed stage')
            terminal=True;same(event['impact_attempts'],sequence);same(event['programming_pulses'],pulses)
            if next(events,None) is not None:raise ValueError('Trailing data')
            break
        elif kind=='write_skipped':
            if coefficient is None:raise ValueError('Unplanned skipped write')
            if pending_pulse is None:raise ValueError('Missing skipped-write guard')
            for key in ('wl','bl','phase','step','filter','tap'):same(event[key],pending_pulse[key])
            same(event['voltage_v'],pending_pulse['level']);same(event['native_r'],pending_pulse['r'])
            sign=0 if event['phase']=='program_reset' else 1;same(event['sign'],sign)
            if event['voltage_v']/event['native_r']<=(.04 if sign==0 else .002):raise ValueError('Unjustified current skip')
            pending_pulse=None
        else:raise ValueError('Unexpected or aborted raw event: '+str(kind))
    else:
        if not (allow_prefix_eof and boundary and accepted):
            raise ValueError('EOF is not an accepted complete-stage boundary')
    if native or bracket or coefficient is not None or pending_pulse is not None or position is not None:
        raise ValueError('Unresolved native evidence at range end')
    return payloads,dict(schema='cifar-stage-prefix-v1',source=source,scope=scope,
        status='completed_range' if terminal else 'verified_complete_stage_prefix',
        protocol_sha256=protocol_sha,first_stage=first_stage,stop_stage_exclusive=stage_count,
        requested_stop_stage_exclusive=stop_stage,accepted_stages=accepted,
        complete_full_tensor=False,hardware_import_enabled=False,
        frozen_full_auditor_pass_claimed=False,terminal_session_event_verified=terminal,
        native_impacts=sequence,programming_pulses=pulses,
        observed_coefficients=all_coefficients,covered_tensor_elements=sum(coverage),
        maximum_shared_zero_shift_v=zero_shift_max,maximum_position_bracket_duration_s=duration_max,
        heldout=heldout_metrics,retention=retention,weight_observations=weight_observations,
        independent_fitting_float_tolerance=dict(absolute=1e-10,relative=1e-10),
        strict_numeric_replay='PASS' if all(v['strict_numeric_replay']=='PASS' for v in numeric_diagnostics) else 'FAIL_condition_only',
        portable_condition_policy='excluded-rank-condition-v1' if portable_condition else None,
        numeric_diagnostics=numeric_diagnostics,phase_counts=phase_counts)
