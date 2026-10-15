"""Independent offline audit of mapped sequential raw commands and arithmetic.

No MemriBoard, collector or primary analyzer imports. Only stdlib and the prior
independent auditor's generic file/numeric helpers are used. No hardware I/O.
"""
import argparse
from collections import Counter, defaultdict
from configparser import ConfigParser
import json
import math
from pathlib import Path
from statistics import mean, stdev
import uuid
from audit_preparation import ROOT, local, sha, load, canonical, require, close, convert_read

# Filled from the reviewed release, never learned from the session being audited.
PINS = {
    'mapped_sequential_mvm.py': 'bd7aa0900bed825de19ccd8019a3ef9010cc1b1cffb2c2874348cd3e736f28a9',
    'mapped_sequential_analysis.py': 'db451c5eb9ca01c7c4062a39d689c4634199fd95c9f84cb2476de0acc691f985',
    'configs/mapped_sequential_mvm.json': '80d1195f8288002e4013686468ce8ff04ac991c75b3210ccba5f78bc46a5b440',
}
PATTERNS = {'dispersed': 8, 'grouped': 4, 'dense_all': 1}


def moments(values):
    return dict(n=len(values), mean=mean(values), sd=stdev(values) if len(values)>1 else 0.) if values else None


def masks(name, n):
    return [[j for j in range(n) if j % PATTERNS[name] == t] for t in range(12)]


def expected_schedule(plan):
    out=[]; block=0
    def add(**values): out.append(dict(schedule_id=len(out)+1, **values))
    for phase in ('pre','basis','selection','evaluation','final','post'):
        if phase in ('pre','post'):
            for wl in range(8):
                for bl in range(32):
                    for repeat in range(3): add(mode=9,phase=phase,wl=wl,bl=bl,repeat=repeat)
        elif phase != 'evaluation':
            for repeat in range(5):
                for bl in range(-1,32):
                    for wl in range(8): add(mode=10,phase=phase,wl=wl,mask=[] if bl<0 else [bl],repeat=repeat,role='anchor')
        else:
            for pair_id,pair in enumerate(plan['pairs']):
                for name in PATTERNS:
                    for repeat in range(5):
                        for t,active in enumerate(masks(name,len(pair['positive_bl']))):
                            for wl,columns in zip(pair['wl'],(pair['positive_bl'],pair['negative_bl'])):
                                block+=1
                                meta=dict(mode=10,phase=phase,pair_id=pair_id,pattern=name,repeat=repeat,t=t,
                                          wl=wl,block_id=block,active_logical_indices=active)
                                add(**meta,mask=[],role='zero_before')
                                for j in active or [None]:
                                    add(**meta,mask=[] if j is None else [columns[j]],logical_index=j,
                                        role='zero_signal' if j is None else 'partial')
                                add(**meta,mask=[],role='zero_after')
    require(len(out)==10706,'Independent schedule budget')
    return out


def guard(meta,g,basis):
    wl=meta['wl'];mask=meta['mask'];reasons=[]
    current=.1*sum(g[wl,b] for b in mask); model=330*current
    gui=sum(.1*round(330*g[wl,b],4) for b in mask)
    if not all(math.isfinite(v) for v in (current,model,gui)) or current>.005 or model>4 or gui>=3.3:
        reasons.append('mode9_model_guard')
    upper=None
    if meta['phase']!='basis':
        zero=basis.get((wl,()));single=basis.get((wl,tuple(mask)))
        if not zero or not single: reasons.append('basis_missing')
        else:
            upper=zero['mean']+3*zero['sd']
            if mask: upper+=2*max(0.,single['mean']-zero['mean']+3*math.hypot(single['sd'],zero['sd']))
            if upper>.8*16383: reasons.append('fresh_basis_headroom')
    return dict(allowed=not reasons,reasons=reasons,mode9_model_current_a=current,mode9_model_output_v=model,
                gui_model_output_v=gui,basis_heuristic_upper_adc=upper,upper_bound_certified=False,physical_current_compliance=False)


def audit_raw(session):
    require(len(PINS)==3,'Audit release pins have not been frozen')
    manifest=load(session/'manifest.json');plan=load(session/'protocol.json');schedule=load(session/'schedule.json')
    require(manifest['status']=='completed','Only complete sessions qualify for this audit')
    require(manifest['source'] in ('hardware','simulator'),'Source')
    for name,pin in PINS.items(): require(manifest['source_sha256'][name]==pin,'Source pin: '+name)
    require(sha(session/'protocol.json')==PINS['configs/mapped_sequential_mvm.json'],'Protocol pin')
    require(schedule==expected_schedule(plan),'Independent schedule mismatch')
    require(canonical(schedule)==manifest['schedule_sha256'],'Schedule hash')
    require(plan['maximum_programming_pulses']==0 and not plan['cifar_replay'],'Forbidden operation')
    for name in ('raw.jsonl','observations.json','analysis.json','frozen_reference.json','protocol.json','schedule.json','board.ini'):
        require(sha(session/name)==manifest['result_sha256'][name],'File hash: '+name)
    cfg=ConfigParser();cfg.read_string((session/'board.ini').read_text(encoding='utf-8-sig'))
    for key,value in dict(dac_bit=12,vol_ref_dac=5,adc_bit=14,vol_ref_adc=5,gain=11,res_load=3000,res_switches=10,sum_gain=330).items():
        require(cfg.getfloat('board',key)==value,'Board parameter: '+key)
    cursor=0;sequence=0;pending=None;allowed=[];reads=[];records=[];skips=[];byblock=defaultdict(list)
    basis_values=defaultdict(list);basis_cache=None;g={};freeze_count=0;reference=load(session/'frozen_reference.json')
    for line in (session/'raw.jsonl').read_text(encoding='utf-8').splitlines():
        r=json.loads(line);event=r['event']
        require(r['session_id']==manifest['session_id'] and r['source']==manifest['source'],'Journal identity')
        if event=='reference_frozen':
            require(pending is None and not allowed and cursor==3408 and freeze_count==0,'Reference timing')
            require(r['before_task_id']==sequence+1 and r['sha256']==canonical(reference),'Reference commitment')
            freeze_count+=1
        elif event=='acquisition_guard':
            require(pending is None and not allowed and cursor<len(schedule),'Guard ordering')
            meta=schedule[cursor];require(meta['mode']==10,'Guard on read')
            group=[meta]
            if meta['phase']=='evaluation':
                require(freeze_count==1,'Evaluation before frozen reference')
                while cursor+len(group)<len(schedule) and schedule[cursor+len(group)].get('block_id')==meta['block_id']:
                    group.append(schedule[cursor+len(group)])
            if not g:
                require(len(reads)==768,'Pre coverage')
                g={(w,b):mean(x['conductance_siemens'] for x in reads if x['wl']==w and x['bl']==b) for w in range(8) for b in range(32)}
            if meta['phase']=='basis': basis={}
            else:
                if basis_cache is None: basis_cache={k:moments(v) for k,v in basis_values.items() if len(v)==5}
                basis=basis_cache
            decisions=[guard(m,g,basis) for m in group];permitted=all(d['allowed'] for d in decisions)
            require(close(r['decisions'],decisions) and r['allowed']==permitted,'Independent guard mismatch')
            require(r['schedule_ids']==[m['schedule_id'] for m in group],'Guard schedule IDs')
            if permitted: allowed=list(group)
            else: skips.append(dict(schedule_ids=r['schedule_ids'],phase=meta['phase'],reasons=sorted({x for d in decisions for x in d['reasons']})))
            cursor+=len(group)
        elif event=='request':
            require(pending is None,'Nested request')
            if allowed: meta=allowed.pop(0)
            else:
                require(cursor<len(schedule) and schedule[cursor]['mode']==9,'Unguarded request')
                meta=schedule[cursor];cursor+=1
            sequence+=1
            require(r['task_id']==sequence and all(r[k]==v for k,v in meta.items()),'Request schedule metadata')
            wl=meta['wl']
            if meta['mode']==9:
                command=dict(mode_flag=9,vol=.1,pulse_width=0.,sign=0,wl=wl,bl=meta['bl'],id=sequence)
                conversion=dict(requested_v=.1,dac_code=82,native_decoded_v=.1)
            else:
                require(len(meta['mask'])<=1,'Non-single physical command')
                command=dict(mode_flag=10,vol=[.1 if b in meta['mask'] else 0. for b in range(32)],wl=wl,id=0)
                conversion=[dict(requested_v=v,dac_code=82 if v else 0,native_decoded_v=v) for v in command['vol']]
            require(r['command_si']==command and close(r['native_voltage_conversion'],conversion),'Command/voltage payload')
            pending=r
        elif event=='result':
            require(pending is not None and all(r[k]==v for k,v in pending.items() if k!='event'),'Request/result mismatch')
            require(r['status']=='ok','Failed result in complete session')
            adc=r['raw_adc'];require(type(adc) is int and 0<=adc<=16382,'ADC')
            require(type(r['returned_id']) is int and r['returned_id']==0 and r['native_result'][1:]==[0,adc],'Native tuple')
            meta=schedule[r['schedule_id']-1]
            if r['mode']==9:
                resistance,conductance=convert_read(adc)
                require(close([r['resistance_ohm'],r['conductance_siemens'],r['q'],r['read_voltage_parameter_v']],
                              [resistance,conductance,330/resistance,.1]),'Native read conversion')
                reads.append({k:v for k,v in r.items() if k not in ('event','latency_s','command_si','native_voltage_conversion','native_result')})
            else:
                require(close(r['output_voltage_v'],round(adc*5/16383,5)) and r['output_voltage_v']<=4.,'Output voltage')
                records.append(dict(**meta,task_id=r['task_id'],raw_adc=adc))
                if r['phase']=='basis':basis_values[r['wl'],tuple(r['mask'])].append(adc)
                if r['phase']=='evaluation':byblock[r['block_id']].append(records[-1])
            pending=None
        else: raise ValueError('Unknown journal event: '+event)
    require(pending is None and not allowed and cursor==10706 and freeze_count==1,'Incomplete journal')
    require(sequence==manifest['impact_attempts'] and sequence+sum(len(s['schedule_ids']) for s in skips)==10706,'Impact ledger')
    require(len(reads)==1536,'Read coverage')
    obs=load(session/'observations.json')
    require(obs['records']==records and obs['readbacks']==reads and obs['skips']==skips,'Raw compact records')
    reconstructed=[]
    for rows in byblock.values():
        first,last=rows[0],rows[-1]
        require(first['role']=='zero_before' and last['role']=='zero_after','Incomplete brackets')
        signals=[r['raw_adc'] for r in rows[1:-1]]
        centered=sum(signals)-len(signals)*(first['raw_adc']+last['raw_adc'])/2
        reconstructed.append(dict(pair_id=first['pair_id'],pattern=first['pattern'],repeat=first['repeat'],t=first['t'],
            wl=first['wl'],block_id=first['block_id'],active_logical_indices=first['active_logical_indices'],
            task_ids=[r['task_id'] for r in rows],schedule_ids=[r['schedule_id'] for r in rows],zero_before_adc=first['raw_adc'],
            signal_adc=signals,zero_after_adc=last['raw_adc'],centered_adc=centered))
    require(close(reconstructed,[{k:v for k,v in b.items() if k!='elapsed_s'} for b in obs['blocks']]),'Raw bracket reconstruction')
    return manifest,plan,records,reads,reconstructed,reference


def anchors(records,phase):
    values=defaultdict(list)
    for r in records:
        if r['phase']==phase: values[r['wl'],tuple(r['mask'])].append(r['raw_adc'])
    result={}
    for wl in range(8):
        z=moments(values[wl,()])
        for bl in range(32):
            s=moments(values[wl,(bl,)])
            valid=bool(z and s and z['n']==s['n']==5)
            d=s['mean']-z['mean'] if valid else None
            noise=math.hypot(s['sd'],z['sd']) if valid else None
            result[wl,bl]=dict(zero=z,single=s,signal=d,quality=bool(valid and d>=5*max(1.,noise) and noise<=.05*d))
    return result


def integrate(values):
    out=[];v=0.
    for current in values:
        pre=.5*v+current;spike=int(pre>=1.);v=0. if spike else pre
        out.append(dict(current=current,before=pre,after=v,spike=spike,margin=pre-1.))
    return out


def check_metric(actual,values,target):
    s=moments(values);limit=max(2.,.05*abs(target))
    mean_pass=bool(s and s['n']==5 and abs(s['mean']-target)<=limit)
    sd_pass=bool(s and s['n']==5 and s['sd']<=limit)
    expected=dict(expected_adc=target,samples_adc=values,statistics=s,tolerance_adc=limit,
                  error_adc=s['mean']-target if s else None,mean_pass=mean_pass,repeatability_pass=sd_pass,passes=mean_pass and sd_pass)
    require(close(actual,expected),'Independent arithmetic metric')
    return mean_pass and sd_pass


def review_science(records,reads,blocks,reference,plan,analysis):
    tables={phase:anchors(records,phase) for phase in ('basis','selection','final')}
    basis=tables['basis'];groups=defaultdict(list)
    for r in reads: groups[r['phase'],r['wl'],r['bl']].append(r['conductance_siemens'])
    cells={};selection={}
    for address,b in basis.items():
        def agree(other):
            return bool(b['quality'] and other['quality'] and abs(other['single']['mean']-b['single']['mean'])<=max(2.,.05*abs(b['signal']))
                        and abs(other['zero']['mean']-b['zero']['mean'])<=2 and other['zero']['sd']<=2)
        selection[address]=agree(tables['selection'][address])
        pre=moments(groups['pre',*address]);post=moments(groups['post',*address])
        drift=bool(pre and post and pre['n']==post['n']==3 and abs(post['mean']-pre['mean'])/pre['mean']<=
                   max(.05,3*math.sqrt(pre['sd']**2/3+post['sd']**2/3)/pre['mean']))
        cells[address]=selection[address] and agree(tables['final'][address]) and drift
    require(len(analysis['cells'])==256 and all(c['singleton_supported']==cells[c['wl'],c['bl']] for c in analysis['cells']),'Singleton gates')
    lookup={(b['pair_id'],b['pattern'],b['repeat'],b['t'],b['wl']):b for b in blocks}
    reported_steps={(s['pair_id'],s['pattern'],s['t']):s for s in analysis['steps']}
    reported_lif={(s['pair_id'],s['pattern'],s['polarity']):s for s in analysis['lif']}
    reported_coeff={(c['pair_id'],c['logical_index']):c for c in analysis['coefficients']}
    require(len(reported_steps)==144 and len(reported_lif)==24 and len(reported_coeff)==47,'Analysis coverage')
    step_statuses=Counter();lif_statuses=Counter();supported=0
    for pair_id,pair in enumerate(plan['pairs']):
        a,b=pair['wl'];weights=[];eligible=[];fresh_selection=[]
        for j,(left,right) in enumerate(zip(pair['positive_bl'],pair['negative_bl'])):
            addresses=((a,left),(b,right));phase_d={};phase_quality={}
            for phase,table in tables.items():
                x,y=(table[k] for k in addresses);valid=x['signal'] is not None and y['signal'] is not None
                d=x['signal']-y['signal'] if valid else None
                bound=sum(v['sd'] for cell in (x,y) for v in (cell['single'],cell['zero'])) if valid else None
                phase_d[phase]=d;phase_quality[phase]=bool(valid and abs(d)>=5*max(1.,bound) and bound<=.05*abs(d))
            d=phase_d['basis'];weights.append(d)
            selected=all(selection[k] for k in addresses) and phase_quality['basis'] and phase_quality['selection']
            selected=bool(selected and abs(phase_d['selection']-d)<=max(2.,.05*abs(d)))
            passed=bool(selected and all(cells[k] for k in addresses) and phase_quality['final'] and abs(phase_d['final']-d)<=max(2.,.05*abs(d)))
            fresh_selection.append(selected);eligible.append(passed);supported+=passed
            require(reported_coeff[pair_id,j]['selection_supported']==selected and reported_coeff[pair_id,j]['final_supported']==passed,'Contrast gates')
        expected={};peak=0.
        if all(w is not None for w in weights):
            for name in PATTERNS:
                expected[name]=[sum(weights[j] for j in active) for active in masks(name,len(weights))]
                v=0.
                for current in expected[name]:v=.5*v+current;peak=max(peak,abs(v))
        scale=1.25/peak if peak else None
        frozen=reference['pairs'][pair_id]
        require(close(frozen['scale'],scale) and close(frozen['expected_currents_adc'],expected),'Basis-only scale/reference')
        require(frozen['selection_supported']==all(fresh_selection),'Frozen selection gate')
        for name in PATTERNS:
            currents=[];complete=True;statuses=[]
            for repeat in range(5):
                trajectory=[]
                for t in range(12):
                    rows=[lookup.get((pair_id,name,repeat,t,wl)) for wl in (a,b)]
                    if any(row is None for row in rows):trajectory.append(None);complete=False
                    else:trajectory.append(rows[0]['centered_adc']-rows[1]['centered_adc'])
                currents.append(trajectory)
            for t,active in enumerate(masks(name,len(weights))):
                report=reported_steps[pair_id,name,t];row_pass=[];row_complete=[];bounds=[]
                for wl,columns in zip((a,b),(pair['positive_bl'],pair['negative_bl'])):
                    observed=[lookup[pair_id,name,r,t,wl] for r in range(5) if (pair_id,name,r,t,wl) in lookup]
                    samples=[v['centered_adc'] for v in observed];row_complete.append(len(samples)==5)
                    if expected:
                        target=sum(basis[wl,columns[j]]['signal'] for j in active)
                        row_pass.append(check_metric(report['rows'][str(wl)],samples,target))
                    else: require(report['rows'][str(wl)] is None,'Unexpected row reference')
                    bound=None
                    if len(observed)==5:
                        k=len(observed[0]['signal_adc']);signals=[stdev(v['signal_adc'][j] for v in observed) for j in range(k)]
                        z0=stdev(v['zero_before_adc'] for v in observed);z1=stdev(v['zero_after_adc'] for v in observed)
                        bound=sum(signals)+k*(z0+z1)/2
                        require(close(report['row_noise_bounds'][str(wl)]['conservative_sd_bound_adc'],bound),'Baseline covariance bound')
                    else: require(report['row_noise_bounds'][str(wl)] is None,'Incomplete noise bound')
                    bounds.append(bound)
                require(close(report['signed_sd_bound_adc'],sum(bounds) if all(v is not None for v in bounds) else None),'Signed noise bound')
                samples=[values[t] for values in currents if values[t] is not None]
                signed=check_metric(report['signed'],samples,expected[name][t]) if expected else False
                support=all(eligible[j] for j in active) if active else all(eligible)
                rs='inconclusive' if not support or not expected or not all(row_complete) else 'pass' if all(row_pass) else 'fail'
                ss='inconclusive' if not support or not expected or len(samples)!=5 else 'pass' if signed else 'fail'
                status='inconclusive' if 'inconclusive' in (rs,ss) else 'pass' if rs==ss=='pass' else 'fail'
                require((report['row_status'],report['signed_status'],report['status'])==(rs,ss,status),'Step gates')
                step_statuses[status]+=1;statuses.append(status)
            for polarity in (1,-1):
                report=reported_lif[pair_id,name,polarity]
                ref=integrate([polarity*scale*x for x in expected[name]]) if scale is not None and expected else None
                actual=[integrate([polarity*scale*x for x in v]) for v in currents] if complete and scale is not None else []
                require(close(report['expected'],ref) and close(report['actual'],actual),'Independent LIF trajectories')
                useful=bool(ref and any(v['spike'] for v in ref) and any(not v['spike'] for v in ref))
                matching=[all(x['spike']==y['spike'] for x,y in zip(v,ref)) for v in actual] if ref else []
                errors=[max(abs(x[k]-y[k]) for x,y in zip(v,ref) for k in ('before','after')) for v in actual] if ref else []
                numerical=bool(complete and useful and matching and all(matching) and max(errors)<=.05)
                ready=complete and useful and all(eligible);mvm=all(s=='pass' for s in statuses)
                status='inconclusive' if not ready else 'pass' if numerical and mvm else 'fail'
                require(close(report['max_membrane_errors'],errors) and report['exact_spikes_by_repeat']==matching,'LIF errors')
                require(report['status']==status and report['operation_qualified']==bool(ready and numerical and mvm),'LIF gates')
                lif_statuses[status]+=1
    summary=dict(singleton_supported=sum(cells.values()),mapped_contrasts_supported=supported,
                 step_statuses=dict(step_statuses),lif_statuses=dict(lif_statuses))
    require(all(analysis[k]==v for k,v in summary.items()),'Scientific summary')
    return summary


def inspect(session):
    manifest,plan,records,reads,blocks,reference=audit_raw(session)
    science=review_science(records,reads,blocks,reference,plan,load(session/'analysis.json'))
    return dict(session_id=manifest['session_id'],source=manifest['source'],passed=True,impacts=manifest['impact_attempts'],
                complete_blocks=len(blocks),**science,hardware_io=False,cifar_qualified=False,trained_weights_programmed=False,
                limitation='Fixed routing and three synthetic patterns only; no arbitrary weights or CIFAR accuracy claim.')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--session',action='append',required=True)
    parser.add_argument('--output');args=parser.parse_args(argv)
    reports=[inspect(local(value)) for value in args.session]
    target=local(args.output or ('results/mapped_sequential_reviews/'+str(uuid.uuid4())))
    target.mkdir(parents=True,exist_ok=False)
    report=dict(auditor_sha256=sha(Path(__file__)),sessions=reports,independent_math=True)
    (target/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(output=target.relative_to(ROOT).as_posix(),sessions=reports)));return 0


if __name__=='__main__':raise SystemExit(main())
