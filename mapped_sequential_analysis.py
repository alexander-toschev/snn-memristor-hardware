"""Pure offline arithmetic for newly measured, sequential one-hot contributions."""
from collections import Counter, defaultdict
import math
from statistics import mean, stdev


def stats(values):
    return dict(n=len(values),mean=mean(values),sd=stdev(values) if len(values)>1 else 0.) if values else None


def pattern(name,n):
    modulus={'dispersed':8,'grouped':4,'dense_all':1}[name]
    return [[j for j in range(n) if j%modulus==t] for t in range(12)]


def lif(currents):
    v=0.;out=[]
    for current in currents:
        before=.5*v+current;spike=int(before>=1.);v=0. if spike else before
        out.append(dict(current=current,before=before,spike=spike,after=v,margin=before-1.))
    return out


def assess(samples,expected):
    s=stats(samples);limit=max(2.,.05*abs(expected))
    return dict(expected_adc=expected,samples_adc=samples,statistics=s,tolerance_adc=limit,
        error_adc=s['mean']-expected if s else None,
        mean_pass=bool(s and len(samples)==5 and abs(s['mean']-expected)<=limit),
        repeatability_pass=bool(s and len(samples)==5 and s['sd']<=limit),
        passes=bool(s and len(samples)==5 and abs(s['mean']-expected)<=limit and s['sd']<=limit))


def anchor_table(records,phase):
    groups=defaultdict(list)
    for r in records:
        if r['phase']==phase:groups[r['wl'],tuple(r['mask'])].append(r['raw_adc'])
    out={}
    for wl in range(8):
        z=stats(groups[wl,()])
        for bl in range(32):
            s=stats(groups[wl,(bl,)])
            valid=bool(z and s and z['n']==s['n']==5)
            signal=s['mean']-z['mean'] if valid else None
            noise=math.hypot(s['sd'],z['sd']) if valid else None
            out[wl,bl]=dict(zero=z,single=s,signal_adc=signal,noise_adc=noise,
                quality=bool(valid and signal>=5*max(1.,noise) and noise<=.05*signal))
    return out


def freeze_reference(records,plan):
    """Only basis and independent selection enter this object, never final data."""
    basis=anchor_table(records,'basis');selection=anchor_table(records,'selection')
    cells=[];lookup={}
    for (wl,bl),b in basis.items():
        s=selection[wl,bl];ok=b['quality'] and s['quality']
        if b['signal_adc'] is not None and s['signal_adc'] is not None:
            error=s['single']['mean']-b['single']['mean']
            zero_error=s['zero']['mean']-b['zero']['mean']
            ok=ok and abs(error)<=max(2.,.05*abs(b['signal_adc'])) and abs(zero_error)<=2 and s['zero']['sd']<=2
        else:error=zero_error=None
        value=dict(wl=wl,bl=bl,basis=b,selection=s,selection_error_adc=error,
            selection_zero_error_adc=zero_error,selection_supported=bool(ok))
        cells.append(value);lookup[wl,bl]=value
    pairs=[]
    for pair_id,mapping in enumerate(plan['pairs']):
        a,b=mapping['wl'];coefficients=[]
        for j,(left,right) in enumerate(zip(mapping['positive_bl'],mapping['negative_bl'])):
            ca,cb=lookup[a,left],lookup[b,right];phases={}
            for phase in ('basis','selection'):
                x,y=ca[phase],cb[phase];valid=x['signal_adc'] is not None and y['signal_adc'] is not None
                d=x['signal_adc']-y['signal_adc'] if valid else None
                noise=sum(v['sd'] for v in (x['single'],x['zero'],y['single'],y['zero'])) if valid else None
                phases[phase]=dict(signal_adc=d,conservative_sd_adc=noise,
                    quality=bool(valid and abs(d)>=5*max(1.,noise) and noise<=.05*abs(d)))
            expected=phases['basis']['signal_adc'];actual=phases['selection']['signal_adc']
            ok=ca['selection_supported'] and cb['selection_supported'] and all(p['quality'] for p in phases.values())
            ok=ok and abs(actual-expected)<=max(2.,.05*abs(expected))
            coefficients.append(dict(logical_index=j,positive_cell=[a,left],negative_cell=[b,right],
                **phases,selection_supported=bool(ok)))
        valid=all(c['basis']['signal_adc'] is not None for c in coefficients)
        sequences={};scale=None
        if valid:
            weights=[c['basis']['signal_adc'] for c in coefficients];peak=0.
            for name in plan['patterns']:
                currents=[sum(weights[j] for j in active) for active in pattern(name,len(weights))]
                sequences[name]=currents;v=0.
                for current in currents:v=.5*v+current;peak=max(peak,abs(v))
            if peak>0:scale=1.25/peak
        signs=[c['basis']['signal_adc'] for c in coefficients if c['basis']['signal_adc'] is not None]
        pairs.append(dict(pair_id=pair_id,wl=mapping['wl'],coefficients=coefficients,
            scale=scale,expected_currents_adc=sequences,
            sign_counts=dict(positive=sum(d>0 for d in signs),negative=sum(d<0 for d in signs),zero=sum(d==0 for d in signs)),
            selection_supported=all(c['selection_supported'] for c in coefficients)))
    return dict(cells=cells,pairs=pairs,reference_data='basis only; eligibility uses basis and selection',
        scale_rule='1.25/maxabs(no-reset tau.5 over all three fixed patterns), one scale per pair',
        final_data_used=False,physical_voltage_calibrated=False)


def centered_block(before,signals,after):
    """One shared bracket per row/timestep, subtracted once for EACH partial."""
    return sum(signals)-len(signals)*(before+after)/2


def block_noise_bound(blocks):
    """Conservative sum-SD bound; the same bracket occurs k times, not sqrt(k)."""
    if len(blocks)!=5 or len({len(b['signal_adc']) for b in blocks})!=1:return None
    k=len(blocks[0]['signal_adc'])
    signal_sd=[stdev(b['signal_adc'][j] for b in blocks) for j in range(k)]
    before=stdev(b['zero_before_adc'] for b in blocks);after=stdev(b['zero_after_adc'] for b in blocks)
    return dict(signal_sd_adc=signal_sd,zero_before_sd_adc=before,zero_after_sd_adc=after,
        bracket_coefficient=k,conservative_sd_bound_adc=sum(signal_sd)+k*(before+after)/2,
        interpretation='Sum-SD bound, unknown covariance; empirical centered SD is the repeatability gate')


def evaluate(records,readbacks,blocks,reference,plan):
    reads=defaultdict(list)
    for r in readbacks:reads[r['phase'],r['wl'],r['bl']].append(r['conductance_siemens'])
    final=anchor_table(records,'final');cells=[];cell_lookup={}
    for old in reference['cells']:
        wl,bl=old['wl'],old['bl'];pre=stats(reads['pre',wl,bl]);post=stats(reads['post',wl,bl]);b=old['basis'];f=final[wl,bl]
        complete=bool(pre and post and pre['n']==post['n']==3)
        delta=(post['mean']-pre['mean'])/pre['mean'] if complete else None
        limit=max(.05,3*math.sqrt(pre['sd']**2/3+post['sd']**2/3)/pre['mean']) if complete else None
        drift=bool(complete and abs(delta)<=limit)
        final_pass=bool(f['quality'] and b['quality'] and abs(f['single']['mean']-b['single']['mean'])<=max(2.,.05*abs(b['signal_adc']))
            and abs(f['zero']['mean']-b['zero']['mean'])<=2 and f['zero']['sd']<=2)
        cell={**old,'final':f,'g_pre':pre,'g_post':post,'g_fraction_delta':delta,'g_fraction_limit':limit,
            'g_consistent':drift,'final_singleton_pass':final_pass,
            'singleton_supported':old['selection_supported'] and final_pass and drift}
        cells.append(cell);cell_lookup[wl,bl]=cell
    bykey={(b['pair_id'],b['pattern'],b['repeat'],b['t'],b['wl']):b for b in blocks}
    steps=[];traces=[];coefficient_results=[]
    for p in reference['pairs']:
        pair_id=p['pair_id'];a,b=p['wl'];coeff=p['coefficients']
        eligible=[]
        for c in coeff:
            x,y=(cell_lookup[tuple(c[k])] for k in ('positive_cell','negative_cell'))
            valid=x['final']['signal_adc'] is not None and y['final']['signal_adc'] is not None
            d=x['final']['signal_adc']-y['final']['signal_adc'] if valid else None
            noise=sum(v['sd'] for cell in (x,y) for v in (cell['final']['single'],cell['final']['zero'])) if valid else None
            target=c['basis']['signal_adc']
            supported=bool(c['selection_supported'] and x['singleton_supported'] and y['singleton_supported'] and valid
                and abs(d)>=5*max(1.,noise) and noise<=.05*abs(d) and abs(d-target)<=max(2.,.05*abs(target)))
            eligible.append(supported)
            coefficient_results.append(dict(pair_id=pair_id,**c,final_signed_adc=d,final_sd_bound_adc=noise,
                final_supported=supported))
        for name in plan['patterns']:
            masks=pattern(name,len(coeff));currents=[];complete=True
            for repeat in range(5):
                values=[]
                for t in range(12):
                    plus=bykey.get((pair_id,name,repeat,t,a));minus=bykey.get((pair_id,name,repeat,t,b))
                    if plus is None or minus is None:complete=False;values.append(None)
                    else:values.append(plus['centered_adc']-minus['centered_adc'])
                currents.append(values)
            expected=p['expected_currents_adc'].get(name)
            for t,active in enumerate(masks):
                row_tests={}
                noise_bounds={}
                for wl,key in ((a,'positive_cell'),(b,'negative_cell')):
                    target=sum(cell_lookup[tuple(coeff[j][key])]['basis']['signal_adc'] for j in active) if expected else None
                    samples=[bykey[pair_id,name,r,t,wl]['centered_adc'] for r in range(5) if (pair_id,name,r,t,wl) in bykey]
                    row_tests[str(wl)]=assess(samples,target) if target is not None else None
                    noise_bounds[str(wl)]=block_noise_bound([bykey[pair_id,name,r,t,wl] for r in range(5) if (pair_id,name,r,t,wl) in bykey])
                signed=assess([v[t] for v in currents if v[t] is not None],expected[t]) if expected else None
                support=all(eligible[j] for j in active) if active else all(eligible)
                denom=sum(abs(coeff[j]['basis']['signal_adc']) for j in active) if expected else None
                cancellation=abs(expected[t])/denom if denom else None
                signed_status='inconclusive' if not support or not signed or len(signed['samples_adc'])!=5 else 'pass' if signed['passes'] else 'fail'
                row_status='inconclusive' if not support or any(v is None or not v['statistics'] or v['statistics']['n']!=5 for v in row_tests.values()) else 'pass' if all(v['passes'] for v in row_tests.values()) else 'fail'
                steps.append(dict(pair_id=pair_id,pattern=name,t=t,active_logical_indices=active,rows=row_tests,signed=signed,
                    expected_cancellation_ratio=cancellation,selected_coefficients_supported=support,
                    row_noise_bounds=noise_bounds,signed_sd_bound_adc=sum(v['conservative_sd_bound_adc'] for v in noise_bounds.values()) if all(noise_bounds.values()) else None,
                    row_status=row_status,signed_status=signed_status,
                    status='inconclusive' if 'inconclusive' in (row_status,signed_status) else 'pass' if row_status==signed_status=='pass' else 'fail'))
            for polarity in (1,-1):
                scale=p['scale'];ref=lif([polarity*scale*x for x in expected]) if scale is not None and expected else None
                useful=bool(ref and any(v['spike'] for v in ref) and any(not v['spike'] for v in ref))
                actual=[lif([polarity*scale*x for x in values]) for values in currents] if complete and scale is not None else []
                errors=[max(abs(x[k]-y[k]) for x,y in zip(trace,ref) for k in ('before','after')) for trace in actual] if ref else []
                matching=[all(x['spike']==y['spike'] for x,y in zip(trace,ref)) for trace in actual] if ref else []
                ready=complete and useful and all(eligible)
                mvm_pass=all(s['status']=='pass' for s in steps if s['pair_id']==pair_id and s['pattern']==name)
                numerical=bool(complete and useful and matching and all(matching) and max(errors)<=.05)
                traces.append(dict(pair_id=pair_id,pattern=name,polarity=polarity,scale=scale,expected=ref,actual=actual,
                    reference_excited_and_silent=useful,all_coefficients_supported=all(eligible),max_membrane_errors=errors,
                    exact_spikes_by_repeat=matching,numerical_lif_status='inconclusive' if not complete or not useful else 'pass' if numerical else 'fail',
                    mvm_steps_pass=mvm_pass,operation_qualified=ready and mvm_pass and numerical,
                    status='inconclusive' if not ready else 'pass' if numerical and mvm_pass else 'fail'))
    return dict(cells=cells,coefficients=coefficient_results,steps=steps,lif=traces,
        singleton_supported=sum(c['singleton_supported'] for c in cells),
        mapped_contrasts_supported=sum(c['final_supported'] for c in coefficient_results),
        step_statuses=dict(Counter(s['status'] for s in steps)),lif_statuses=dict(Counter(t['status'] for t in traces)),
        analog_joint_mvm_tested=False,physical_lif_tested=False,cifar_tested=False,
        limitations=['Only the fixed three patterns were independently remeasured; not all possible logical inputs.',
            'Digital polarity reversal does not by itself validate mixed-sign cancellation.',
            'No programming or target trained-weight representation was tested.'])
