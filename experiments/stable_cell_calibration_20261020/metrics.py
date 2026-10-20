"""Pure calibration summaries and predeclared selection; no Connector imports."""
import itertools
import math
import statistics
from .plan import SPEC, CELLS, digest


def stats(values):
    return dict(mean=statistics.mean(values), sd=statistics.stdev(values) if len(values)>1 else 0.,
                count=len(values))


def checkpoint(brackets):
    good = [b for b in brackets if b['status'] == 'measured']
    if len(good) != SPEC['repeats']:
        return dict(status='missing', measured=len(good))
    signals = stats([b['centered_adc'] for b in good])
    zeros = stats([b['after_adc']-b['before_adc'] for b in good])
    rs = [b['resistance_ohm'] for b in good]
    return dict(status='measured', signal=signals, zero_difference=zeros,
                zero_shift_limit=max(2., 3*zeros['sd']/math.sqrt(10)),
                resistance=stats(rs), conductance_siemens=stats([1/r for r in rs]),
                q_proxy=stats([330/r for r in rs]))


def cell_summary(wl, bl, checkpoints, updates):
    own = [c for c in checkpoints if (c['wl'], c['bl']) == (wl, bl)]
    indexed = {(c['cycle'], c['state'], c['role']): c for c in own}
    row = dict(wl=wl, bl=bl, eligible=False, hard_eligible=False,
               hard_reasons=[], reasons=[], checkpoint_ids=[c['id'] for c in own])
    needed = [(cycle, state, role) for cycle in range(2) for state in ('R','M','H')
              for role in ('immediate','delay_0','delay_10','delay_60')] + [(-1,'H','final')]
    if any(k not in indexed or indexed[k]['summary']['status'] != 'measured' for k in needed):
        row['reasons'].append('missing_required_state_checkpoint')
        row['hard_reasons'].append('missing_required_state_checkpoint')
        return row
    spans = []
    for cycle in range(2):
        low = indexed[cycle,'R','immediate']['summary']['signal']
        high = indexed[cycle,'H','immediate']['summary']['signal']
        noise = math.hypot(low['sd'], high['sd'])
        span = high['mean']-low['mean']-3*noise/math.sqrt(10)
        spans.append(dict(cycle=cycle, lower_span_adc=span, noise_adc=noise))
        admitted = [u for u in updates if u['phase']=='A' and u['cycle']==cycle
                    and (u['wl'],u['bl'])==(wl,bl) and u['admitted']]
        if not all(any(u['sign']==sign for u in admitted) for sign in (0,1)):
            row['reasons'].append('no_admitted_reset_or_set_cycle_'+str(cycle))
            row['hard_reasons'].append('no_admitted_reset_or_set_cycle_'+str(cycle))
    span = min(s['lower_span_adc'] for s in spans)
    row.update(span_adc=span, spans=spans)
    if any(span < max(10.,5*s['noise_adc']) for s in spans):
        row['reasons'].append('insufficient_reproducible_span')
        row['hard_reasons'].append('insufficient_reproducible_span')
    sd_limit, drift_limit, cycle_limit = max(2.,.05*span),max(3.,.05*span),max(4.,.1*span)
    scores = []
    for key in needed:
        c = indexed[key]; summary = c['summary']
        cycle,state,role = key
        reference = indexed[cycle if cycle>=0 else 1,state,'immediate']['summary']['signal']['mean']
        drift = abs(summary['signal']['mean']-reference)
        scores.extend([summary['signal']['sd']/sd_limit, drift/drift_limit])
        scores.append(abs(summary['zero_difference']['mean'])/summary['zero_shift_limit'])
        if summary['signal']['sd']>sd_limit:row['reasons'].append('repeat_noise:'+str(key))
        if drift>drift_limit:row['reasons'].append('state_drift:'+str(key))
        if abs(summary['zero_difference']['mean'])>summary['zero_shift_limit']:
            row['reasons'].append('zero_shift:'+str(key))
    for state in ('R','M','H'):
        delta = abs(indexed[0,state,'immediate']['summary']['signal']['mean']-
                    indexed[1,state,'immediate']['summary']['signal']['mean'])
        scores.append(delta/cycle_limit)
        if delta>cycle_limit:row['reasons'].append('cross_cycle:'+state)
    for cycle in range(2):
        a,m,b = [indexed[cycle,s,'immediate']['summary']['signal']['mean'] for s in ('R','M','H')]
        scores.append(max(0.,a-m,m-b)/cycle_limit)
        if not a-cycle_limit<=m<=b+cycle_limit:row['reasons'].append('middle_outside_endpoints')
    row.update(worst_normalized_error=max(scores), eligible=not row['reasons'],
               hard_eligible=not row['hard_reasons'],
               initial_checkpoint_descriptive_only=True)
    return row


def matchings(available=tuple(range(8))):
    """All nonempty disjoint pair matchings, including unused WLs."""
    if not available:
        yield ()
        return
    first,*rest = available
    yield from matchings(tuple(rest))
    for i,second in enumerate(rest):
        for tail in matchings(tuple(rest[:i]+rest[i+1:])):
            yield ((first,second),)+tail


def select_rectangle(cells):
    by = {(c['wl'],c['bl']):c for c in cells}
    candidates = []
    for pairs in matchings():
        if not pairs:continue
        wls = sorted(w for pair in pairs for w in pair)
        bls = [b for b in range(32) if all(by[w,b]['eligible'] for w in wls)]
        if not bls:continue
        score = max(by[w,b]['worst_normalized_error'] for w in wls for b in bls)
        candidates.append(dict(pairs=[list(p) for p in pairs], bls=bls,
                               coefficient_slots=len(pairs)*len(bls),
                               worst_normalized_error=score))
    candidates.sort(key=lambda c:(-c['coefficient_slots'],c['worst_normalized_error'],
                                 c['pairs'],c['bls']))
    chosen = candidates[0] if candidates else None
    fallback=[]
    if chosen is None:
        for a,b in itertools.combinations(range(8),2):
            scored=[(max(by[a,k]['worst_normalized_error'],by[b,k]['worst_normalized_error']),k)
                    for k in range(32) if by[a,k].get('hard_eligible') and by[b,k].get('hard_eligible')]
            scored=sorted(scored)[:8]
            if scored:
                fallback.append(dict(pairs=[[a,b]],bls=sorted(k for _,k in scored),
                    coefficient_slots=len(scored),worst_normalized_error=max(s for s,_ in scored)))
        fallback.sort(key=lambda c:(-c['coefficient_slots'],c['worst_normalized_error'],c['pairs'],c['bls']))
        if fallback:chosen=fallback[0]
    return dict(candidates=candidates, selected=chosen, selection_sha256=digest(chosen),
                fallback_candidates=fallback,quality_qualified=bool(candidates),
                status='qualified' if candidates else 'diagnostic_best_available' if chosen else 'no_resolvable_rectangle',
                selection_uses='phase_A_only', covariance_inferred=False)


def qualify_pairs(selection, cells, observations):
    by = {(r['wl'],r['bl']):r for r in cells}
    reports = []
    for a,b in selection['pairs']:
        for bl in selection['bls']:
            span = min(by[a,bl]['span_adc'],by[b,bl]['span_adc'])
            reasons=[];hard_reasons=[]; configurations={}
            for config in ('plus','minus','zero','checkerboard'):
                found=[o for o in observations if o['pair']==[a,b] and o['bl']==bl and o['configuration']==config]
                keyed={o['role']:o for o in found}
                values={}
                for role in ('immediate','delayed'):
                    item=keyed.get(role)
                    if item is None or item['status']!='measured':
                        reason='missing:'+config+':'+role
                        reasons.append(reason);hard_reasons.append(reason);continue
                    s=stats(item['differences_adc']);values[role]=s
                    if s['sd']>max(2.,.05*span):reasons.append('noise:'+config+':'+role)
                    direction=1 if config=='plus' else -1 if config=='minus' else 0
                    if direction and direction*s['mean']<max(5.,5*s['sd']):
                        reasons.append('signed_response:'+config+':'+role)
                        hard_reasons.append('signed_response:'+config+':'+role)
                if len(values)==2 and abs(values['delayed']['mean']-values['immediate']['mean'])>max(3.,.05*span):
                    reasons.append('drift:'+config)
                configurations[config]=values
            reports.append(dict(pair=[a,b],bl=bl,pair_span_adc=span,reasons=reasons,
                                hard_reasons=hard_reasons,mandatory_pass=not hard_reasons,
                                qualified=not reasons,configurations=configurations))
    return dict(qualified=bool(reports) and all(r['qualified'] for r in reports),
                mandatory_pass=bool(reports) and all(r['mandatory_pass'] for r in reports),
                cells=reports, rerank=False, zero_recipe_is_not_zero_weight=True)


def fit_matrix(basis, fit, selection):
    import numpy as np
    d=len(next(iter(basis.values())))
    def arrays(rows):
        return (np.asarray([r['centered_v'] for r in rows],dtype=np.float64),
                np.asarray([sum((np.asarray(basis[b]) for b in r['mask']),np.zeros(d)) for r in rows]))
    x,y=arrays(fit);sx,sy=arrays(selection);ridge=.0001
    def spectrum(a):
        s=np.linalg.svd(a,compute_uv=False);tol=np.finfo(np.float64).eps*max(a.shape)*s[0]
        return dict(values=s.tolist(),tol=float(tol),rank=int((s>tol).sum()),
                    margin_pass=bool(len(s)>=d and s[-1]>=16*tol and s[-1]>0))
    xs,ys=spectrum(x),spectrum(y)
    norms=np.linalg.norm(x,axis=0)
    condition=float(np.linalg.cond(x/np.where(norms>0,norms,1)))
    condition=condition if math.isfinite(condition) else None
    models=[('identity',np.eye(d)),('diagonal',np.diag((x*y).sum(0)/((x*x).sum(0)+ridge)))]
    if xs['rank']==ys['rank']==d and xs['margin_pass'] and ys['margin_pass'] and condition is not None and condition<=1e6 and (norms>0).all():
        models.append(('full',np.linalg.solve(x.T@x+ridge*np.eye(d),x.T@y).T))
    candidates=[]
    for name,c in models:
        gain=float(np.max(np.sum(np.abs(c),axis=1)))
        if np.isfinite(c).all() and gain<=10:
            candidates.append(dict(name=name,matrix=c.tolist(),gain=gain,
                                   selection_rmse_v=float(np.sqrt(np.mean((sx@c.T-sy)**2)))))
    chosen=min(candidates,key=lambda c:(round(c['selection_rmse_v'],12),c['name']!='identity',c['name']))
    return dict(**chosen,candidates=candidates,x_spectrum=xs,y_spectrum=ys,
                normalized_condition=condition,intercept=False,ridge_v2=ridge,
                target='fresh_same_state_singleton_superposition',trained_weights_used=False)


def heldout(matrix, basis, rows, pairs, wls):
    import numpy as np
    c=np.asarray(matrix['matrix']);reports=[]
    for pair in pairs:
        a,b=[wls.index(w) for w in pair]
        for density in range(1,5):
            group=[r for r in rows if len(r['mask'])==density]
            if not group:
                reports.append(dict(pair=pair,density=density,status='not_tested'));continue
            errors=[];raw_errors=[];targets=[]
            for r in group:
                x=np.asarray(r['centered_v']);y=sum((np.asarray(basis[bl]) for bl in r['mask']),np.zeros(len(wls)))
                corrected=c@x;target=y[a]-y[b]
                targets.append(target);errors.append(corrected[a]-corrected[b]-target)
                raw_errors.append(x[a]-x[b]-target)
            unit=5/16383
            rmse=float(np.sqrt(np.mean(np.square(errors)))/unit)
            target=float(np.sqrt(np.mean(np.square(targets)))/unit)
            reports.append(dict(pair=pair,density=density,status='measured',count=len(group),
                corrected_rmse_adc_equivalent=rmse,raw_rmse_adc_equivalent=float(np.sqrt(np.mean(np.square(raw_errors)))/unit),
                target_rms_adc_equivalent=target,limit=max(4.,.1*target),qualified=rmse<=max(4.,.1*target)))
    return dict(qualified=bool(rows) and all(r.get('qualified',False) for r in reports if r['status']=='measured'),
                strata=reports,heldout_refit=False)


def basis_values(rows):
    import statistics
    by={}
    for row in rows:
        if any(b['status']!='measured' for b in row['brackets']):
            raise ValueError('Missing basis measurement; no zero substitution')
        if row['mask']:by.setdefault(row['mask'][0],[]).append([b['centered_v'] for b in row['brackets']])
    return {bl:[statistics.mean(v[i] for v in values) for i in range(len(values[0]))] for bl,values in by.items()}


def vector_values(rows):
    out=[]
    for row in rows:
        if any(b['status']!='measured' for b in row['brackets']):
            raise ValueError('Missing joint measurement; no zero substitution')
        out.append({k:v for k,v in row.items() if k!='brackets'} |
                   dict(centered_v=[b['centered_v'] for b in row['brackets']]))
    return out

