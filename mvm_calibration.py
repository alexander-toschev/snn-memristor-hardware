"""Offline calibration: measured eight-output vectors in, corrected outputs out.

Input bits are used only to form independent references, never by apply().
Fit, selection and final acquisitions are disjoint. No hardware imports.
"""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np


def groups(vectors,phase):
    result=defaultdict(list)
    for v in vectors:
        if v['phase']==phase:result[tuple(v['mask'])].append(v['adc'])
    return {k:np.asarray(v,dtype=float) for k,v in result.items()}


def apply(candidate,measured,zero):
    """The sole inference transform. No mask, ideal dot product, or lookup here."""
    return (np.asarray(measured)-zero)@candidate['C'].T+candidate['d']


def reference(H,mask):
    return H[:,list(mask)].sum(axis=1) if mask else np.zeros(8)


def candidate_models(fit,H,zero,settings):
    complete={k:v for k,v in fit.items() if v.shape==(5,8)}
    candidates=[dict(name='identity',C=np.eye(8),d=np.zeros(8),eligible=True,gain=1.,rank=8,condition=1.)]
    if not complete:return candidates
    X=np.asarray([v.mean(axis=0)-zero for v in complete.values()])
    Y=np.asarray([reference(H,k) for k in complete])
    scale=np.std(X,axis=0);scale=np.where(scale>0,scale,1.)
    Xn=X/scale;Xc=Xn-Xn.mean(axis=0)
    rank=int(np.linalg.matrix_rank(Xc));condition=float(np.linalg.cond(Xc))
    for kind in ('diagonal','full'):
        for ridge in settings['ridge']:
            C=np.zeros((8,8));d=np.zeros(8);eligible=True
            for row in range(8):
                columns=[row] if kind=='diagonal' else list(range(8))
                A=Xn[:,columns];Ac=A-A.mean(axis=0);yc=Y[:,row]-Y[:,row].mean()
                if np.linalg.matrix_rank(Ac)<len(columns):eligible=False
                augmented=np.vstack((Ac,np.sqrt(ridge)*np.eye(len(columns))))
                beta=np.linalg.lstsq(augmented,np.r_[yc,np.zeros(len(columns))],rcond=None)[0]
                C[row,columns]=beta/scale[columns]
                d[row]=Y[:,row].mean()-X.mean(axis=0)@C[row]
            gain=float(np.linalg.norm(C,2))
            if kind=='full' and (rank<8 or not np.isfinite(condition) or condition>settings['maximum_condition']):eligible=False
            eligible=bool(eligible and gain<=settings['maximum_gain'] and np.isfinite(C).all() and np.isfinite(d).all())
            candidates.append(dict(name=f'{kind}_ridge_{ridge:g}',C=C,d=d,eligible=eligible,gain=gain,
                rank=rank,condition=condition if np.isfinite(condition) else None))
    return candidates


def assess(candidate,data,H,zero,settings):
    points=[]
    for mask,samples in data.items():
        if samples.shape!=(5,8):continue
        corrected=apply(candidate,samples,zero);target=reference(H,mask)
        mu=corrected.mean(axis=0);sd=corrected.std(axis=0,ddof=1)
        tolerance=np.maximum(settings['absolute_error_adc'],settings['relative_error']*np.abs(target))
        noise_tolerance=np.maximum(settings['absolute_error_adc'],settings['maximum_cv']*np.abs(target))
        for wl in range(8):
            points.append(dict(mask=list(mask),wl=wl,expected_adc=float(target[wl]),mean_adc=float(mu[wl]),
                sd_adc=float(sd[wl]),signed_error_adc=float(mu[wl]-target[wl]),
                tolerance_adc=float(tolerance[wl]),noise_tolerance_adc=float(noise_tolerance[wl]),
                mean_pass=bool(abs(mu[wl]-target[wl])<=tolerance[wl]),
                repeatability_pass=bool(sd[wl]<=noise_tolerance[wl]),
                pointwise_errors_adc=(corrected[:,wl]-target[wl]).tolist()))
    return points


def select(candidates,selection,anchors,H,zero,settings):
    """No test data argument: final results cannot influence model selection."""
    scores=[]
    for c in candidates:
        mixed=assess(c,selection,H,zero,settings);anchor=assess(c,anchors,H,zero,settings)
        allpoints=mixed+anchor
        score=float(np.mean([(p['signed_error_adc']/p['tolerance_adc'])**2 for p in allpoints])) if allpoints else None
        failures=sum(not(p['mean_pass'] and p['repeatability_pass']) for p in allpoints)
        full_anchors=len(anchor)==33*8
        qualifies=bool(c['eligible'] and full_anchors and len(mixed)==settings['selection_required_sparse_masks']*8 and failures==0)
        scores.append(dict(name=c['name'],eligible=c['eligible'],selection_pass=qualifies,
            normalized_mse=score,failed_points=failures,anchor_points=len(anchor),mixed_points=len(mixed)))
    eligible=[(i,s) for i,s in enumerate(scores) if s['eligible'] and s['normalized_mse'] is not None]
    index=min(eligible,key=lambda p:(not p[1]['selection_pass'],p[1]['failed_points'],p[1]['normalized_mse']))[0] if eligible else 0
    return candidates[index],scores,index


def conductance(readbacks,wl,bl):
    arrays=[np.asarray([r['conductance_siemens'] for r in readbacks if r['phase']==phase and r['wl']==wl and r['bl']==bl]) for phase in ('pre','post')]
    if any(len(a)!=3 for a in arrays):return dict(status='unmeasured',consistent=False)
    pre,post=arrays;mu=pre.mean();delta=(post.mean()-mu)/mu
    limit=max(.05,3*np.sqrt(pre.var(ddof=1)/3+post.var(ddof=1)/3)/mu)
    return dict(status='consistent' if abs(delta)<=limit else 'changed',consistent=bool(abs(delta)<=limit),
        pre_mean_s=float(mu),post_mean_s=float(post.mean()),fractional_delta=float(delta),tolerance=float(limit))


def signed_assess(candidate,data,H,zero,settings):
    points=[]
    for mask,samples in data.items():
        if samples.shape!=(5,8):continue
        corrected=apply(candidate,samples,zero);target=reference(H,mask)
        for pair in range(4):
            a,b=2*pair,2*pair+1;values=corrected[:,a]-corrected[:,b];expected=target[a]-target[b]
            tolerance=max(2.,.05*abs(expected));mu=float(values.mean());sd=float(values.std(ddof=1))
            points.append(dict(mask=list(mask),pair=[a,b],expected_adc=float(expected),mean_adc=mu,
                sd_adc=sd,signed_error_adc=float(mu-expected),tolerance_adc=float(tolerance),
                mean_pass=bool(abs(mu-expected)<=tolerance),repeatability_pass=bool(sd<=tolerance),
                pointwise_errors_adc=(values-expected).tolist()))
    return points


def partial_cells(basis,reads):
    """Preserve observed addresses even when one missing vector prevents full-C fit."""
    zero=basis.get(())
    result=[]
    for wl in range(8):
        for bl in range(32):
            values=basis.get((bl,))
            row=dict(wl=wl,bl=bl,status='unmeasured' if values is None else 'basis_observed_calibration_incomplete',
                conductance=conductance(reads,wl,bl),basis_count=0 if values is None else len(values),
                arbitrary_mixed_qualified=False,programmability_tested=False)
            if values is not None:
                row['basis_mean_adc']=float(values[:,wl].mean())
                row['basis_sd_adc']=float(values[:,wl].std(ddof=1)) if len(values)>1 else None
                if zero is not None:row['signal_adc']=float(values[:,wl].mean()-zero[:,wl].mean())
            result.append(row)
    return result


def path_probes(vectors,plan):
    """Post-test address-path evidence; never used to fit/select C or diagnose cells."""
    data=groups(vectors,'input_path_probe');anchors=groups(vectors,'test_anchors');result=[]
    for anchor in plan['path_probe_anchors']:
        for bl in plan['path_probe_columns']:
            mask=tuple(sorted((anchor,bl)));keys=(mask,(anchor,),(bl,),())
            samples=[data.get(mask),anchors.get((anchor,)),anchors.get((bl,)),anchors.get(())]
            for wl in range(8):
                row=dict(wl=wl,anchor_bl=anchor,probe_bl=bl,status='inconclusive',reference_phase='test_anchors',
                    physical_cause=None,used_for_model_selection=False)
                if any(s is None or s.shape!=(5,8) for s in samples):
                    row['reason']='missing_probe_or_independent_anchor';result.append(row);continue
                values=[s[:,wl] for s in samples];means=[float(v.mean()) for v in values];sds=[float(v.std(ddof=1)) for v in values]
                p,a,b,z=means;increment=p-a;single=b-z;epsilon=p-a-b+z
                se_increment=float(np.hypot(sds[0],sds[1])/np.sqrt(5));se_epsilon=float(np.linalg.norm(sds)/np.sqrt(5))
                row.update(status='observed',pair_adc=values[0].tolist(),anchor_adc=values[1].tolist(),
                    single_adc=values[2].tolist(),zero_adc=values[3].tolist(),mean_adc=means,sd_adc=sds,
                    increment_adc=increment,zero_subtracted_single_adc=single,epsilon_adc=epsilon,
                    se_increment_adc=se_increment,se_epsilon_adc=se_epsilon,
                    increment_distinguishable=bool(abs(increment)>max(2.,3*se_increment)),
                    epsilon_distinguishable=bool(abs(epsilon)>max(2.,3*se_epsilon)),
                    interpretation='Descriptive 3SE comparison; correlations, clipping and time drift unresolved. No proof of correct physical addressing or a dead cell.')
                row['task_ids']=[[v['task_ids'][wl] for v in vectors if v['phase']==phase and tuple(v['mask'])==key]
                    for phase,key in zip(('input_path_probe','test_anchors','test_anchors','test_anchors'),keys)]
                result.append(row)
    return result


def lif(currents):
    v=0.;pre=[];post=[];spikes=[]
    for current in currents:
        u=.5*v+float(current);spike=int(u>=1.);v=0. if spike else u
        pre.append(u);post.append(v);spikes.append(spike)
    return dict(pre=pre,post=post,spikes=spikes)


def lif_tests(vectors,plan,H,zero,chosen,basis,reads,pair_selection):
    rows={(v['group_id'],v['pattern'],v['repeat'],v['t']):v['adc'] for v in vectors if v['phase']=='synthetic_lif'}
    result=[]
    for group_id,columns in enumerate(plan['synthetic_groups']):
        for pair in range(4):
            wl0,wl1=2*pair,2*pair+1;diff=H[wl0]-H[wl1]
            diff_noise=np.asarray([np.std(basis[(bl,)][:,wl0]-basis[(bl,)][:,wl1],ddof=1) for bl in columns])
            zero_noise=float(np.std(basis[()][:,wl0]-basis[()][:,wl1],ddof=1))
            noise=np.hypot(diff_noise,zero_noise)
            basis_excited=bool(np.any(np.abs(diff[columns])>=5*np.maximum(1.,noise)))
            drift_ok=all(conductance(reads,w,b)['consistent'] for w in (wl0,wl1) for b in columns)
            expected_unscaled={name:[float(sum(diff[columns[j]] for j in step)) for step in pattern]
                for name,pattern in plan['synthetic_patterns'].items()}
            maximum=0.
            for currents in expected_unscaled.values():
                linear_v=0.
                for current in currents:linear_v=.5*linear_v+current;maximum=max(maximum,abs(linear_v))
            scale=1.25/maximum if maximum>0 else None
            for name,unscaled in expected_unscaled.items():
              for polarity in (1,-1):
                record=dict(group_id=group_id,columns=columns,pattern=name,pair=[wl0,wl1],polarity=polarity,status='inconclusive',
                    label='current-state synthetic normalized LIF; identity BN, not trained CIFAR Conv1')
                record.update(differential_basis_signal=diff[columns].tolist(),differential_basis_noise=noise.tolist(),
                    differential_basis_excited=basis_excited,selection_pass=pair_selection[pair],
                    normalization='1.25/maxabs(no-reset tau.5 reference over both fixed patterns); basis only',
                    no_reset_reference_peak=maximum,scale=scale,conductance_consistent=drift_ok)
                if scale is None:record['reason']='zero_calibration_scale';result.append(record);continue
                if not basis_excited or not pair_selection[pair]:
                    record['reason']='low_differential_basis_or_unqualified_selection';result.append(record);continue
                expected=[polarity*scale*current for current in unscaled]
                target=lif(expected);record.update(expected=target,expected_currents=expected)
                if not any(target['spikes']) or all(target['spikes']):record['reason']='expected_sequence_not_exciting';result.append(record);continue
                tests=[]
                for repeat in range(5):
                    if any((group_id,name,repeat,t) not in rows for t in range(12)):
                        tests.append(dict(repeat=repeat,status='inconclusive',reason='missing_measured_timestep'));continue
                    currents=[]
                    for t in range(12):
                        measured=apply(chosen,rows[group_id,name,repeat,t],zero)
                        currents.append(float(polarity*scale*(measured[wl0]-measured[wl1])))
                    actual=lif(currents)
                    error=max(abs(a-b) for field in ('pre','post') for a,b in zip(actual[field],target[field]))
                    tests.append(dict(repeat=repeat,status='pass' if actual['spikes']==target['spikes'] and error<=.05 else 'fail',
                        maximum_membrane_error=error,actual=actual,currents=currents))
                record['repeats']=tests
                record['threshold_margin']=float(min(abs(v-1.) for v in target['pre']))
                if any(t['status']=='inconclusive' for t in tests):record['reason']='incomplete_trajectory'
                else:record['status']='pass' if all(t['status']=='pass' for t in tests) and drift_ok else 'fail'
                record['conductance_consistent']=drift_ok;result.append(record)
    return result


def analyze(observations,plan):
    vectors,reads=observations['vectors'],observations['readbacks'];settings=plan['calibration']
    basis=groups(vectors,'basis');cells=[]
    if () not in basis or basis[()].shape!=(5,8):
        return dict(status='inconclusive',reason='missing_zero_basis',cells=partial_cells(basis,reads))
    zero=basis[()].mean(axis=0);sd0=basis[()].std(axis=0,ddof=1);H=np.zeros((8,32));quality=np.zeros((8,32),bool);complete_basis=True
    for bl in range(32):
        samples=basis.get((bl,))
        if samples is None or samples.shape!=(5,8):complete_basis=False;continue
        H[:,bl]=samples.mean(axis=0)-zero
        noise=np.hypot(samples.std(axis=0,ddof=1),sd0)
        quality[:,bl]=(H[:,bl]>0)&(H[:,bl]/np.maximum(1.,noise)>=5)&(noise<=.05*H[:,bl])
    # Do not fabricate missing basis coefficients as measured zero.
    if not complete_basis:
        return dict(status='inconclusive',reason='missing_single_basis',cells=partial_cells(basis,reads))
    candidates=candidate_models(groups(vectors,'fit'),H,zero,settings)
    fit_means=np.asarray([v.mean(axis=0) for v in groups(vectors,'fit').values() if v.shape==(5,8)])
    fit_rank=int(np.linalg.matrix_rank(fit_means-fit_means.mean(axis=0))) if len(fit_means)>0 else 0
    chosen,scores,index=select(candidates,groups(vectors,'selection'),groups(vectors,'selection_anchors'),H,zero,settings)
    selection_points=assess(chosen,groups(vectors,'selection'),H,zero,settings)
    selection_anchors=assess(chosen,groups(vectors,'selection_anchors'),H,zero,settings)
    selection_by_wl=[]
    for wl in range(8):
        mixed=[p for p in selection_points if p['wl']==wl];anchor=[p for p in selection_anchors if p['wl']==wl]
        selection_by_wl.append(dict(wl=wl,mixed_count=len(mixed),anchor_count=len(anchor),
            passed=bool(chosen['eligible'] and len(mixed)==settings['selection_required_sparse_masks'] and len(anchor)==33 and all(p['mean_pass'] and p['repeatability_pass'] for p in mixed+anchor))))
    signed_selection=signed_assess(chosen,groups(vectors,'selection'),H,zero,settings)
    signed_selection_anchors=signed_assess(chosen,groups(vectors,'selection_anchors'),H,zero,settings)
    pair_selection=[]
    for pair in range(4):
        points=[p for p in signed_selection+signed_selection_anchors if p['pair']==[2*pair,2*pair+1]]
        pair_selection.append(bool(all(selection_by_wl[w]['passed'] for w in (2*pair,2*pair+1)) and len(points)==33+settings['selection_required_sparse_masks'] and all(p['mean_pass'] and p['repeatability_pass'] for p in points)))
    tests=assess(chosen,groups(vectors,'test'),H,zero,settings)
    anchors=assess(chosen,groups(vectors,'test_anchors'),H,zero,settings)
    for wl in range(8):
        for bl in range(32):
            samples=basis[bl,][:,wl];signal=H[wl,bl];noise=float(np.hypot(samples.std(ddof=1),sd0[wl]));drift=conductance(reads,wl,bl)
            anchor=next((p for p in anchors if p['mask']==[bl] and p['wl']==wl),None)
            measured=[p for p in tests if bl in p['mask'] and p['wl']==wl]
            good=bool(selection_by_wl[wl]['passed'] and quality[wl,bl] and drift['consistent'] and anchor and anchor['mean_pass'] and anchor['repeatability_pass'])
            cells.append(dict(wl=wl,bl=bl,source_scope='current conductances; no programming',
                status='single_response_supported' if good else 'uncertain_or_failed',zero_adc=float(zero[wl]),
                basis_mean_adc=float(samples.mean()),basis_sd_adc=float(samples.std(ddof=1)),signal_adc=float(signal),
                snr=float(signal/max(1.,noise)),combined_sd_adc=noise,cv=float(noise/signal) if signal>0 else None,
                basis_quality=bool(quality[wl,bl]),row_selection_pass=selection_by_wl[wl]['passed'],conductance=drift,heldout_single=anchor,
                tested_mixed_points=len(measured),mixed_points_pass=sum(p['mean_pass'] and p['repeatability_pass'] for p in measured),
                arbitrary_mixed_qualified=False,programmability_tested=False,
                physical_failure_diagnosis=None,input_path_unresolved=bool(signal==0)))
    drift_by_pair=[all(c['conductance']['consistent'] for c in cells if c['wl'] in (2*p,2*p+1)) for p in range(4)]
    lif_result=lif_tests(vectors,plan,H,zero,chosen,basis,reads,pair_selection)
    signed=signed_assess(chosen,groups(vectors,'test'),H,zero,settings)
    signed_anchors=signed_assess(chosen,groups(vectors,'test_anchors'),H,zero,settings)
    counts=defaultdict(lambda:dict(points=0,mean_pass=0,repeatability_pass=0))
    for p in tests:
        bucket=counts[len(p['mask'])];bucket['points']+=1;bucket['mean_pass']+=p['mean_pass'];bucket['repeatability_pass']+=p['repeatability_pass']
    return dict(status='analyzed',selected_model=chosen['name'],selection=scores,
        selected_by_selection_only=True,selection_pass=scores[index]['selection_pass'],
        selection_by_wl=selection_by_wl,selection_by_signed_pair=pair_selection,
        selection_pass_scope='eight row outputs, selection mixed vectors and independent zero/single anchors; signed pairs evaluated separately',
        C=chosen['C'].tolist(),d=chosen['d'].tolist(),zero=zero.tolist(),basis_H=H.tolist(),
        basis_rank=int(np.linalg.matrix_rank(H)),fit_design_rank=fit_rank,full_matrix_required_rank=8,
        candidate_metadata=[{k:v for k,v in c.items() if k not in ('C','d')} for c in candidates],
        cells=cells,final_mixed=tests,final_anchors=anchors,final_signed_pairs=signed,final_signed_anchors=signed_anchors,
        input_path_probes=path_probes(vectors,plan),
        conductance_consistent_by_pair=drift_by_pair,coverage_by_density=dict(counts),lif=lif_result,
        all256_qualified=False,full_cifar_qualified=False,source_states_matched=False,
        uncertainty='Fixed 5%/2 ADC error and repeatability bounds; no confidence claim from shared-zero or fitted-parameter uncertainty',
        correction_interpretation='C maps eight measured outputs, not 256 independently programmable weights',
        skipped_vectors=len(observations.get('skips',[])))


def verify_acquisition(session,observations):
    """Reconcile paired immutable raw journal with compact analysis inputs."""
    from read_all import sha256
    manifest=json.loads((session/'manifest.json').read_bytes())
    for name in ('raw.jsonl','observations.json','protocol.json','schedule.json'):
        if sha256(session/name)!=manifest['result_sha256'][name]:raise ValueError('Manifest result hash mismatch: '+name)
    expected=json.loads((session/'schedule.json').read_bytes());lookup={e['vector_id']:e for e in expected}
    pending=None;sequence=0;responses=[];guards=[]
    with (session/'raw.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line)
            if row.get('session_id')!=manifest['session_id'] or row.get('source')!=manifest['source']:raise ValueError('Raw session/source mismatch')
            if row['event']=='vector_guard':
                meta=lookup.get(row['vector_id'])
                if meta is None or any(row[k]!=v for k,v in meta.items()):raise ValueError('Guard schedule mismatch')
                guards.append(row)
            elif row['event']=='request':
                if pending is not None:raise ValueError('Unpaired request')
                sequence+=1
                if row['task_id']!=sequence or row['mode'] not in (9,10):raise ValueError('Task sequence/mode mismatch')
                command=row['command_si']
                if command['mode_flag']!=row['mode'] or command['wl']!=row['wl']:raise ValueError('Command metadata mismatch')
                if row['mode']==10:
                    if command['vol']!=[.1 if bl in row['mask'] else 0. for bl in range(32)] or len(row['mask'])>4 or command['id']!=0:raise ValueError('MVM payload mismatch')
                    meta=lookup.get(row['vector_id'])
                    if meta is None or any(row[k]!=v for k,v in meta.items()):raise ValueError('Impact schedule mismatch')
                elif command['vol']!=.1 or command['bl']!=row['bl']:raise ValueError('Readback payload mismatch')
                pending=row
            elif row['event']=='result':
                if pending is None or row['task_id']!=pending['task_id'] or row['command_si']!=pending['command_si']:raise ValueError('Request/result mismatch')
                if row['status']=='ok':
                    adc=row['raw_adc']
                    if type(adc)is not int or not 0<=adc<=16382 or row['returned_id']!=0 or row['native_result'][2]!=adc:raise ValueError('Native result mismatch')
                    responses.append(row)
                pending=None
            else:raise ValueError('Unknown raw event')
    if pending is not None:raise ValueError('Missing result journal entry')
    if sequence!=manifest['impact_attempts']:raise ValueError('Manifest impact count mismatch')
    reconstructed=[]
    for meta in expected:
        rows=[r for r in responses if r['mode']==10 and r['vector_id']==meta['vector_id']]
        if len(rows)==8:
            if [r['wl'] for r in rows]!=list(range(8)):raise ValueError('WL vector order mismatch')
            reconstructed.append(dict(**meta,adc=[r['raw_adc'] for r in rows],task_ids=[r['task_id'] for r in rows]))
        elif rows and manifest['status']=='completed':raise ValueError('Partial vector in completed session')
    if reconstructed!=observations['vectors']:raise ValueError('Compact MVM observations differ from raw')
    compact_reads=[{k:v for k,v in r.items() if k not in ('event','latency_s','command_si','native_voltage_conversion','native_result')} for r in responses if r['mode']==9]
    if compact_reads!=observations['readbacks']:raise ValueError('Compact readbacks differ from raw')
    skips=[{**lookup[r['vector_id']],'reasons':r['reasons']} for r in guards if not r['allowed']]
    if skips!=observations['skips']:raise ValueError('Compact skips differ from raw')
    if manifest['status']=='completed' and (len(compact_reads)!=1536 or len(guards)!=len(expected) or len(reconstructed)+len(skips)!=len(expected)):
        raise ValueError('Completed acquisition coverage mismatch')
    return dict(passed=True,scope='Stored result hashes, exact request/result schedule and compact/raw reconciliation; not physical calibration',
        paired_impacts=sequence,complete_vectors=len(reconstructed),skipped_vectors=len(skips))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--session',required=True);args=parser.parse_args(argv)
    from read_all import local_path,save_json,sha256
    session=local_path(args.session);output=session/'analysis'
    observations=json.loads((session/'observations.json').read_bytes());plan=json.loads((session/'protocol.json').read_bytes())
    verified=verify_acquisition(session,observations)
    result=analyze(observations,plan);result['raw_reconciliation']=verified
    result['input_sha256']={p.relative_to(local_path('.')).as_posix():sha256(p) for p in (session/'observations.json',session/'protocol.json')}
    output.mkdir();save_json(output/'analysis.json',result)
    with (output/'cells.csv').open('x',encoding='utf-8',newline='') as stream:
        fields=['wl','bl','status','signal_adc','snr','cv','basis_quality','tested_mixed_points','mixed_points_pass','input_path_unresolved']
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(result['cells'])
    print(json.dumps(dict(status=result['status'],selected_model=result.get('selected_model'),all256_qualified=False)))
    return 0


if __name__=='__main__':raise SystemExit(main())
