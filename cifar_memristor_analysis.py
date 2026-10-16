"""Offline arithmetic for measured CIFAR Conv1; never performs device I/O."""
import math
import statistics as st


def apply_matrix(matrix,values):
    if len(values)!=2 or len(matrix)!=2 or any(len(row)!=2 for row in matrix):raise ValueError('Expected a two-channel linear correction')
    out=[sum(a*b for a,b in zip(row,values)) for row in matrix]
    if not all(math.isfinite(v) for v in out):raise ValueError('Nonfinite corrected output')
    return out


def select_matrix(train,selection,adc_unit_v):
    """Only calibration observations enter fitting/selection; zero offset is fixed."""
    import numpy as np
    x=np.array([r['measured_mean_v'] for r in train],dtype=float)
    y=np.array([r['reference_v'] for r in train],dtype=float)
    if x.ndim!=2 or x.shape[1]!=2 or y.shape!=x.shape or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError('Incomplete/nonfinite matrix calibration')
    if not selection or not math.isfinite(adc_unit_v) or adc_unit_v<=0:raise ValueError('Missing selection/ADC resolution')
    for record in selection:
        samples=record['measured_repeats_v'];reference=record['reference_v']
        if len(samples)!=3 or any(len(row)!=2 for row in samples) or len(reference)!=2 or not all(math.isfinite(v) for row in samples+[reference] for v in row):
            raise ValueError('Incomplete/nonfinite independent selection')
    rank=int(np.linalg.matrix_rank(x));condition=float(np.linalg.cond(x)) if rank==2 else None
    candidates=[('identity',np.eye(2),True)]
    diagonal=[]
    for i in range(2):
        denominator=float(x[:,i]@x[:,i]);diagonal.append(float(x[:,i]@y[:,i])/denominator if denominator>0 else 1.)
    candidates.append(('diagonal',np.diag(diagonal),all(v>0 for v in diagonal)))
    if rank==2 and condition<=1000:
        ridge=.01*float(np.trace(x.T@x))/2
        correction=np.linalg.solve(x.T@x+ridge*np.eye(2),x.T@y).T
        candidates.append(('full_ridge_0.01',correction,True))
    reports=[]
    for name,matrix,eligible in candidates:
        singular=np.linalg.svd(matrix,compute_uv=False);gain=float(singular[0]);minimum=float(singular[-1])
        candidate_condition=gain/minimum if minimum>0 else None
        eligible=eligible and math.isfinite(gain) and gain<=10 and minimum>=.1 and candidate_condition<=1000
        errors=[];passes=True;checks=[]
        for record in selection:
            corrected=[apply_matrix(matrix.tolist(),row) for row in record['measured_repeats_v']]
            for i in range(2):
                samples=[row[i] for row in corrected];mean=st.mean(samples);sd=st.stdev(samples)
                target=record['reference_v'][i];tolerance=max(2*adc_unit_v,.05*abs(target))
                error=mean-target;ok=abs(error)<=tolerance and sd<=tolerance
                checks.append(dict(channel=i,error_v=error,sd_v=sd,tolerance_v=tolerance,passes=ok))
                errors.append(st.mean((sample-target)**2 for sample in samples)/(tolerance*tolerance));passes=passes and ok
        reports.append(dict(name=name,matrix=matrix.tolist(),operator_gain=gain,eligible=bool(eligible),
            singular_values=singular.tolist(),matrix_condition=candidate_condition,
            selection_pass=bool(eligible and passes),selection_score=st.mean(errors),checks=checks))
    eligible=[r for r in reports if r['eligible']]
    chosen=min(eligible,key=lambda r:(r['selection_score'],r['name']!='identity',r['name'])) if eligible else reports[0]
    return dict(selected_model=chosen['name'],matrix=chosen['matrix'],offset_v=[0.,0.],
        selection_pass=chosen['selection_pass'],identity_fallback=not eligible,candidates=reports,
        selection_score_definition='mean_over_channels_and_probes(mean_over_repeats((corrected_sample-fixed_target)^2)/tolerance_squared)',
        training_design_rank=rank,training_design_condition=condition,
        inputs_only_measured_two_channel_voltage=True,uses_cifar_or_trained_targets=False)


def scalar_calibration(weights, differences, adc_floor_v):
    """Five independent anchor repeats; one positive scale, no per-tap correction."""
    if len(differences) != 5 or any(len(row) != len(weights) for row in differences):
        raise ValueError('Incomplete calibration anchors')
    if not all(math.isfinite(v) for row in differences for v in row):
        raise ValueError('Nonfinite calibration')
    if not all(math.isfinite(w) for w in weights) or not math.isfinite(adc_floor_v) or adc_floor_v<=0:
        raise ValueError('Invalid weights or quantization reference')
    denominator = sum(w*w for w in weights)
    if denominator <= 0: raise ValueError('No nonzero trained weights')
    repeats = [sum(w*d for w,d in zip(weights,row))/denominator for row in differences]
    alpha, sd = st.mean(repeats), st.stdev(repeats)
    means = [st.mean(row[i] for row in differences) for i in range(len(weights))]
    rms = math.sqrt(st.mean(d*d for d in means))
    usable = math.isfinite(alpha) and alpha > 0 and alpha > 5*sd/math.sqrt(5) and rms > 0
    effective = [d/alpha for d in means] if usable else None
    residual = [e-w for e,w in zip(effective,weights)] if usable else None
    return dict(alpha_v_per_weight=alpha,alpha_repeat_values=repeats,alpha_sd=sd,
                alpha_se=sd/math.sqrt(5),heuristic_resolution_5se=alpha>5*sd/math.sqrt(5),
                difference_rms_v=rms,five_adc_quality_reference_v=adc_floor_v,rms_at_least_five_adc=rms>=adc_floor_v,
                quantization_uncertainty_not_in_sample_sd=True,decode_usable=usable,
                alpha_cv=sd/abs(alpha) if alpha else None,alpha_cv_within_5pct=alpha>0 and sd/alpha<=.05,
                measured_difference_mean_v=means,effective_weights=effective,weight_residuals=residual,
                weight_rmse=math.sqrt(st.mean(x*x for x in residual)) if residual else None,
                sign_mismatch_count=sum(w*d<0 for w,d in zip(weights,means)),
                zero_difference_count=sum(d==0 for d in means),
                calibration_uses_cifar=False,alpha_is_supervised_common_units_fit_to_trained_weights=True,confidence_claim=False)


def temporal_response(currents, bntt, filter_id):
    if len(currents)!=12 or any(x is None or not math.isfinite(x) for x in currents):
        return None
    parameters=bntt['parameters_t12x64'];v=0.;out=[]
    for t,conv in enumerate(currents):
        current=(conv-parameters['running_mean'][t][filter_id])/math.sqrt(
            parameters['running_var'][t][filter_id]+bntt['eps'])*parameters['weight'][t][filter_id]+parameters['bias'][t][filter_id]
        before=.5*v+current;spike=int(before>=1.);v=0. if spike else before
        out.append(dict(t=t,conv_sum=conv,bntt_current=current,lif_before_reset=before,lif_after_reset=v,lif_spike=spike))
    return out


def aggregate(tile_results,routing,fixture):
    """Measured contributions only in actual branch. W_eff is a separate reference."""
    indexed={(r['filter_id'],r['tile_id']):r for r in tile_results}
    if len(indexed)!=len(tile_results):raise ValueError('Duplicate tile results')
    outputs=[]
    for f in routing['filters']:
        fid=f['filter_id'];needed=[indexed.get((fid,t['tile_id'])) for t in f['tiles']]
        if any(r is None for r in needed): continue
        for ci,case in enumerate(fixture['cases']):
            for repeat in range(5):
                actual=[0.]*12;effective=[0.]*12;complete=True
                for tile,result in zip(f['tiles'],needed):
                    ledger={(p['case'],p['repeat'],p['t']):p for p in result['partials']}
                    if len(ledger)!=600 or len(result['partials'])!=600 or set(ledger)!={(c,r,t) for c in range(10) for r in range(5) for t in range(12)}:
                        raise ValueError('Incomplete or duplicate tile trajectories')
                    for t in range(12):
                        p=ledger.get((ci,repeat,t))
                        if p is None:raise ValueError('Missing measured timestep; no zero fill')
                        value=p['trained_unit_contribution']
                        if value is None:complete=False
                        else:actual[t]+=value
                        weights=result['calibration']['effective_weights']
                        if weights is None:complete=False
                        else:
                            effective[t]+=sum(w*case['spikes_t12x75'][t][s['tap_index']]
                                for w,s in zip(weights,[s for s in tile['slots'] if not s['padding']]))
                measured=temporal_response(actual,fixture['bntt'],fid) if complete else None
                eff=temporal_response(effective,fixture['bntt'],fid) if complete else None
                golden=case['reference_t12x64']
                comparisons={}
                if measured is not None:
                    for name,reference in [('trained',None),('effective',eff)]:
                        conv_ref=[golden['conv_sum'][t][fid] if reference is None else reference[t]['conv_sum'] for t in range(12)]
                        spike_ref=[golden['lif_spikes'][t][fid] if reference is None else reference[t]['lif_spike'] for t in range(12)]
                        comparisons[name]=dict(conv_rmse=math.sqrt(st.mean((actual[t]-conv_ref[t])**2 for t in range(12))),
                            spike_mismatches=sum(measured[t]['lif_spike']!=spike_ref[t] for t in range(12)))
                        for key in ('bntt_current','lif_before_reset','lif_after_reset'):
                            ref=[golden[key][t][fid] if reference is None else reference[t][key] for t in range(12)]
                            comparisons[name][key+'_max_abs_error']=max(abs(measured[t][key]-ref[t]) for t in range(12))
                outputs.append(dict(filter_id=fid,case=ci,original_test_index=case['original_test_index'],repeat=repeat,
                    status='measured' if complete else 'inconclusive_unresolved_alpha',actual=measured,
                    effective_weight_reference=eff,comparison=comparisons))
    return dict(trajectories=outputs,arithmetic='float64; golden reference is CPU float32',
                actual_branch_uses_software_dot_product=False,full_network_accuracy_claim=False)
