"""Source-specific held-out MVM and digital LIF analysis; no I/O or driver imports."""
import math
from statistics import mean, stdev


def stats(values):
    return dict(n=len(values),mean=mean(values),sd=stdev(values) if len(values)>1 else 0.)


def lif(currents):
    v=0.;trace=[]
    for current in currents:
        before=.5*v+current;spike=int(before>=1.);v=0. if spike else before
        trace.append(dict(current=current,before_reset=before,spike=spike,after_reset=v,threshold_margin=before-1.))
    return trace


def reference(cal,bits):
    """Calibration mean and SE² with shared zero's covariance retained exactly."""
    active=[1<<i for i in range(3) if bits&(1<<i)]
    if not active:return cal[0]['mean'],cal[0]['sd']**2/cal[0]['n']
    k=len(active)
    return (sum(cal[i]['mean'] for i in active)-(k-1)*cal[0]['mean'],
            sum(cal[i]['sd']**2/cal[i]['n'] for i in active)+(k-1)**2*cal[0]['sd']**2/cal[0]['n'])


def basis_quality(cal):
    result=[]
    for bit in (1,2,4):
        signal=cal[bit]['mean']-cal[0]['mean'];sd=math.hypot(cal[bit]['sd'],cal[0]['sd'])
        snr=signal/max(1.,sd);cv=sd/signal if signal>0 else None
        result.append(dict(bit=bit,signal_adc=signal,combined_sd_adc=sd,snr=snr,cv=cv,
                           passes=signal>0 and snr>=5 and cv is not None and cv<=.05))
    return result


def error_test(observed,predicted,signal,se_squared):
    tolerance=max(.05*abs(signal),3*math.sqrt(se_squared),2.)
    error=observed-predicted
    return dict(observed_adc=observed,predicted_adc=predicted,error_adc=error,
                propagated_se_adc=math.sqrt(se_squared),tolerance_adc=tolerance,passes=abs(error)<=tolerance)


def repeatability(signal,sd,expected_signal,calibration_noise,*,zero=False):
    near_zero=zero or abs(expected_signal)<=max(5.,5*calibration_noise)
    absolute_limit=max(2.,calibration_noise)
    cv=sd/abs(signal) if signal else None
    passes=sd<=absolute_limit if near_zero else cv is not None and cv<=.05
    return dict(sample_sd_adc=sd,cv=cv,near_zero=near_zero,absolute_sd_limit_adc=absolute_limit,
                passes=passes,reason=None if passes else 'validation_repetition_variance')


def evaluate(records,readbacks,plan):
    # Each phase has disjoint task IDs. Only calibration supplies coefficients.
    cal={};validation={};sequences={};rows=[];pairs=[];lif_results=[]
    for wl in range(8):
        for pedestal in (False,True):
            cal[wl,pedestal]={b:stats([r['raw_adc'] for r in records if r['phase']=='calibration' and r['wl']==wl
                  and r['pedestal']==pedestal and r['bits']==b]) for b in (0,1,2,4)}
            validation[wl,pedestal]={b:stats([r['raw_adc'] for r in records if r['phase']=='validation' and r['wl']==wl
                  and r['pedestal']==pedestal and r['bits']==b]) for b in range(8)}
            c=cal[wl,pedestal];quality=basis_quality(c)
            pedestal_clear=(not pedestal or c[0]['mean']>=max(5.,5*c[0]['sd']))
            tests=[]
            for b,v in validation[wl,pedestal].items():
                prediction,var=reference(c,b)
                trial=[r['raw_adc'] for r in records if r['phase']=='validation' and r['wl']==wl and r['pedestal']==pedestal and r['bits']==b]
                repeat=repeatability(v['mean']-c[0]['mean'],math.hypot(v['sd'],c[0]['sd']),
                    prediction-c[0]['mean'],math.sqrt(var*5),zero=b==0)
                test=error_test(v['mean'],prediction,prediction-c[0]['mean'],var+v['sd']**2/v['n'])
                tests.append({**test,'bits':b,'mean_pass':test['passes'],'repeatability':repeat,
                    'per_repeat_errors_adc':[x-prediction for x in trial],'passes':test['passes'] and repeat['passes']})
            eligible=all(x['passes'] for x in quality) and pedestal_clear
            rows.append(dict(wl=wl,pedestal=pedestal,calibration=c,basis_quality=quality,pedestal_away_from_zero=pedestal_clear,
                validation=tests,status='inconclusive' if not eligible else 'pass' if all(x['passes'] for x in tests) else 'fail'))
    for pair_id,pair in enumerate(plan['pairs']):
        a,b=pair['wl']
        for pedestal in (False,True):
            ca,cb=cal[a,pedestal],cal[b,pedestal]
            weights=[ca[i]['mean']-ca[0]['mean']-cb[i]['mean']+cb[0]['mean'] for i in (1,2,4)]
            offset=ca[0]['mean']-cb[0]['mean'];denominator=sum(abs(w) for w in weights)
            # Full covariance of sum of signed basis estimates includes 3 shared zeros.
            variance_sum=sum(ca[i]['sd']**2/5+cb[i]['sd']**2/5 for i in (1,2,4))+9*(ca[0]['sd']**2/5+cb[0]['sd']**2/5)
            useful=denominator>=max(5.,5*math.sqrt(variance_sum))
            pedestal_clear=not pedestal or all(c[0]['mean']>=max(5.,5*c[0]['sd']) for c in (ca,cb))
            calibration_valid=all(x['passes'] for c in (ca,cb) for x in basis_quality(c))
            eligible=useful and pedestal_clear and calibration_valid
            tests=[]
            for bits in range(8):
                pa,va=reference(ca,bits);pb,vb=reference(cb,bits)
                # Paired acquisitions preserve cross-WL covariance in the observed SE.
                diffs=[next(r['raw_adc'] for r in records if r['phase']=='validation' and r['wl']==a and r['pedestal']==pedestal and r['bits']==bits and r['repeat']==rep)
                       -next(r['raw_adc'] for r in records if r['phase']=='validation' and r['wl']==b and r['pedestal']==pedestal and r['bits']==bits and r['repeat']==rep) for rep in range(5)]
                measured=stats(diffs)
                repeat=repeatability(measured['mean']-offset,math.sqrt(measured['sd']**2+ca[0]['sd']**2+cb[0]['sd']**2),
                    pa-pb-offset,math.sqrt(5*(va+vb)),zero=bits==0)
                test=error_test(measured['mean'],pa-pb,pa-pb-offset,va+vb+measured['sd']**2/5)
                tests.append({**test,'bits':bits,'mean_pass':test['passes'],'repeatability':repeat,
                    'per_repeat_errors_adc':[x-(pa-pb) for x in diffs],'passes':test['passes'] and repeat['passes']})
            pairs.append(dict(pair_id=pair_id,wl=[a,b],pedestal=pedestal,signed_basis_adc=weights,calibrated_offset_adc=offset,
                normalization_denominator_adc=denominator,normalization_noise_se_adc=math.sqrt(variance_sum),
                calibration_valid=calibration_valid,pedestal_away_from_zero=pedestal_clear,usable_differential_signal=useful,
                validation=tests,status='inconclusive' if not eligible else 'pass' if all(x['passes'] for x in tests) else 'fail',
                row_additivity_required_for_pair=False))
            for sequence_name,bits_list in plan['sequences'].items():
                for polarity in (1,-1):
                    if not eligible:
                        lif_results.append(dict(pair_id=pair_id,pedestal=pedestal,sequence=sequence_name,polarity=polarity,
                            status='inconclusive',reason='inadequate_calibration_or_differential_signal'));continue
                    scale=2./denominator
                    expected=lif([polarity*scale*sum(weights[i] for i in range(3) if bits&(1<<i)) for bits in bits_list])
                    excited=any(t['spike'] for t in expected) and any(not t['spike'] for t in expected)
                    trials=[]
                    for rep in range(5):
                        currents=[];task_ids=[]
                        for t,bits in enumerate(bits_list):
                            rr=[next(r for r in records if r['phase']=='sequence' and r['sequence']==sequence_name and r['t']==t
                                and r['repeat']==rep and r['pedestal']==pedestal and r['wl']==wl) for wl in (a,b)]
                            currents.append(polarity*scale*(rr[0]['raw_adc']-rr[1]['raw_adc']-offset));task_ids.append([r['task_id'] for r in rr])
                        measured=lif(currents)
                        error=max(abs(m[k]-e[k]) for m,e in zip(measured,expected) for k in ('before_reset','after_reset'))
                        spike_agreement=[x['spike'] for x in measured]==[x['spike'] for x in expected]
                        trials.append(dict(repeat=rep,task_ids=task_ids,measured=measured,max_membrane_error=error,
                                           spike_agreement=spike_agreement,passes=spike_agreement and error<=.05))
                    lif_results.append(dict(pair_id=pair_id,pedestal=pedestal,sequence=sequence_name,polarity=polarity,
                        calibration_only_scale=scale,identity_bn=True,expected=expected,expected_has_spike_and_silent=excited,
                        minimum_expected_threshold_distance=min(abs(t['threshold_margin']) for t in expected),trials=trials,
                        status='inconclusive' if not excited else 'pass' if all(t['passes'] for t in trials) else 'fail'))
    drift=[]
    for pair in plan['pairs']:
        for wl in pair['wl']:
            for bl in pair['data_bl']+[plan['pedestal_bl']]:
                pre,post=(stats([r['conductance_siemens'] for r in readbacks if r['wl']==wl and r['bl']==bl and r['phase']==phase]) for phase in ('pre','post'))
                fraction=(post['mean']-pre['mean'])/pre['mean'];se=math.sqrt(pre['sd']**2/3+post['sd']**2/3)/pre['mean']
                tolerance=max(.05,3*se)
                drift.append(dict(wl=wl,bl=bl,pre=pre,post=post,fractional_delta=fraction,tolerance=tolerance,passes=abs(fraction)<=tolerance))
    return dict(row_additivity=rows,signed_pairs=pairs,digital_lif=lif_results,conductance_comparison=drift,
        all_readbacks_consistent=all(d['passes'] for d in drift),overall_qualified=False,
        acquisition_complete=True,trained_bn_applied=False,analog_lif_tested=False,programming_qualified=False,
        source_reference='Independent calibration at the same source; no hardware/simulator absolute state matching',
        qualification_policy='Row, pair, digital LIF and readback consistency reported separately; no automatic CIFAR approval')
