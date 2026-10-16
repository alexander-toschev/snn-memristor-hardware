"""Compact offline fidelity/LIF diagnostics; no backend, raw copies or refitting."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics

ANALYZER_SHA = '0ced0d8d3b3f5b126374571659c38b5399be4af8cdfc0ebe39a99a0c9331d8c0'
MAX_BYTES = 250000


def require(ok, message):
    if not ok: raise ValueError(message)


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def fidelity(stage, slots):
    reasons=[]
    if stage['raw_target_nrmse']>.20: reasons.append('mean_nrmse_above_0.20')
    if stage['uncertainty_nrms']>.20: reasons.append('uncertainty_nrms_above_0.20')
    if stage['resolved_sign_error_slots']: reasons.append('resolved_wrong_sign')
    if stage['measured_sign_unresolved_significant_slots']: reasons.append('sign_unresolved_for_significant_target')
    require(bool(stage['physical_fidelity_pass']) == (not reasons), 'Recorded fidelity gate mismatch')
    def taps(key): return [slots[i]['tap_index'] for i in stage[key]]
    return dict(pass_recorded=stage['physical_fidelity_pass'],reasons=reasons,
        mean_nrmse=stage['raw_target_nrmse'],uncertainty_nrms=stage['uncertainty_nrms'],
        wrong_sign_taps=taps('resolved_sign_error_slots'),
        significant_sign_unresolved_taps=taps('measured_sign_unresolved_significant_slots'),
        small_target_taps=taps('unresolved_quantization_slots'),
        one_adc_weight_resolution=stage['one_adc_weight_resolution'],
        pairs=[dict(tap=s['tap_index'],positive_cell=s['positive_cell'],negative_cell=s['negative_cell'],
            target=s['trained_weight'],measured=stage['raw_measured_weight'][i],error=stage['raw_target_errors'][i],
            observed_sd=stage['observed_weight_sd'][i],engineering_uncertainty=stage['engineering_weight_uncertainty'][i])
            for i,s in enumerate(slots)])


def point(row):
    return dict(conv=row['conv_sum'],bn=row['bntt_current'],v_before=row['lif_before_reset'],
        threshold_margin=row['lif_before_reset']-1.,v_after=row['lif_after_reset'],spike=row['lif_spike'])


def mismatch_details(trajectories, fixture):
    mismatches=[];groups=defaultdict(list);seen=set();counts=Counter()
    for row in trajectories:
        f,c,r=row['filter_id'],row['case'],row['repeat'];key=(f,c,r)
        require(key not in seen,'Duplicate trajectory');seen.add(key)
        require(len(row['actual'])==len(row['effective_weight_reference'])==12,'Incomplete trajectory')
        golden=fixture['cases'][c]['reference_t12x64']
        for t,actual in enumerate(row['actual']):
            trained={k:golden[k][t][f] for k in ('conv_sum','bntt_current','lif_before_reset','lif_after_reset')}
            trained['lif_spike']=golden['lif_spikes'][t][f]
            effective=row['effective_weight_reference'][t]
            refs=[name for name,ref in [('trained',trained),('effective',effective)] if actual['lif_spike']!=ref['lif_spike']]
            groups[(f,c,t)].append(dict(repeat=r,**point(actual)))
            if refs:
                counts.update(refs)
                mismatches.append(dict(filter=f,case=c,original_test_index=row['original_test_index'],repeat=r,t=t,
                    mismatched_references=refs,actual=point(actual),trained=point(trained),effective=point(effective),
                    previous_v_after=dict(actual=row['actual'][t-1]['lif_after_reset'] if t else 0.,
                        trained=golden['lif_after_reset'][t-1][f] if t else 0.,
                        effective=row['effective_weight_reference'][t-1]['lif_after_reset'] if t else 0.)))
    variation=[]
    for (f,c,t),rows in sorted(groups.items()):
        require(sorted(x['repeat'] for x in rows)==list(range(5)),'Missing/duplicate repeats')
        if len({x['spike'] for x in rows})>1:
            variation.append(dict(filter=f,case=c,t=t,conv_sample_sd=statistics.stdev(x['conv'] for x in rows),
                v_before_sample_sd=statistics.stdev(x['v_before'] for x in rows),repeats=sorted(rows,key=lambda x:x['repeat'])))
    return dict(mismatch_counts={name:counts[name] for name in ('trained','effective')},
        mismatch_rows=mismatches,repeat_spike_variation_groups=variation,
        repeat_spike_variation_group_count=len(variation),total_timepoints=12*len(trajectories))


def generate(repo, session, audit, output):
    repo=Path(repo).resolve();analyzer=repo/'analyze_fresh_stability.py'
    require(digest(analyzer)==ANALYZER_SHA,'Pinned analyzer changed')
    import analyze_fresh_stability as analysis
    require(Path(analysis.__file__).resolve()==analyzer,'Unexpected analyzer location')
    destination=analysis.local(repo,output)
    require(not destination.exists(),'Output must be a new file')
    report,trajectories=analysis.analyze(repo,session,audit) # verifies complete inventory, fixture and audit bindings
    require(report['complete'],'This diagnostic requires a complete audited session')
    directory=analysis.local(repo,session);manifest=analysis.load(directory/'manifest.json')
    fixture=analysis.load(repo/'weights/prepared_cifar100_conv1.json')
    tiles=[];reason_counts=Counter()
    for t in report['tiles']:
        f,tid=t['filter'],t['tile'];raw=analysis.load(directory/'tiles'/f'f{f}_t{tid}'/'tile.json')
        stages={name:fidelity(raw[key],raw['effective_route']) for name,key in
                [('post_all','physical_fidelity'),('post_cifar','final_physical_fidelity')]}
        passed=all(v['pass_recorded'] for v in stages.values())
        require(passed==raw['physical_weight_quality_pass'],'Tile quality combination mismatch')
        reason_counts.update(set(reason for v in stages.values() for reason in v['reasons']))
        tiles.append(dict(filter=f,tile=tid,quality_pass=passed,stages=stages))
    traces=mismatch_details(trajectories,fixture)
    for name in ('trained','effective'):
        require(traces['mismatch_counts'][name]==report['trace_metrics'][name]['spike_differences'],'Mismatch count disagreement')
    result=dict(schema='fresh-fidelity-diagnostics-v1',provenance=dict(generator_sha256=digest(Path(__file__)),
        analyzer_sha256=ANALYZER_SHA,session=session,session_id=report['session_id'],source=report['source'],
        source_commit=report['source_commit'],manifest_sha256=report['manifest_sha256'],audit=audit,
        audit_sha256=report['audit_sha256'],auditor_sha256=report['auditor_sha256'],audit_checks=report['audit_checks'],
        fixture_sha256=report['fixture_sha256'],summary_sha256=manifest['result_sha256']['summary.json'],
        inventory_verified=True,raw_replay_performed=False),
        quality=dict(passed=sum(t['quality_pass'] for t in tiles),failed=sum(not t['quality_pass'] for t in tiles),
            tiles=len(tiles),failure_reason_tile_counts=dict(reason_counts)),tiles=tiles,**traces,
        limits=['Pair errors do not establish a causal defect in either cell.',
            'Effective reference uses C times post-all calibration H / fixed gain; it is not the immediate own-pair reference.',
            'Signed threshold margin = V_before_reset - 1; spike iff margin >= 0; negative potentials retained.',
            'BN/LIF is digital after all 75 taps. This is not full-classifier accuracy.',
            'Only spike-mismatch rows and repeat spike-disagreement groups are exported; other numerical differences are not absent.',
            'Raw, full trajectories, fixture inputs and private source are not copied. No fitting or threshold changes.'])
    require(digest(analyzer)==ANALYZER_SHA and digest(directory/'manifest.json')==report['manifest_sha256'],'Source changed during analysis')
    data=(json.dumps(result,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode('utf-8')
    require(len(data)<=MAX_BYTES,'Diagnostics exceeds 250000-byte bound; no silent truncation')
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('xb') as stream: stream.write(data)
    return dict(output=output,bytes=len(data),sha256=digest(destination),quality=result['quality'],mismatch_counts=traces['mismatch_counts'],
        repeat_spike_variation_groups=len(traces['repeat_spike_variation_groups']))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',default='.')
    p.add_argument('--session',required=True);p.add_argument('--audit',required=True);p.add_argument('--output',required=True)
    a=p.parse_args(argv);print(json.dumps(generate(a.repo,a.session,a.audit,a.output),indent=2))


if __name__=='__main__':main()
