"""Offline saved-array audit. Never constructs a network or invokes an experiment."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SESSION=ROOT/'results/mnist_stdp_retraining_20261019/440fc54b-a3bc-48f9-ae3c-43196809b96d'
OUT=ROOT/'reports/2026-10-19/mnist-stdp-training-review'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
array_sha=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main():
    m=json.loads((SESSION/'manifest.json').read_text(encoding='utf-8'))
    assert m['status']=='completed' and m['scope']=='train_only128_smoke'
    inventory={p.name for p in SESSION.iterdir() if p.is_file()}
    assert inventory==set(m['result_sha256'])|{'manifest.json'}
    for name,pin in m['result_sha256'].items():assert sha(SESSION/name)==pin,name
    for name,pin in m['source_sha256'].items():assert sha(ROOT/name)==pin,name
    entry=ROOT/'experiments/mnist_stdp_retraining_20261019/experiment.py'
    assert sha(entry)==m['entrypoint_sha256']
    events=[json.loads(line) for line in (SESSION/'events.jsonl').read_text().splitlines()]
    rows=[row for row in events if row['kind']=='stdp_image']
    assert [row['image_id'] for row in rows]==list(range(128))
    assert all(row['steps']==100 and row['native_norm_none'] is True for row in rows)
    assert [r['kind'] for r in events]==['first_image_clipping_and_learning']+['stdp_image']*128+['stdp_training_complete']
    initial=np.load(SESSION/'initial_weights.npy',allow_pickle=False)
    trace=np.load(SESSION/'first_image_postupdate_weights.npy',allow_pickle=False)
    weights=np.load(SESSION/'stdp_end_of_image_weights.npy',allow_pickle=False)
    final=np.load(SESSION/'trained_weights.npy',allow_pickle=False)
    threshold=np.load(SESSION/'trained_threshold.npy',allow_pickle=False)
    ids=np.load(SESSION/'stdp_train_ids.npy',allow_pickle=False)
    assert initial.shape==(64,1,5,5) and trace.shape==(100,64,1,5,5) and weights.shape==(128,64,1,5,5)
    assert all(a.dtype==np.float32 and np.isfinite(a).all() for a in [initial,trace,weights,final,threshold])
    assert np.array_equal(ids,np.arange(128)) and np.array_equal(trace[0],np.clip(initial,0,1))
    assert np.array_equal(trace[-1],weights[0]) and np.array_equal(weights[-1],final)
    assert ((trace>=0)&(trace<=1)).all() and ((weights>=0)&(weights<=1)).all()
    assert threshold.shape==() and threshold.item()==np.float32(.38).item()
    thresholds=rows[0]['threshold_sha256'];bias=rows[0]['bias_sha256']
    assert all(r['threshold_sha256']==thresholds and r['bias_sha256']==bias for r in rows)
    assert array_sha(threshold)==thresholds['Conv1']
    deltas=[];changed=[]
    for i,row in enumerate(rows):
        before=initial if i==0 else weights[i-1]
        delta=weights[i]-before
        assert array_sha(weights[i])==row['weight_sha256']
        assert float(weights[i].min())==row['weight_min'] and float(weights[i].max())==row['weight_max']
        count=int(np.count_nonzero(delta));l1=float(np.abs(delta).sum(dtype=np.float64))
        assert count==row['changed_coefficients'] and np.isclose(l1,row['delta_l1'],rtol=2e-6,atol=1e-6)
        changed.append(count);deltas.append(l1)
    assert array_sha(initial)==m['training']['initial_weight_sha256']
    assert array_sha(final)==m['training']['final_weight_sha256']
    import torch
    checkpoint=torch.load(SESSION/'trained_model.pt',map_location='cpu',weights_only=True)
    assert np.array_equal(checkpoint['W'].numpy(),final) and np.array_equal(checkpoint['v_thresh'].numpy(),threshold)
    assert checkpoint['trained_samples']==128 and checkpoint['final_weight_sha256']==array_sha(final)
    assert sum(r['output_spikes'] for r in rows)==m['training']['output_spikes']
    times=np.asarray([r['seconds'] for r in rows]);steady=times[1:]
    total=m['training']['training_seconds'];wall=(datetime.fromisoformat(m['finished_utc'])-datetime.fromisoformat(m['started_utc'])).total_seconds()
    clipped=trace[0];post_clip=weights-clipped
    result=dict(schema='mnist-stdp-author-saved-array-audit-v1',status='PASS',session=SESSION.relative_to(ROOT).as_posix(),
        manifest_sha256=sha(SESSION/'manifest.json'),audit_tool_sha256=sha(__file__),source_sha256=m['entrypoint_sha256'],spec_sha256=m['spec_sha256'],
        files_verified=len(m['result_sha256']),sources_verified=len(m['source_sha256']),image_ids_exact=True,images=128,timesteps=12800,
        initial=dict(min=float(initial.min()),max=float(initial.max()),per_filter_sum_min=float(initial.sum((1,2,3)).min()),
                     per_filter_sum_max=float(initial.sum((1,2,3)).max()),coefficients_above_one=int((initial>1).sum())),
        first_step=dict(pure_clipping_exact=True,l1_change=float(np.abs(initial-clipped).sum(dtype=np.float64))),
        first_window=dict(post_clipping_changed_coefficients=int(np.count_nonzero(trace[-1]-clipped)),
                          post_clipping_l1_change=float(np.abs(trace[-1]-clipped).sum(dtype=np.float64)),
                          postwindow_equals_last_native_record_exact=True),
        subsequent_images=dict(all127_change_weights=all(c>0 for c in changed[1:]),
            changed_coefficients_min=min(changed[1:]),changed_coefficients_max=max(changed[1:]),
            l1_delta_min=min(deltas[1:]),l1_delta_max=max(deltas[1:])),
        final=dict(min=float(final.min()),max=float(final.max()),mean=float(final.mean()),
            changed_vs_first_clipped_coefficients=int(np.count_nonzero(final-clipped)),
            l1_vs_first_clipped=float(np.abs(final-clipped).sum(dtype=np.float64)),
            above_first_clipped=int((final>clipped).sum()),below_first_clipped=int((final<clipped).sum()),
            saturated_zero=int((final==0).sum()),saturated_one=int((final==1).sum()),
            weight_trajectory_all_finite_in_bounds=True),
        activity=dict(input_spikes=sum(r['input_spikes'] for r in rows),output_spikes=sum(r['output_spikes'] for r in rows),
                      silent_images=sum(r['output_spikes']==0 for r in rows)),
        fixed_state=dict(threshold_hashes=thresholds,bias_hash=bias,all128_unchanged=True),
        timing=dict(training_seconds=total,session_wall_seconds=wall,first_image_seconds=float(times[0]),
            image_mean_seconds=float(times.mean()),image_median_seconds=float(np.median(times)),image_p95_seconds=float(np.percentile(times,95)),
            steady_mean_seconds=float(steady.mean()),full30000_from_total_ratio_seconds=total/128*30000,
            full30000_from_steady_mean_seconds=float(steady.mean())*30000,
            feature314_batches_pilot_sensitivity_seconds=[5.035*314,8.544*314],readout_runtime='unmeasured'),
        resources=dict(session_bytes=sum(p.stat().st_size for p in SESSION.iterdir() if p.is_file()),
            cuda_peak_allocated_bytes=max(r['cuda_peak_allocated_bytes'] for r in rows),
            cuda_peak_reserved_bytes=max(r['cuda_peak_reserved_bytes'] for r in rows),
            full_weight_trajectory_bytes=30000*1600*4,full_counts_bytes=40000*36864*4),
        accuracy=None,test_samples=0,readout_trained=False,actual_head_updates=0,actual_feature_batches=0,physical_memristor=False,
        manifest_planned_fields_not_executed=dict(head_updates=m['head_updates'],feature_batches=m['feature_batches'],counts_bytes=m['counts_bytes']),
        limitations=['Saved-array consistency audit, not independent reexecution of all native STDP updates.',
                     'Only first image has per-step weights; later evidence is per-image trajectory.',
                     'Aggregate deltas alone do not separately identify every LTD/LTP event.',
                     'Prefix128 runtime is a planning estimate, not a full30k runtime guarantee.'],
        full_proposal=dict(keep_existing_wall_budget_seconds=28800,keep_maximum_output_bytes=8*1024**3,
            keep_initial_free_disk_bytes=18*1024**3,independent_review_required=True,full_gate_opened=False))
    (OUT/'SMOKE128_AUTHOR_AUDIT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
