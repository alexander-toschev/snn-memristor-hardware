"""Inspect the saved interrupted MNIST prefix; no network, model or device calls."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SESSION = ROOT / 'results/mnist_stdp_retraining_20261019/7a536bf8-0269-4edd-b454-a8ba2c33c219'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def array_sha(a):
    return hashlib.sha256(a.tobytes(order='C')).hexdigest()


def audit():
    import numpy as np
    require(not (SESSION / 'manifest.json').exists(), 'This audit targets the interrupted prefix only')
    before = {p.name: sha(p) for p in SESSION.iterdir() if p.is_file()}
    start = json.loads((SESSION / 'start_manifest.json').read_text())
    require(start['entrypoint_sha256'] == '87eedc25cc3ca73aa429d5dff9b210c9076b8257538ce2789cbec4aa5a6016fd', 'Unreviewed entry')
    require(start['full_run_spec_sha256'] == '5e0e64cf86142de7c55ac0e505014c56e619c2c2e26f93349deac89720843ebe', 'Wrong protocol')
    for name, pin in start['source_sha256'].items():
        path = (ROOT / name).resolve()
        require(path.is_relative_to(ROOT) and sha(path) == pin, 'Source mismatch ' + name)
    rows = [json.loads(line) for line in (SESSION / 'events.jsonl').read_text().splitlines()]
    require(len(rows) == 30244, 'Unexpected event count')
    require(rows[0]['kind'] == 'first_image_clipping_and_learning', 'Missing first-step evidence')
    image_rows = rows[1:30001]
    require(all(r['kind'] == 'stdp_image' and r['image_id'] == i and r['steps'] == 100 and r['native_norm_none'] is True for i, r in enumerate(image_rows)), 'Training order/coverage')
    summary = rows[30001]
    require(summary['kind'] == 'stdp_training_complete' and summary['training_samples'] == 30000 and summary['timesteps'] == 3000000, 'Training completion')
    features = rows[30002:30237]
    require(rows[30237] == dict(kind='feature_split_complete', split='train', rows=30000, new_weights=True), 'Train feature completion')
    require(rows[30238]['kind'] == 'scaler_complete' and rows[30238]['train_rows'] == 30000 and rows[30238]['test_access'] is False, 'Scaler completion')
    heads = rows[30239:]
    require(all(r['kind'] == 'head_epoch' and r['epoch'] == i+1 and r['updates'] == 59*(i+1) and r['test_access'] is False for i, r in enumerate(heads)), 'Head prefix')
    def load(name):
        return np.load(SESSION / name, mmap_mode='r', allow_pickle=False)
    ids = load('stdp_train_ids.npy')
    require(ids.dtype == np.int64 and np.array_equal(ids, np.arange(30000)), 'Training IDs')
    require(np.array_equal(load('train_ids.npy'), ids), 'Feature IDs')
    initial = load('initial_weights.npy'); trajectory = load('stdp_end_of_image_weights.npy')
    final = load('trained_weights.npy'); trace = load('first_image_postupdate_weights.npy')
    require(initial.shape == final.shape == (64, 1, 5, 5) and trajectory.shape == (30000, 64, 1, 5, 5) and trace.shape == (100, 64, 1, 5, 5), 'Weight shapes')
    require(all(a.dtype == np.float32 for a in (initial, trajectory, final, trace)), 'Weight dtype')
    require(np.isfinite(initial).all() and (initial >= 0).all(), 'Initial values')
    require(np.array_equal(trace[0], np.clip(initial, 0, 1)) and np.array_equal(trace[-1], trajectory[0]), 'Initial clipping/trace boundary')
    require(np.array_equal(final, trajectory[-1]) and array_sha(final) == summary['final_weight_sha256'], 'Final saved W')
    previous = initial; changed_after_first = 0
    for index, (weight, row) in enumerate(zip(trajectory, image_rows)):
        require(np.isfinite(weight).all() and ((weight >= 0) & (weight <= 1)).all(), 'Weight range')
        require(array_sha(weight) == row['weight_sha256'], 'Weight row SHA')
        changed = int(np.count_nonzero(weight != previous))
        require(changed == row['changed_coefficients'], 'Weight change telemetry')
        require(np.isclose(float(np.abs(weight - previous).sum(dtype=np.float32)), row['delta_l1'], rtol=2e-6, atol=1e-6), 'Weight delta telemetry')
        require(row['threshold_sha256'] == image_rows[0]['threshold_sha256'] and row['bias_sha256'] == image_rows[0]['bias_sha256'], 'Fixed-state telemetry')
        changed_after_first += int(index > 0 and changed > 0)
        previous = weight
    require(changed_after_first == summary['images_with_change_after_initial_image'], 'Plasticity count')
    require(sum(r['output_spikes'] for r in image_rows) == summary['output_spikes'], 'Training spike count')
    counts = load('train_counts.npy')
    require(counts.shape == (30000, 36864) and counts.dtype == np.float32, 'Feature shape/dtype')
    frozen = features[0]['frozen_state']; total_spikes = 0; silent = 0
    require(frozen['weights'] == {"('Input', 'Conv1')": array_sha(final)}, 'Train features do not bind new W')
    require(frozen['thresholds']['Conv1'] == array_sha(load('trained_threshold.npy')), 'Saved threshold differs')
    for index, row in enumerate(features):
        lo = index*128; hi = min(lo+128, 30000)
        require(row['kind'] == 'new_weight_features' and (row['split'], row['start'], row['end']) == ('train', lo, hi) and row['frozen_state'] == frozen, 'Feature coverage/frozen-state')
        values = np.asarray(counts[lo:hi])
        require(np.isfinite(values).all() and ((values >= 0) & (values <= 100)).all() and np.array_equal(values, np.floor(values)), 'Invalid feature counts')
        require(array_sha(values) == row['counts_sha256'], 'Feature block SHA')
        spike_count = int(values.sum(dtype=np.int64))
        require(spike_count == row['output_spikes'], 'Feature spike telemetry')
        total_spikes += spike_count; silent += int(np.count_nonzero(values.sum(axis=1) == 0))
    for index, row in enumerate(heads):
        order = load('epoch%02d_train_order.npy' % index)
        require(order.dtype == np.int64 and np.array_equal(np.sort(order), ids) and array_sha(order) == row['order_sha256'], 'Head permutation')
    mean = load('scaler_mean.npy'); sd = load('scaler_sd.npy')
    require(mean.shape == sd.shape == (36864,) and mean.dtype == sd.dtype == np.float32 and np.isfinite(mean).all() and np.isfinite(sd).all() and (sd >= 1e-6).all(), 'Scaler values')
    require(not any(name.startswith('test_') or name in ('head_weight.npy', 'head_bias.npy', 'shuffle_rng_final.npy') for name in before), 'Unexpected final/evaluation artifacts')
    require(before == {p.name: sha(p) for p in SESSION.iterdir() if p.is_file()}, 'Prefix changed during audit')
    return dict(schema='mnist-interrupted-prefix-independent-audit-v1', status='saved_prefix_consistent',
        session=SESSION.relative_to(ROOT).as_posix(), source_sha256=start['source_sha256'], artifact_sha256=before,
        auditor_sha256=sha(Path(__file__)), final_manifest_present=False, completed_full_experiment=False,
        recorded_training=summary, training_images_verified=30000, trajectory_rows_verified=30000,
        train_feature_blocks_verified=235, train_feature_rows_verified=30000, train_spikes=total_spikes,
        train_silent_images=silent, frozen_state=frozen, completed_head_epochs=5, completed_head_updates=295,
        test_evaluation_evidence_present=False, accuracy=None,
        limitations=['No SNN/STDP forward or input-encoder replay.', 'Stored checkpoint bytes hashed, not deserialized here.',
            'Scaler shape/range verified, not recomputed here; recovery qualification must replay train-only scaler.',
            'Events verify saved state telemetry, not independent model execution.',
            'Start failed/network_executed=false are initial fail-safe markers, not a completion result.'],
        hardware_access=False, model_executed=False)


if __name__ == '__main__':
    result = audit()
    path = ROOT / 'reports/2026-10-19/mnist-stdp-training-review/INTERRUPTED_PREFIX_INDEPENDENT_AUDIT.json'
    with path.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps({k: result[k] for k in ('status', 'training_images_verified', 'train_feature_blocks_verified', 'train_spikes', 'train_silent_images', 'completed_head_epochs')}))
