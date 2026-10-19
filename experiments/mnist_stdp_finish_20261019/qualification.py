"""Read-only qualification of an interrupted session; no neural execution."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'results/mnist_stdp_retraining_20261019/7a536bf8-0269-4edd-b454-a8ba2c33c219'
SNAPSHOT = Path(__file__).with_name('INPUT_SNAPSHOT.json')
SNAPSHOT_SHA = '321bceaf0e7fc390883c70ec43206526b066de15ce5cce6cf41bfb2b3e7f5fee'
TRAINING_SHA = '7f2a35d9fe5b832686d73b57d940acbc6fae99a3406de41258377d4d2fd9ee3f'
RECOVERY_SHA = '06b370e89135981ce756442b715e04ed280498de150cc42f43dfd39b4d4ce394'
FULL_SHA = '87eedc25cc3ca73aa429d5dff9b210c9076b8257538ce2789cbec4aa5a6016fd'
REVIEW = ROOT/'reports/2026-10-19/mnist-stdp-training-review/INTERRUPTED_PREFIX_INDEPENDENT_AUDIT.json'
REVIEW_SHA = '36b20afb4043ba7f9f26fb94f8eaef960c9120ef95d2de352cc52ada6c646641'


def require(value, message):
    if not value: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''): h.update(block)
    return h.hexdigest()


def digest(value): return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
def array_sha(value): return hashlib.sha256(value.tobytes(order='C')).hexdigest()


def verify_snapshot():
    require(sha(SNAPSHOT) == SNAPSHOT_SHA, 'Interrupted input snapshot changed')
    snapshot = json.loads(SNAPSHOT.read_text())
    require(sha(REVIEW)==REVIEW_SHA, 'Independent partial qualification changed')
    review=json.loads(REVIEW.read_text())
    require(review['status']=='saved_prefix_consistent' and review['session']==snapshot['session']
            and review['artifact_sha256']=={name:info['sha256'] for name,info in snapshot['files'].items()}, 'Independent review/input snapshot mismatch')
    require(snapshot['session'] == SOURCE.relative_to(ROOT).as_posix(), 'Wrong source session')
    require({p.name for p in SOURCE.iterdir()} == set(snapshot['files']), 'Incomplete-session inventory changed')
    require('manifest.json' not in snapshot['files'], 'Do not reclassify a finalized source session')
    for name, info in snapshot['files'].items():
        path = (SOURCE / name).resolve()
        require(path.parent == SOURCE and path.is_file(), 'Invalid source filename')
        require(path.stat().st_size == info['bytes'] and sha(path) == info['sha256'], 'Interrupted artifact changed: ' + name)
    return snapshot


def parse_prefix(rows, size=30000, epochs=5):
    items = iter(rows)
    def take(kind):
        row = next(items, None)
        require(row is not None and row['kind'] == kind, 'Interrupted sequence expected ' + kind)
        return row
    clip = take('first_image_clipping_and_learning'); images = []
    for i in range(size):
        row = take('stdp_image'); require(row['image_id'] == i and row['steps'] == 100 and row['native_norm_none'] is True, 'STDP sequence')
        images.append(row)
    summary = take('stdp_training_complete'); features = []
    for start in range(0, size, 128):
        row = take('new_weight_features')
        require((row['split'], row['start'], row['end']) == ('train', start, min(size, start + 128)), 'Train feature coverage')
        features.append(row)
    split = take('feature_split_complete')
    require(split == dict(kind='feature_split_complete', split='train', rows=size, new_weights=True), 'Incomplete train split')
    scaler = take('scaler_complete'); require(scaler['train_rows'] == size and scaler['test_access'] is False, 'Scaler leakage')
    heads = []
    for i in range(epochs):
        row = take('head_epoch')
        require(row['epoch'] == i + 1 and row['updates'] == (i + 1) * ((size + 511) // 512)
                and row['train_examples'] == size and row['test_access'] is False, 'Head prefix differs')
        heads.append(row)
    require(next(items, None) is None, 'Unexpected later/test events in interrupted input')
    return dict(clipping=clip, images=images, summary=summary, features=features, scaler=scaler, heads=heads)


def qualify(recovery):
    import gzip
    import struct
    import numpy as np
    import torch
    snapshot = verify_snapshot()
    start = json.loads((SOURCE / 'start_manifest.json').read_text())
    require(start['entrypoint_sha256'] == FULL_SHA and start['acquisition_helper_sha256'] == TRAINING_SHA, 'Wrong original execution route')
    require(start['full_run_spec_sha256'] == '5e0e64cf86142de7c55ac0e505014c56e619c2c2e26f93349deac89720843ebe'
            and digest(start['full_run_spec']) == start['full_run_spec_sha256'], 'Wrong full specification')
    require(start['spec_sha256'] == 'db6a90ce72a25a37144971f405c6a23097de83c9583dd0c9e713848c50e205bd'
            and digest(start['spec']) == start['spec_sha256'], 'Wrong training specification')
    for name, pin in start['source_sha256'].items():
        path = (ROOT / name).resolve(); require(path.is_relative_to(ROOT) and sha(path) == pin, 'Original source bytes differ')
    with (SOURCE / 'events.jsonl').open() as stream: prefix = parse_prefix(json.loads(line) for line in stream)
    load = lambda name: np.load(SOURCE / name, mmap_mode='r', allow_pickle=False)
    initial = load('initial_weights.npy'); trace = load('first_image_postupdate_weights.npy')
    trajectory = load('stdp_end_of_image_weights.npy'); final = load('trained_weights.npy'); threshold = load('trained_threshold.npy')
    require(initial.shape == final.shape == (64, 1, 5, 5) and trace.shape == (100, 64, 1, 5, 5)
            and trajectory.shape == (30000, 64, 1, 5, 5), 'Training array shapes')
    for a in (initial, trace, trajectory, final, threshold): require(a.dtype == np.float32 and np.isfinite(a).all(), 'Invalid trained values')
    require((initial >= 0).all() and np.allclose(initial.sum((1, 2, 3)), 78.4, rtol=1e-6, atol=1e-5), 'Initial normalization')
    require(np.array_equal(trace[0], np.clip(initial, 0, 1)) and np.array_equal(trace[-1], trajectory[0])
            and np.array_equal(trajectory[-1], final), 'Weight trajectory/clipping links')
    require(((trajectory >= 0) & (trajectory <= 1)).all() and ((trace >= 0) & (trace <= 1)).all(), 'Trained native bounds')
    require(threshold.shape == () and threshold.item() == np.float32(.38).item(), 'Trained threshold')
    require(np.array_equal(load('stdp_train_ids.npy'), np.arange(30000)), 'Training IDs')
    thresholds = dict(Conv1=array_sha(threshold), Inhib1=array_sha(threshold)); bias = array_sha(np.zeros(64, np.float32))
    changed = 0
    for i, row in enumerate(prefix['images']):
        w = trajectory[i]; d = w - (initial if i == 0 else trajectory[i - 1]); n = int(np.count_nonzero(d))
        changed += int(i > 0 and n > 0)
        require(row['weight_sha256'] == array_sha(w) and row['threshold_sha256'] == thresholds and row['bias_sha256'] == bias, 'Per-image learned state')
        require(row['changed_coefficients'] == n and np.isclose(row['delta_l1'], np.abs(d).sum(dtype=np.float64), rtol=2e-6, atol=1e-6), 'Weight delta telemetry')
        require(row['weight_min'] == float(w.min()) and row['weight_max'] == float(w.max()), 'Weight min/max')
    summary = prefix['summary']
    clipping=prefix['clipping']; actual_changed=int(np.count_nonzero(trace[-1]-trace[0]))
    require(summary['clipping']=={k:v for k,v in clipping.items() if k!='kind'}
            and clipping['first_step_clipping_verified'] is True
            and clipping['initial_out_of_bounds']==int(((initial<0)|(initial>1)).sum())
            and clipping['post_first_step_changed_coefficients']==actual_changed, 'First-clipping summary differs')
    require(np.isclose(clipping['clipping_l1'],np.abs(trace[0]-initial).sum(dtype=np.float64),rtol=2e-6,atol=1e-6)
            and np.isclose(clipping['post_first_step_l1'],np.abs(trace[-1]-trace[0]).sum(dtype=np.float64),rtol=2e-6,atol=1e-6), 'First-clipping delta differs')
    require(summary['training_samples'] == 30000 and summary['timesteps'] == 3000000
            and summary['images_with_change_after_initial_image'] == changed and summary['final_weight_sha256'] == array_sha(final)
            and summary['initial_weight_sha256'] == array_sha(initial), 'Completed training summary')
    require(summary['output_spikes'] == sum(r['output_spikes'] for r in prefix['images']), 'Training spike total')
    checkpoint = torch.load(SOURCE / 'trained_model.pt', map_location='cpu', weights_only=True)
    require(np.array_equal(checkpoint['W'].numpy(), final) and np.array_equal(checkpoint['v_thresh'].numpy(), threshold)
            and checkpoint['trained_samples'] == 30000 and checkpoint['cfg'] == start['effective_cfg']
            and checkpoint['source_protocol_sha256'] == start['spec_sha256'] and checkpoint['new_reconstruction'] is True
            and checkpoint['initial_weight_sha256']==array_sha(initial) and checkpoint['final_weight_sha256']==array_sha(final), 'Checkpoint provenance')
    frozen = dict(weights={"('Input', 'Conv1')": array_sha(final)}, biases={"('Input', 'Conv1')": bias}, thresholds=thresholds)
    counts = load('train_counts.npy'); ids = load('train_ids.npy'); labels = load('train_labels.npy')
    require(counts.shape == (30000, 36864) and counts.dtype == np.float32 and ids.dtype == labels.dtype == np.int64
            and np.array_equal(ids, np.arange(30000)), 'Train feature shape/IDs')
    data = ROOT / 'experiments/mnist_baseline_20261019/data/MNIST/raw'
    with gzip.open(data / 'train-labels-idx1-ubyte.gz', 'rb') as stream:
        require(struct.unpack('>II', stream.read(8)) == (2049, 60000), 'IDX labels header')
        require(np.array_equal(labels, np.frombuffer(stream.read(30000), np.uint8).astype(np.int64)), 'Train labels')
    for row in prefix['features']:
        block = counts[row['start']:row['end']]
        require(np.isfinite(block).all() and ((block >= 0) & (block <= 100) & (block == np.round(block))).all(), 'Invalid train counts')
        require(array_sha(block) == row['counts_sha256'] and row['frozen_state'] == frozen, 'Feature new-weight binding')
    mean, sd = recovery.fit_scaler(counts)
    require(np.array_equal(mean.numpy(), load('scaler_mean.npy')) and np.array_equal(sd.numpy(), load('scaler_sd.npy')), 'Train-only scaler replay')
    for i, row in enumerate(prefix['heads']):
        order = load('epoch%02d_train_order.npy' % i)
        require(order.dtype == np.int64 and np.array_equal(np.sort(order), np.arange(30000)) and array_sha(order) == row['order_sha256'], 'Original head permutation')
    report = dict(status='qualified_interrupted_prefix', snapshot_sha256=SNAPSHOT_SHA, independent_review_sha256=REVIEW_SHA,
        inherited=dict(training_images=30000, training_steps=3000000, train_feature_rows=30000, train_feature_batches=235),
        discarded_incomplete_readout=dict(completed_epochs=5, updates=295, optimizer_checkpoint_available=False),
        trained_weight_sha256=array_sha(final), frozen_state=frozen, training_seconds=summary['training_seconds'],
        train_feature_seconds=sum(r['seconds'] for r in prefix['features']), actual_test_examples_before_recovery=0,
        source_session_is_incomplete=True, source_start_status_is_not_final_outcome=True,
        limitations=['Saved-array/event consistency, not independent SNN/STDP forward replay.',
                    'Source has no final manifest; this independent input snapshot qualifies only its completed prefix.'])
    return report, prefix, start, checkpoint
