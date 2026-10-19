"""CPU-only audit of completed full MNIST saved evidence; never runs/trains an SNN."""
import argparse
from datetime import datetime
import gzip
import hashlib
import json
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / 'reports/2026-10-19/mnist-stdp-training-review'
WRAPPER_SHA = '87eedc25cc3ca73aa429d5dff9b210c9076b8257538ce2789cbec4aa5a6016fd'
SPEC_SHA = '5e0e64cf86142de7c55ac0e505014c56e619c2c2e26f93349deac89720843ebe'
FREEZE_PINS = {
    'reports/2026-10-19/mnist-stdp-training-review/FULL_SOURCE_FREEZE.json': '3ac01494f7ecbee74c70daa4e81f6a7b3389eb61755b217634ef4ceb671a6b79',
    'reports/2026-10-19/mnist-stdp-training-review/SOURCE_FREEZE.json': '7f7f67969055d2c605b2deb5f87a50e798bc1770115725e04283f6f3db8d7a2a',
    'reports/2026-10-19/mnist-feature-pilot-review/SOURCE_FREEZE.json': '5ac1ee6560ef8c29a5a4ac42e24dc549b0533a07f6c57c5f97da9f6d1b28b16a'}
DATA = ROOT / 'experiments/mnist_baseline_20261019/data/MNIST/raw'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            value.update(block)
    return value.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def array_sha(value):
    return hashlib.sha256(value.tobytes(order='C')).hexdigest()


def inside(base, name):
    path = (base / name).resolve()
    require(not Path(name).is_absolute() and path.is_relative_to(base.resolve()), 'Path escapes repository/session')
    return path


def schedule(rows, training=30000, train=30000, test=10000, epochs=25):
    """Strict event order/coverage, separately from any preflight/planned counters."""
    iterator = iter(rows)
    def take(kind):
        row = next(iterator, None)
        require(row is not None and row.get('kind') == kind, 'Expected event ' + kind)
        return row
    first = take('first_image_clipping_and_learning')
    images = []
    for i in range(training):
        row = take('stdp_image')
        require(row['image_id'] == i and row['steps'] == 100 and row['native_norm_none'] is True, 'Training coverage/order')
        images.append(row)
    summary = take('stdp_training_complete')
    def features(split, size):
        batches = []
        for start in range(0, size, 128):
            row = take('new_weight_features')
            require((row['split'], row['start'], row['end']) == (split, start, min(start + 128, size)), 'Feature gap/overlap/split')
            batches.append(row)
        end = take('feature_split_complete')
        require(end['split'] == split and end['rows'] == size and end['new_weights'] is True, 'Feature completion')
        return batches
    train_rows = features('train', train)
    scaler = take('scaler_complete')
    require(scaler['train_rows'] == train and scaler['test_access'] is False, 'Scaler trained on wrong split')
    heads = []
    for epoch in range(1, epochs + 1):
        row = take('head_epoch')
        require(row['epoch'] == epoch and row['updates'] == epoch * ((train + 511) // 512)
                and row['train_examples'] == train and row['test_access'] is False, 'Readout coverage/order')
        heads.append(row)
    frozen = take('head_frozen_before_test')
    test_rows = features('test', test)
    evaluation = take('test_complete')
    require(next(iterator, None) is None, 'Extra events/test evaluation')
    actual = dict(training_images=training, feature_batches=dict(train=len(train_rows), test=len(test_rows)),
                  feature_rows=dict(train=train, test=test), head_updates_at_last_completed_epoch=heads[-1]['updates'],
                  completed_head_epochs=epochs, test_evaluations=1)
    return dict(first=first, images=images, summary=summary, train=train_rows, test=test_rows,
                scaler=scaler, heads=heads, frozen=frozen, evaluation=evaluation, actual=actual)


def predictions(logits, predicted, labels, confusion):
    import numpy as np
    require(logits.shape == (len(labels), 10) and logits.dtype == np.float32 and np.isfinite(logits).all(), 'Invalid logits')
    require(predicted.dtype == np.int64 and np.array_equal(predicted, logits.argmax(1)), 'Predictions differ from saved logits')
    require(labels.dtype == np.int64 and ((labels >= 0) & (labels < 10)).all(), 'Invalid labels')
    expected = np.zeros((10, 10), dtype=np.int64)
    np.add.at(expected, (labels, predicted), 1)
    require(confusion.dtype == np.int64 and np.array_equal(confusion, expected), 'Confusion matrix mismatch')
    correct = int((predicted == labels).sum())
    return dict(correct=correct, total=len(labels), accuracy=correct / len(labels))


def linear_replay(counts, mean, sd, weight, bias, saved, batch=128):
    """CPU float32 preprocessing then float64 dot; fixed predeclared tolerances."""
    import numpy as np
    import torch
    require(counts.shape[1] == len(mean) == len(sd) == weight.shape[1] and saved.shape == (len(counts), len(bias)), 'Linear replay shapes')
    # Absolute and relative tolerances are fixed before the actual result is known.
    atol, rtol = 1e-3, 1e-4
    w = torch.from_numpy(np.array(weight, copy=True)).double()
    b = torch.from_numpy(np.array(bias, copy=True)).double()
    mu = torch.from_numpy(np.array(mean, copy=True)); scale = torch.from_numpy(np.array(sd, copy=True))
    maximum = 0.; violations = 0; differences = []; min_margin = float('inf'); margin_uncertain = 0
    for start in range(0, len(counts), batch):
        raw = torch.from_numpy(np.array(counts[start:start + batch], dtype=np.float32, copy=True))
        x = (torch.log1p(raw) - mu) / scale
        reference = (x.double() @ w.T + b).numpy()
        observed = np.asarray(saved[start:start + batch], dtype=np.float64)
        require(np.isfinite(reference).all() and np.isfinite(observed).all(), 'Nonfinite linear replay')
        error = np.abs(reference - observed); tolerance = atol + rtol * np.abs(reference)
        maximum = max(maximum, float(error.max())); violations += int((error > tolerance).sum())
        ref_pred = reference.argmax(1); actual_pred = observed.argmax(1)
        differences.extend((start + np.flatnonzero(ref_pred != actual_pred)).tolist())
        sorted_scores = np.sort(reference, axis=1); margin = sorted_scores[:, -1] - sorted_scores[:, -2]
        min_margin = min(min_margin, float(margin.min()))
        margin_uncertain += int((margin <= 2 * tolerance.max(1)).sum())
    return dict(method='CPU float32 log1p/normalize then float64 dot; no network or optimizer',
                atol=atol, rtol=rtol, maximum_absolute_logit_error=maximum, logit_tolerance_violations=violations,
                argmax_disagreements=len(differences), disagreement_indices=differences,
                minimum_reference_top2_margin=min_margin, examples_with_margin_within_2tolerance=margin_uncertain,
                passed=violations == 0 and not differences)


def provenance(session, manifest, start):
    for name, pin in FREEZE_PINS.items():
        require(sha(ROOT / name) == pin, 'Reviewed source freeze changed')
    require(manifest['status'] == 'completed' and manifest['scope'] == 'full_stdp30000_then_new_weight_readout', 'Full completed session required')
    require(manifest['entrypoint_sha256'] == WRAPPER_SHA and manifest['full_run_spec_sha256'] == SPEC_SHA, 'Wrong registered full route')
    for key, value in start.items():
        if key not in ('status', 'network_executed'):
            require(manifest[key] == value, 'Start/final provenance differs: ' + key)
    require(start['network_executed'] is False and manifest['network_executed'] is True, 'Execution state transition')
    require(manifest['physical_memristor'] is False and manifest['hardware_accuracy'] is False, 'Model/hardware attribution')
    require(digest(manifest['full_run_spec']) == SPEC_SHA and digest(manifest['spec']) == manifest['spec_sha256'], 'Protocol digest')
    freeze = json.loads((REVIEW / 'FULL_SOURCE_FREEZE.json').read_text())
    require(freeze['full_registered'] is True and freeze['full_spec'] == manifest['full_run_spec'], 'Full source freeze mismatch')
    require(manifest['qualification'] == dict(
        smoke_manifest_sha256=freeze['qualifying_smoke_manifest_sha256'], independently_reviewed=True,
        independent_review_path='reports/2026-10-19/mnist-stdp-training-review/SMOKE128_INDEPENDENT_AUDIT.json',
        independent_review_sha256=manifest['full_run_spec']['qualification_review_sha256']), 'Qualification references')
    for name, pin in freeze['source_sha256'].items():
        require(sha(inside(ROOT, name)) == pin, 'Reviewed source changed: ' + name)
    old = json.loads((REVIEW / 'SOURCE_FREEZE.json').read_text())
    require(manifest['spec_sha256'] == old['spec_sha256'] == manifest['full_run_spec']['training_spec_sha256']
            and manifest['acquisition_helper_sha256'] == old['entrypoint_sha256'], 'Training spec/helper chain')
    expected = dict(old['source_sha256'])
    expected.update({
        'experiments/mnist_baseline_20261019/data/MNIST/raw/t10k-images-idx3-ubyte.gz': '8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6',
        'experiments/mnist_baseline_20261019/data/MNIST/raw/t10k-labels-idx1-ubyte.gz': 'f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6',
        'experiments/mnist_stdp_retraining_20261019/experiment.py': manifest['acquisition_helper_sha256']})
    require(manifest['source_sha256'] == expected, 'Exact source inventory differs')
    for name, pin in expected.items():
        require(sha(inside(ROOT, name)) == pin, 'Source changed: ' + name)
    require(sha(ROOT / 'experiments/mnist_stdp_retraining_20261019/run_full.py') == WRAPPER_SHA, 'Wrapper bytes')
    actual = {p.name for p in session.iterdir()}
    require(actual == set(manifest['result_sha256']) | {'manifest.json'}, 'Unexpected/missing result inventory')
    for name, pin in manifest['result_sha256'].items():
        require(sha(inside(session, name)) == pin, 'Result SHA mismatch: ' + name)
    env = manifest['environment']; pilot = ROOT / 'experiments/mnist_feature_pilot_20261019'
    envfreeze = json.loads((ROOT / 'reports/2026-10-19/mnist-feature-pilot-review/SOURCE_FREEZE.json').read_text())
    require(env['full_installed_versions'] == envfreeze['dependency_versions'], 'Acquisition environment versions')
    for name, pin in env['dependency_lock_sha256'].items():
        require(sha(inside(pilot, name)) == pin, 'Dependency lock changed')
    require(sha(pilot / 'BINDSNET_SOURCE.json') == env['bindsnet_source_manifest_sha256'], 'Native source manifest')
    binds = json.loads((pilot / 'BINDSNET_SOURCE.json').read_text())
    for name, pin in binds['source_files'].items():
        require(sha(inside(ROOT / '.venv-mnist-pilot/Lib/site-packages', name)) == pin, 'Native dependency source changed')


def audit(session):
    import numpy as np
    import torch  # CPU saved-array arithmetic and restricted checkpoint decoding only.
    require(torch.__version__ == '2.4.1+cu124' and np.__version__ == '2.0.2', 'Use isolated pinned MNIST Python environment')
    torch.set_num_threads(2)
    session = inside(ROOT, session)
    require(session.parent == ROOT / 'results/mnist_stdp_retraining_20261019', 'Wrong result namespace')
    m = json.loads((session / 'manifest.json').read_text())
    start = json.loads((session / 'start_manifest.json').read_text())
    provenance(session, m, start)
    with (session / 'events.jsonl').open() as stream:
        ev = schedule(json.loads(line) for line in stream)
    require(m['confirmed_execution'] == ev['actual'], 'Actual coverage counters differ')
    require(m['planned'] == dict(head_updates=1475, feature_batches=dict(train=235, test=79), counts_bytes=5898240000), 'Planned counters differ')
    load = lambda name: np.load(session / name, allow_pickle=False, mmap_mode='r')
    initial = load('initial_weights.npy'); trace = load('first_image_postupdate_weights.npy')
    trajectory = load('stdp_end_of_image_weights.npy'); final = load('trained_weights.npy'); threshold = load('trained_threshold.npy')
    require(initial.shape == final.shape == (64, 1, 5, 5) and trace.shape == (100, 64, 1, 5, 5)
            and trajectory.shape == (30000, 64, 1, 5, 5), 'Weight shapes')
    for value in (initial, trace, trajectory, final, threshold):
        require(value.dtype == np.float32 and np.isfinite(value).all(), 'Nonfinite/wrong dtype weights')
    require(np.array_equal(load('stdp_train_ids.npy'), np.arange(30000)), 'Training IDs')
    clipped = np.clip(initial, 0, 1)
    require((initial >= 0).all(), 'Negative initial weights')
    require(np.allclose(initial.sum((1, 2, 3)), 78.4, rtol=1e-6, atol=1e-5), 'Initial normalization')
    require(np.array_equal(trace[0], clipped) and np.array_equal(trace[-1], trajectory[0])
            and np.array_equal(trajectory[-1], final), 'Clipping/trajectory links')
    require(((trajectory >= 0) & (trajectory <= 1)).all() and ((trace >= 0) & (trace <= 1)).all(), 'Native bounds')
    require(threshold.shape == () and threshold.item() == np.float32(.38).item(), 'Threshold')
    first = ev['first']
    require(first['first_step_clipping_verified'] is True and first['initial_out_of_bounds'] == int(((initial < 0) | (initial > 1)).sum())
            and first['post_first_step_changed_coefficients'] == int(np.count_nonzero(trace[-1] - clipped)), 'First clipping event counts')
    require(np.isclose(first['clipping_l1'], np.abs(trace[0] - initial).sum(dtype=np.float64), rtol=2e-6, atol=1e-6)
            and np.isclose(first['post_first_step_l1'], np.abs(trace[-1] - clipped).sum(dtype=np.float64), rtol=2e-6, atol=1e-6), 'First clipping event deltas')
    require(first['monitor_semantics'] == 'native post-update[t], before post-window hook/normalization', 'Weight monitor semantics')
    th = dict(Conv1=array_sha(threshold), Inhib1=array_sha(threshold)); bias = array_sha(np.zeros(64, dtype=np.float32))
    changed = 0
    for i, row in enumerate(ev['images']):
        value = trajectory[i]; delta = value - (initial if i == 0 else trajectory[i - 1])
        count = int(np.count_nonzero(delta)); changed += int(i > 0 and count > 0)
        require(array_sha(value) == row['weight_sha256'] and row['threshold_sha256'] == th and row['bias_sha256'] == bias, 'Per-image state evidence')
        require(count == row['changed_coefficients'] and np.isclose(np.abs(delta).sum(dtype=np.float64), row['delta_l1'], rtol=2e-6, atol=1e-6), 'Weight delta telemetry')
        require(float(value.min()) == row['weight_min'] and float(value.max()) == row['weight_max'], 'Weight min/max telemetry')
    summary = m['training']
    require(summary['clipping'] == {k: v for k, v in first.items() if k != 'kind'}
            and summary['learning_beyond_initial_clipping_observed'] == bool(changed or np.count_nonzero(trace[-1] - clipped)), 'Clipping/learning summary')
    require(ev['summary'] == dict(kind='stdp_training_complete', **summary), 'Training summary event')
    require(summary['training_samples'] == 30000 and summary['timesteps'] == 3000000
            and summary['images_with_change_after_initial_image'] == changed, 'Training summary counts')
    require(summary['initial_weight_sha256'] == array_sha(initial) and summary['final_weight_sha256'] == array_sha(final), 'Summary weight hashes')
    require(summary['output_spikes'] == sum(r['output_spikes'] for r in ev['images']), 'Spike total')
    checkpoint = torch.load(session / 'trained_model.pt', map_location='cpu', weights_only=True)
    require(np.array_equal(checkpoint['W'].numpy(), final) and np.array_equal(checkpoint['v_thresh'].numpy(), threshold), 'Checkpoint arrays')
    require(checkpoint['trained_samples'] == 30000 and checkpoint['cfg'] == m['effective_cfg']
            and checkpoint['source_protocol_sha256'] == m['spec_sha256'] and checkpoint['new_reconstruction'] is True, 'Checkpoint provenance')
    require(checkpoint['initial_weight_sha256'] == array_sha(initial) and checkpoint['final_weight_sha256'] == array_sha(final), 'Checkpoint weight provenance')
    frozen = dict(weights={"('Input', 'Conv1')": array_sha(final)}, biases={"('Input', 'Conv1')": bias}, thresholds=th)
    counts = {}; activity = {}
    for split, size, prefix, source_count in [('train', 30000, 'train', 60000), ('test', 10000, 't10k', 10000)]:
        ids = load(split + '_ids.npy'); labels = load(split + '_labels.npy'); values = load(split + '_counts.npy')
        require(ids.dtype == np.int64 and np.array_equal(ids, np.arange(size)), 'Feature IDs')
        require(values.shape == (size, 36864) and values.dtype == np.float32, 'Feature count shape/dtype')
        with gzip.open(DATA / (prefix + '-labels-idx1-ubyte.gz'), 'rb') as stream:
            require(struct.unpack('>II', stream.read(8)) == (2049, source_count), 'IDX labels header')
            official = np.frombuffer(stream.read(size), dtype=np.uint8).astype(np.int64)
        require(labels.dtype == np.int64 and np.array_equal(labels, official), 'Labels differ from official fixed split')
        total_spikes = 0; silent_rows = 0
        for row in ev[split]:
            block = values[row['start']:row['end']]
            require(np.isfinite(block).all() and ((block >= 0) & (block <= 100) & (block == np.round(block))).all(), 'Invalid spike counts')
            require(array_sha(block) == row['counts_sha256'] and row['frozen_state'] == frozen, 'Feature hash/new-weight freeze')
            # Runtime sums float32, whose integer precision is limited for large batches.
            require(abs(float(block.sum(dtype=np.float64)) - row['output_spikes']) <= max(4, abs(row['output_spikes']) * 2e-6), 'Feature spike telemetry')
            total_spikes += int(block.sum(dtype=np.float64))
            silent_rows += int((block.sum(axis=1, dtype=np.float64) == 0).sum())
        counts[split] = values
        activity[split] = dict(rows=size, total_spikes=total_spikes, silent_rows=silent_rows,
                               uninformative_all_silent=total_spikes == 0)
    # Independent CPU implementation of the frozen train-only scaler; bounded 128-row chunks.
    total = 0; mu = None; m2 = None
    for start_row in range(0, 30000, 128):
        x = torch.log1p(torch.from_numpy(np.array(counts['train'][start_row:start_row + 128], copy=True))).double()
        n = len(x); local = x.mean(0); local_m2 = ((x - local) ** 2).sum(0)
        if mu is None: mu, m2 = local, local_m2
        else:
            delta = local - mu; m2 += local_m2 + delta.square() * (total * n / (total + n)); mu += delta * (n / (total + n))
        total += n
    mean = load('scaler_mean.npy'); sd = load('scaler_sd.npy')
    require(mean.dtype == sd.dtype == np.float32 and np.array_equal(mean, mu.float().numpy())
            and np.array_equal(sd, (m2 / (total - 1)).sqrt().float().clamp_min(1e-6).numpy()), 'Train-only scaler replay mismatch')
    for i, row in enumerate(ev['heads']):
        order = load('epoch%02d_train_order.npy' % i)
        require(order.dtype == np.int64 and np.array_equal(np.sort(order), np.arange(30000)) and array_sha(order) == row['order_sha256'], 'Epoch permutation')
        require(np.isfinite(row['train_loss']) and row['train_loss'] >= 0, 'Readout loss')
    expected_frozen = {'head_weight.npy', 'head_bias.npy', 'scaler_mean.npy', 'scaler_sd.npy', 'trained_model.pt'}
    require(set(ev['frozen']['sha256']) == expected_frozen, 'Head/scaler freeze inventory')
    for name, pin in ev['frozen']['sha256'].items(): require(sha(session / name) == pin, 'Changed after test freeze: ' + name)
    weight = load('head_weight.npy'); hbias = load('head_bias.npy')
    require(weight.shape == (10, 36864) and hbias.shape == (10,) and weight.dtype == hbias.dtype == np.float32
            and np.isfinite(weight).all() and np.isfinite(hbias).all(), 'Final readout arrays')
    result = predictions(load('test_logits.npy'), load('test_predictions.npy'), load('test_labels.npy'), load('test_confusion.npy'))
    replay = linear_replay(counts['test'], mean, sd, weight, hbias, load('test_logits.npy'))
    for key, value in result.items(): require(m[key] == value and ev['evaluation'][key] == value, 'Test metric mismatch: ' + key)
    require(m['head_updates'] == 1475 and m['test_evaluations'] == 1 and m['selected_epoch'] == 25, 'Final head/test counters')
    require(ev['evaluation'] == dict(kind='test_complete', **result, test_evaluations=1,
            selected_epoch=25, hardware_accuracy=False, reconstruction=True), 'Exact final evaluation event')
    wall = (datetime.fromisoformat(m['finished_utc']) - datetime.fromisoformat(m['started_utc'])).total_seconds()
    return dict(schema='mnist-full-stdp-saved-evidence-audit-v1', status='pass' if replay['passed'] else 'review_required', session=session.relative_to(ROOT).as_posix(),
                auditor_sha256=sha(__file__), manifest_sha256=sha(session / 'manifest.json'), planned=m['planned'], actual=ev['actual'],
                training=dict(timesteps=3000000, subsequent_images_with_change=changed,
                    first_step_pure_clipping=True, first_window_post_clipping_changed=int(np.count_nonzero(trace[-1] - clipped)),
                    final_weight_sha256=array_sha(final), seconds=summary['training_seconds'],
                    final_min=float(final.min()), final_max=float(final.max()),
                    final_at_zero=int((final == 0).sum()), final_at_one=int((final == 1).sum())),
                feature_activity=activity,
                train_only_scaler_replayed_exact=True, results_verified=len(m['result_sha256']), wall_seconds=wall,
                physical_memristor=False, reconstruction=True, linear_readout_replay=replay, **result,
                limits=['No SNN/STDP forward reexecution; saved state trajectory and runtime telemetry are checked.',
                        'Head optimizer updates are evidenced by frozen code, epoch logs and permutations, not independently retrained.',
                        'CPU float64 readout reference uses fixed tolerances; any logit violation or argmax disagreement requires review.',
                        'Feature input encodings/output spike trains are not independently replayed.',
                        'Fresh seed/dependency reconstruction does not guarantee historical69.65%.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); target = inside(ROOT, args.output)
    require(target.is_relative_to(ROOT / 'reports'), 'Audit outputs belong in reports/')
    require(not target.exists(), 'Do not overwrite an existing report')
    result = audit(args.session)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x', encoding='utf-8') as stream: json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(result, allow_nan=False))
    return 0 if result['status'] == 'pass' else 3


if __name__ == '__main__': raise SystemExit(main())
