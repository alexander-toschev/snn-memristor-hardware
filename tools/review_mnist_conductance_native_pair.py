"""Saved-evidence second review; no acquisition imports, raw replay or model."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FREEZE = 'reports/2026-10-19/mnist-physical-design/PILOT128_CODE_FREEZE.json'
FREEZE_SHA = '54c0865f841430187aa6e242c3698a8b721a5d207dda273119d1ecb8bc22d7b8'
AUDITOR = 'tools/audit_mnist_conductance.py'
AUDITOR_SHA = 'ca36aaeba54d22a8cee066227cf175917b5e3c5edd67d5975708b671c9a30825'


def require(value, message):
    if not value:
        raise ValueError(message)


def local(name):
    path = (ROOT / name).resolve()
    require(not Path(name).is_absolute() and path.is_relative_to(ROOT), 'Repository-relative path required')
    return path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def phase_summary(outcomes):
    result = {}
    for phase in ('preprobe', 'train', 'postprobe', 'retentionprobe'):
        images = [row for row in outcomes if row['phase'] == phase]
        require(len(images) == (128 if phase == 'train' else 16), 'Phase image count')
        posts = [[row['neurons'][wl]['posts'] for row in images] for wl in range(8)]
        require(all(len(p) == 32 and all(type(x) is int and x in (0, 1) for x in p)
                    for neuron in posts for p in neuron), 'Binary LIF trace shape')
        total = [sum(map(sum, neuron)) for neuron in posts]
        signatures = [tuple(bit for row in images for bit in row['neurons'][wl]['posts'])
                      for wl in range(8)]
        result[phase] = dict(
            images=len(images), posts_by_wl=total,
            silent_images_by_wl=[sum(not any(p) for p in neuron) for neuron in posts],
            maximum_posts_per_image_by_wl=[max(map(sum, neuron)) for neuron in posts],
            all_32_steps_spiking_images_by_wl=[sum(all(p) for p in neuron) for neuron in posts],
            distinct_neuron_traces=len(set(signatures)),
            distinct_learning_neuron_traces=len(set(signatures[:4])),
            unique_eight_neuron_outputs=len({tuple(bit for n in row['neurons'] for bit in n['posts'])
                                             for row in images}),
            all_learning_traces_identical=len(set(signatures[:4])) == 1)
    return result


def bank_delta(before, after):
    result = {}
    for name, wls in [('learning', range(4)), ('control', range(4, 8))]:
        values = [after[wl][bl] - before[wl][bl] for wl in wls for bl in range(32)]
        require(all(math.isfinite(x) for x in values), 'Nonfinite conductance')
        result[name] = dict(changed_cells=sum(x != 0 for x in values),
                           positive=sum(x > 0 for x in values), negative=sum(x < 0 for x in values),
                           sum_abs_delta=sum(map(abs, values)), maximum_abs_delta=max(map(abs, values)))
    return result


def inspect(audit_path, freeze):
    audit = read(audit_path)
    require(audit['status'] == 'PASS' and audit['source'] == 'simulator'
            and audit['outcome'] == 'completed', 'Completed independently audited nativeSIM required')
    require(audit['analyzer_sha256'] == sha(local(AUDITOR)) == AUDITOR_SHA, 'Auditor source identity')
    require(audit['code_freeze_sha256'] == FREEZE_SHA, 'Audit freeze identity')
    session = local(audit['session'])
    require(session.parent == ROOT / 'results/mnist_conductance_20261019', 'Session namespace')
    manifest = read(session / 'manifest.json')
    require(sha(session / 'manifest.json') == audit['manifest_sha256'], 'Manifest audit binding')
    require(manifest['status'] == 'completed' and manifest['source'] == 'simulator'
            and manifest['session_id'] == session.name, 'Manifest source/scope')
    require(manifest['protocol_sha256'] == audit['protocol_sha256'] == freeze['protocol_sha256'],
            'Protocol identity')
    require(read(session / 'protocol.json') == freeze['protocol'], 'Protocol content')
    actual = {p.relative_to(session).as_posix() for p in session.rglob('*') if p.is_file()} - {'manifest.json'}
    require(actual == set(manifest['result_sha256']), 'Exact result inventory')
    for name, pin in manifest['result_sha256'].items():
        path = (session / name).resolve()
        require(path.is_relative_to(session) and sha(path) == pin, 'Result bytes changed')
    require(len(manifest['source_sha256']) == freeze['acquisition_source_count'], 'Source count')
    for name, pin in manifest['source_sha256'].items():
        require(freeze['source_sha256'].get(name) == pin and sha(local(name)) == pin, 'Source bytes changed')
    require(sha(session / 'array_initial.cb') == manifest['array_initial_sha256'], 'Initial SIM state identity')
    index = read(session / 'raw/index.json')
    require(sha(session / 'raw/index.json') == audit['raw_index_sha256'], 'Raw index binding')
    require(sum(p['records'] for p in index['parts']) == audit['records'], 'Raw records')
    for part in index['parts']:
        path = session / 'raw' / part['path']
        require(sha(path) == part['compressed_sha256'] and path.stat().st_size == part['compressed_bytes'],
                'Closed raw part identity')
    obs = read(session / 'observations.json')
    require(obs['classification_accuracy_measured'] is False and obs['physical_lif_integrator'] is False
            and obs['full10k_executed'] is False, 'Scientific scope')
    require(obs['measured_images'] == manifest['measured_images'] == audit['measured_images'] == 208,
            'Image exposure count')
    require(obs['gain'] == manifest['gain'] == audit['selected_gain'], 'Frozen gain binding')
    require(obs['impacts'] == manifest['impacts'] == audit['native_impacts'], 'Impact count')
    require(obs['pulses'] == manifest['pulses'] == audit['pulses'], 'Pulse count')
    training = [u for u in obs['updates'] if u['phase'] == 'training_update']
    admitted = [u for u in training if u['admitted']]
    require(all(u['wl'] < 4 for u in training), 'Training writes on control WL')
    occupancy = Counter((u['wl'], u['bl']) for u in admitted)
    sign = Counter()
    for update in admitted:
        change = 1 / update['post']['r'] - 1 / update['guard']['r']
        require(abs(change - update['delta_inverse_r']) < 1e-15, 'Post-read delta binding')
        sign['unchanged' if change == 0 else
             'requested_direction' if ((change > 0) == bool(update['sign'])) else 'opposite_direction'] += 1
    checkpoints = {c['phase']: c['mean_inverse_r'] for c in obs['checkpoints']}
    require(len(checkpoints) == 14, 'Checkpoint coverage')
    phase = phase_summary(obs['outcomes'])
    for name, stats in phase.items():
        observed = obs['diagnostics']['phases'][name]
        require(stats['posts_by_wl'] == [sum(x) for x in observed['post_counts_by_wl']]
                and stats['silent_images_by_wl'] == observed['silent_images_by_wl']
                and stats['unique_eight_neuron_outputs'] == observed['unique_eight_neuron_outputs']
                and stats['all_learning_traces_identical'] == observed['all_learning_traces_identical'],
                'Recomputed phase diagnostics differ')
    result = dict(session=audit['session'], audit_path=audit_path.relative_to(ROOT).as_posix(),
                  audit_sha256=sha(audit_path), manifest_sha256=sha(session / 'manifest.json'),
                  source_files_verified=len(manifest['source_sha256']), result_files_verified=len(actual),
                  raw_closed_parts_verified=len(index['parts']), native_impacts=obs['impacts'],
                  pulses=obs['pulses'], selected_gain=obs['gain'], phase_summary=phase,
                  training_attempts=len(training), admitted_training_pulses=len(admitted),
                  distinct_training_updated_cells=len(occupancy),
                  training_pulses_by_wl=[sum(v for (wl, _), v in occupancy.items() if wl == i) for i in range(8)],
                  training_update_sign_response=dict(sign),
                  before_to_after_training=bank_delta(checkpoints['after_preprobe'], checkpoints['after_train_128']),
                  post_to_retention=bank_delta(checkpoints['after_postprobe'], checkpoints['after_retentionprobe']),
                  runtime=manifest['invocation'], array_initial_sha256=manifest['array_initial_sha256'],
                  reference=manifest['reference'])
    return result, obs, manifest


def review(main_path, reference_path):
    require(sha(local(FREEZE)) == FREEZE_SHA, 'Reviewed source freeze changed')
    freeze = read(local(FREEZE))
    for name, pin in freeze['source_sha256'].items():
        require(sha(local(name)) == pin, 'Frozen source/test/fixture changed')
    main, mo, mm = inspect(main_path, freeze)
    reference, ro, rm = inspect(reference_path, freeze)
    require(mm['reference'] is None and rm['reference'] is not None, 'Main/reference roles')
    binding = rm['reference']
    require(binding['session'] == main['session']
            and binding['manifest_sha256'] == main['manifest_sha256']
            and binding['observations_sha256'] == sha(local(main['session']) / 'observations.json')
            and binding['fit_raw_index_sha256'] == sha(local(main['session']) / 'raw/index.json')
            and binding['gain'] == main['selected_gain']
            and binding['array_initial_sha256'] == main['array_initial_sha256'],
            'Reference not bound to this verified main')
    calibration = mo['calibration']
    require(calibration['fit_train_indices'] == list(range(11000, 11032))
            and calibration['control_used'] is False and calibration['labels_used'] is False,
            'Fit/control/label separation')
    eligible = []
    for candidate in calibration['candidates']:
        rows = candidate['per_wl']
        require([r['wl'] for r in rows] == list(range(4)), 'Fit WL scope')
        good = all(len(r['post_counts']) == 32 and 8 <= sum(r['post_counts']) <= 64
                   and sum(x > 0 for x in r['post_counts']) >= 4
                   and max(r['post_counts']) <= 8 and r['empty_input_posts'] == 0 for r in rows)
        if good:
            eligible.append((abs(sum(sum(r['post_counts']) for r in rows) - 128), candidate['gain']))
    require(eligible and min(eligible)[1] == main['selected_gain'], 'Independent fixed gain choice')
    require(main['array_initial_sha256'] == reference['array_initial_sha256'], 'Unmatched initial state')
    require(main['selected_gain'] == reference['selected_gain'], 'Unmatched frozen gain')
    require(main['admitted_training_pulses'] > 0 and reference['admitted_training_pulses'] == 0,
            'Learning/no-update roles not demonstrated')
    require(reference['pulses'] == 2048, 'Reference pulses beyond preparation')
    require(mm['fixture_sha256'] == rm['fixture_sha256'], 'Unmatched fixture')
    require(mo['input_event_counts'] == ro['input_event_counts'], 'Input exposures differ')
    comparisons = {}
    for phase in ('preprobe', 'postprobe', 'retentionprobe'):
        a = [x for x in mo['outcomes'] if x['phase'] == phase]
        b = [x for x in ro['outcomes'] if x['phase'] == phase]
        require([x['train_index'] for x in a] == [x['train_index'] for x in b], 'Probe order differs')
        comparisons[phase] = dict(
            exact_eight_neuron_trace_matches=sum(
                [n['posts'] for n in x['neurons']] == [n['posts'] for n in y['neurons']]
                for x, y in zip(a, b)),
            hamming_bits_by_wl=[sum(p != q for x, y in zip(a, b)
                                   for p, q in zip(x['neurons'][wl]['posts'], y['neurons'][wl]['posts']))
                                for wl in range(8)])
    return dict(schema='mnist-native-pair-second-review-v1',
                status='native_sim_mechanism_qualified_not_efficacy',
                reviewer_sha256=sha(Path(__file__)), freeze_sha256=FREEZE_SHA,
                freeze_files_verified=len(freeze['source_sha256']), main=main, reference=reference,
                probe_comparisons=comparisons, full_raw_replay_repeated=False,
                relies_on_bound_independent_raw_audits=True, hardware_grant_issued=False,
                full10k_qualified=False, classification_accuracy_measured=False,
                limitations=['No intrinsic physical STDP or physical LIF proof.',
                             'Common initial seed does not align random draws after update paths diverge.',
                             'Readout labels/test and classification head are absent.',
                             'Diversity and conductance changes do not prove useful features or learning benefit.',
                             'Simulator waiting interval does not establish physical retention.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main-audit', required=True)
    parser.add_argument('--reference-audit', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = local(args.output)
    require(output.is_relative_to(ROOT / 'reports') and not output.exists(), 'New report path required')
    result = review(local(args.main_audit), local(args.reference_audit))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=result['status'], output=args.output)))


if __name__ == '__main__':
    main()
