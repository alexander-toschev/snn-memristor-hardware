"""Compare two completed full case0 reports/sidecars; stdlib, no raw/model/device I/O."""
import argparse
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath
import statistics

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = 'experiments/cifar_full_case0_20261019/'
PROTOCOL_SHA = '2cb1d57689508725f7155bf7b1d0d17fec9d4f6b65782171e2b85481bc9aa558'
DIGITAL_SHA = 'a29140edd81334de22b66d93cd2fbbccc68b524d87a02ecd49ef6f36979201b4'
PINS = {
    'audit_native.py': 'd3cd8bf2cf11512f81aabff0300c5fde391dad240fd9008834150e698910e717',
    'classifier.py': '64ba6e85cda19367f93a0c8294f77b6c9b6f5f26cbf40d188685bc47430053e5',
    'collector.py': '03c6121560ca558904c5722228b4c1a6e7e8eb6d7264b6fb6f18cc393b2e5c21',
    'experiment.py': '5e8647da40961558fd76276d7a5b5eef188e57b77dfb44478a86cb2e376a70cf',
    'plan.py': 'd795d6f820eeacca36e6746746bcfd558850bc5d1bedce32cc969820b3ce94d7',
    'reconstruct.py': '516810cdc7d8a23d18d3487c98bf0de53659bd6f82bc126c43d9481a46ade9b7',
    'trace.py': 'c8093a346b46f2124b389fe9bd7f9d1fea6ac161dce7892df70181b058542d37'}
CASE = dict(original_test_index=9, fine_label=0, remapped_label=0,
            image_chw_uint8_sha256='877945525f9274b34e0f0052ed145993db5cb488ffca908a45b9d10123a9d774')
LABELS = [0, 1, 2, 3, 5, 8, 13, 14, 17, 19]
FIXTURE_SHA = '8cb9ec0fe30242328527073bb076f8019295d73a748343f58fc2a8671bf7c154'
CHECKPOINT_SHA = 'ee4fc4f58a3d010218116e9ad4c7ba808787959f9ed7687cd58cbb32d90f62fb'
CPU_ATOL = CPU_RTOL = 1e-5


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''): h.update(block)
    return h.hexdigest()


def digest(value): return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def local(value):
    p = Path(value); win = PureWindowsPath(value)
    require(not p.is_absolute() and not win.drive and not win.root, 'Use repository-relative paths')
    path = (ROOT / p).resolve(); require(path.is_relative_to(ROOT), 'Path escapes repository')
    return path


def finite(value):
    require(isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value), 'Expected finite number')
    return float(value)


def read_json(path):
    def reject(value): raise ValueError('Nonfinite JSON constant: ' + value)
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'Duplicate JSON key'); result[key] = value
        return result
    result = json.loads(path.read_text(encoding='utf-8'), parse_constant=reject, object_pairs_hook=unique)
    def check(value):
        if isinstance(value, float): require(math.isfinite(value), 'Nonfinite JSON number')
        elif isinstance(value, dict):
            for item in value.values(): check(item)
        elif isinstance(value, list):
            for item in value: check(item)
    check(result)
    return result


def matrix(value):
    require(isinstance(value, list) and len(value) == 12 and all(isinstance(row, list) and len(row) == 10 for row in value), 'Expected12x10 output')
    return [[finite(x) for x in row] for row in value]


def decode(value, true_label):
    spikes = matrix(value['output_spikes']); membranes = matrix(value['output_membranes_post_reset'])
    require(all(x in (0., 1.) for row in spikes for x in row), 'Nonbinary output spikes')
    scores = [finite(x) for x in value['twd_scores']]; require(len(scores) == 10, 'TWD class count')
    first = [next((t for t in range(12) if spikes[t][c]), 12) for c in range(10)]
    totals = [sum(row[c] for row in spikes) for c in range(10)]
    silent = not any(totals); winners = [i for i, t in enumerate(first) if t == min(first)]
    require(value['all_output_classes_silent'] is silent and value['first_spike_winner_count'] == len(winners), 'Silence/first-spike metadata')
    require(not silent or all(x == 0 for x in scores), 'Silent spikes but nonzero TWD')
    membrane_scores = [math.fsum(row[c] for row in membranes) for c in range(10)]
    require(set(value['predicted_subset_indices']) == {'first_spike', 'membrane', 'twd'}, 'Decoder inventory')
    decoded = {}
    for mode, metric, minimize in [('first_spike', first, True), ('membrane', membrane_scores, False), ('twd', scores, False)]:
        selected = value['predicted_subset_indices'][mode]
        require(type(selected) is int and 0 <= selected < 10, 'Prediction outside class range')
        best = min(metric) if minimize else max(metric)
        exact = [i for i, x in enumerate(metric) if x == best]
        # float32 temporal membrane reduction can differ slightly from stdlib fsum.
        tolerance = CPU_ATOL + CPU_RTOL * abs(best) if mode == 'membrane' else 0.
        near = [i for i, x in enumerate(metric) if abs(x - best) <= tolerance]
        require(selected in near and (mode == 'membrane' or selected == exact[0]), 'Mechanical decoder differs from saved outputs')
        require(value['predicted_cifar100_fine_labels'][mode] == LABELS[selected], 'Fine-label decoding mismatch')
        valid = (not silent if mode in ('first_spike', 'twd') else True) and len(near) == 1
        decoded[mode] = dict(mechanical_subset_index=selected, mechanical_fine_label=LABELS[selected],
            mechanical_matches_true_label=selected == true_label, valid_unique_decision=valid,
            valid_correct=selected == true_label if valid else None, exact_winners=exact, numerical_tie_candidates=near,
            reason='no_output_spike' if silent and mode in ('first_spike', 'twd') else ('tie_or_numerically_ambiguous' if not valid else 'unique'),
            scores=metric)
    return dict(total_spikes=sum(totals), spike_counts_by_class=totals, first_spike_times=first,
                all_output_classes_silent=silent, first_spike_winner_count=len(winners), decoders=decoded,
                output_spikes=spikes, output_membranes_post_reset=membranes, twd_scores=scores)


def validate_report(report, source):
    require(report['schema'] == 'full-case0-classification-v1' and report['source'] == source + '_conv1_digital_remaining_network', 'Wrong report schema/source')
    require(report['registered_classifier_import_verified'] is True and report['full_dataset_accuracy_measured'] is False, 'Unqualified report/accuracy claim')
    require(report['classifier_sha256'] == PINS['classifier.py'] and report['auditor_sha256'] == PINS['audit_native.py']
            and report['digital_downstream_sha256'] == DIGITAL_SHA, 'Unreviewed classifier chain')
    require(report['case'] == CASE and report['output_class_fine_labels'] == LABELS, 'Different fixed case/classes')
    require(report['artifact_sha256']['weights/best_fs.pt'] == CHECKPOINT_SHA
            and report['artifact_sha256']['weights/cifar100_conv1_fixture.json'] == FIXTURE_SHA, 'Different checkpoint/fixture')
    audit = report['native_audit']
    require(audit['source'] == source and audit['scope'] == 'full' and audit['status'] == 'native_trace_replayed'
            and audit['complete_full_tensor'] is True and audit['observed_coefficients'] == 4800
            and audit['covered_tensor_elements'] == 1806336, 'Incomplete native acquisition')
    for key in ('heldout', 'retention'):
        require(len(audit[key]) == 48 and sorted(row['stage'] for row in audit[key]) == list(range(48)), 'Missing/duplicate stage: ' + key)
    rows = audit['weight_observations']; require(len(rows) == 4800, 'Missing weight observations')
    seen = set()
    for row in rows:
        f, tap = row['filter'], row['tap']; require(type(f) is int and type(tap) is int and 0 <= f < 64 and 0 <= tap < 75, 'Coefficient outside scope')
        require((f, tap) not in seen, 'Duplicate coefficient'); seen.add((f, tap))
        tile = tap // 32; bl = tap % 32; stage = (f // 4) * 3 + tile; pair = f % 4
        require(row['stage'] == stage and row['bl'] == bl and row['positive_cell'] == [2 * pair, bl]
                and row['negative_cell'] == [2 * pair + 1, bl] and row['per_coefficient_fitting_used'] is False, 'Coefficient layout/fitting mismatch')
    require(set(report['predictions']) == {'cpu_reference', 'raw', 'calibrated'} and set(report['conv1_errors']) == {'raw', 'calibrated'}, 'Branch inventory')
    for branch in report['conv1_errors'].values():
        require(set(branch) == {'conv1_rmse', 'conv1_max_abs', 'relative_l2'} and all(finite(x) >= 0 for x in branch.values()), 'Invalid Conv1 errors')
    cv = report['cpu_validation']
    require(0 <= finite(cv['direct_forward_max_abs']) <= 1e-6
            and cv['tiled_forward_max_abs']['spikes'] == 0
            and all(0 <= finite(x) <= 1e-5 for x in cv['tiled_forward_max_abs'].values()), 'CPU forward selfcheck failed')
    return {name: decode(value, CASE['remapped_label']) for name, value in report['predictions'].items()}


def bind_session(report, session, source):
    require(session.parent == ROOT / 'results/cifar_full_case0_20261019', 'Wrong completed-session namespace')
    manifest_path = session / 'manifest.json'; m = read_json(manifest_path)
    require(sha(manifest_path) == report['acquisition_manifest_sha256'], 'Report is not bound to supplied manifest')
    require(m['status'] == 'completed' and m['scope'] == 'full' and m['source'] == source and m['session_id'] == session.name, 'Session incomplete/wrong source')
    require(m['protocol_sha256'] == PROTOCOL_SHA, 'Different physical/digital protocol')
    p = read_json(session / 'protocol.json')
    require(digest(p) == PROTOCOL_SHA and sha(session / 'protocol.json') == m['result_sha256']['protocol.json'], 'Protocol sidecar hash')
    require(p['simulator_gain'] == .015 and p['hardware_gain'] == .125, 'Source gains changed')
    require(type(m['impacts']) is int and 0 < m['impacts'] <= p['maximum_impacts']
            and type(m['pulses']) is int and 0 <= m['pulses'] <= p['maximum_programming_pulses'], 'Recorded finite acquisition budget exceeded')
    require(m['impacts'] == report['native_audit']['native_impacts'] and m['pulses'] == report['native_audit']['programming_pulses'], 'Impact/pulse attestation mismatch')
    sources = m['source_sha256']
    for name, pin in PINS.items(): require(sources.get(DIRECTORY + name) == pin, 'Recorded source pin mismatch: ' + name)
    for name in ('vendor/MemriBoard/manager/board.py', 'vendor/MemriBoard/manager/service/converters.py',
                 'vendor/MemriBoard/manager/service/drivers.py', 'configs/raspberry_board.ini', 'weights/conv1_weights.npy'):
        require(name in sources, 'Required recorded source missing')
    verified = {}
    for name, pin in sources.items():
        path = local(name)
        require(path.is_file(), 'Required local source missing; use complete reviewed Pi checkout: ' + name)
        require(sha(path) == pin, 'Local source changed: ' + name); verified[name] = pin
    require(sha(ROOT / 'full_cifar_classifier.py') == DIGITAL_SHA, 'Local downstream source changed')
    for name, pin in report['artifact_sha256'].items(): require(sha(local(name)) == pin, 'Classifier artifact changed: ' + name)
    return dict(session=session.relative_to(ROOT).as_posix(), manifest_sha256=sha(manifest_path), protocol_sha256=PROTOCOL_SHA,
                source_sha256=sources, locally_byte_verified_source_sha256=verified,
                all_recorded_source_bytes_verified=True,
                execution_environment=m['execution_environment'], raw_replayed_by_comparison=False)


def weight_summary(rows, source):
    gain = .015 if source == 'simulator' else .125
    target = [finite(r['trained_weight']) for r in rows]; norm = math.sqrt(math.fsum(x * x for x in target))
    require(norm > 0, 'Zero target weight norm')
    result = {}
    for key in ('immediate_weight', 'post_all_raw_weight', 'post_all_calibrated_weight', 'final_raw_weight', 'final_calibrated_weight'):
        values = [finite(r[key]) for r in rows]; delta = [v - t for v, t in zip(values, target)]
        within = [abs(d * gain) <= max(.00061, abs(t * gain) * .05) for d, t in zip(delta, target)]
        if key == 'immediate_weight':
            require(all(r['immediate_within_tolerance'] is ok for r, ok in zip(rows, within)), 'Immediate tolerance telemetry')
        result[key] = dict(rmse=math.sqrt(math.fsum(d * d for d in delta) / len(delta)),
            relative_l2=math.sqrt(math.fsum(d * d for d in delta)) / norm, maximum_absolute_error=max(abs(d) for d in delta),
            direct_within_tolerance=sum(within), total=len(rows), exact_zero_effective=sum(v == 0 for v in values),
            wrong_sign_nonzero_target=sum(v * t < 0 for v, t in zip(values, target)),
            zero_effective_nonzero_target=sum(v == 0 and t != 0 for v, t in zip(values, target)))
    return dict(gain_v_per_weight=gain, tolerance='abs((effective-trained)*gain)<=max(.00061V,.05*abs(trained*gain))', phases=result)


def heldout_summary(rows):
    rows = sorted(rows, key=lambda row: row['stage'])
    raw = [finite(r['raw_rmse_v']) for r in rows]; cal = [finite(r['calibrated_rmse_v']) for r in rows]
    require(all(x >= 0 for x in raw + cal), 'Negative heldout RMSE')
    return dict(stages=48, raw_mean_rmse_v=statistics.fmean(raw), calibrated_mean_rmse_v=statistics.fmean(cal),
                improved=sum(c < r for c, r in zip(cal, raw)), unchanged=sum(c == r for c, r in zip(cal, raw)),
                worsened=sum(c > r for c, r in zip(cal, raw)), per_stage=rows)


def cpu_compare(sim, hardware):
    stats = {}; passed = True
    for key in ('output_membranes_post_reset', 'output_spikes', 'twd_scores'):
        left = sim[key]; right = hardware[key]
        if key != 'twd_scores': left = sum(left, []); right = sum(right, [])
        differences = [abs(a - b) for a, b in zip(left, right)]
        bad = sum(d > (0 if key == 'output_spikes' else CPU_ATOL + CPU_RTOL * max(abs(a), abs(b))) for a, b, d in zip(left, right, differences))
        stats[key] = dict(maximum_absolute_difference=max(differences), outside_tolerance=bad)
        passed = passed and bad == 0
    prediction_differences = [mode for mode in sim['decoders'] if sim['decoders'][mode]['mechanical_subset_index'] != hardware['decoders'][mode]['mechanical_subset_index']]
    return dict(passed=passed and not prediction_differences, absolute_tolerance=CPU_ATOL, relative_tolerance=CPU_RTOL,
                spike_tolerance=0, arrays=stats, prediction_differences=prediction_differences)


def compare(sim, hardware):
    decoded = {'simulator': validate_report(sim, 'simulator'), 'hardware': validate_report(hardware, 'hardware')}
    require(sim['artifact_sha256'] == hardware['artifact_sha256'], 'Classifier artifacts differ')
    sr = sorted(sim['native_audit']['weight_observations'], key=lambda r: (r['filter'], r['tap']))
    hr = sorted(hardware['native_audit']['weight_observations'], key=lambda r: (r['filter'], r['tap']))
    require(all(s['trained_weight'] == h['trained_weight'] for s, h in zip(sr, hr)), 'Trained coefficient targets differ')
    cpu = cpu_compare(decoded['simulator']['cpu_reference'], decoded['hardware']['cpu_reference'])
    summaries = {}
    for source, report in [('simulator', sim), ('hardware', hardware)]:
        audit = report['native_audit']
        summaries[source] = dict(weights=weight_summary(audit['weight_observations'], source), heldout=heldout_summary(audit['heldout']),
            retention=audit['retention'], conv1_errors=report['conv1_errors'], downstream=decoded[source],
            impacts=audit['native_impacts'], programming_pulses=audit['programming_pulses'],
            maximum_shared_zero_shift_v=audit['maximum_shared_zero_shift_v'],
            maximum_position_bracket_duration_s=audit['maximum_position_bracket_duration_s'])
    return dict(schema='full-cifar-case0-sim-hardware-comparison-v1', status='pass' if cpu['passed'] else 'review_required',
        status_semantics='pass means report consistency and compatible CPU controls, not correct CIFAR inference or hardware equivalence',
        case=CASE, output_class_fine_labels=LABELS, coverage=dict(coefficients=4800, stages=48, tensor_elements=1806336, cases=1),
        protocol_sha256=PROTOCOL_SHA, cpu_comparator=cpu, sources=summaries,
        full_dataset_accuracy_measured=False, calibrated_matrix_repairs_trained_weights=False,
        interpretation=['One fixed CIFAR100-subset example, not test-set accuracy.',
            'Conv1 contributions come from MemriBoard; downstream BN/LIF and remaining network are digital.',
            'SIM gain .015V/weight and HW .125V/weight are different nominal mappings: cross-source confound, not calibrated physical equivalence.',
            'Silent first-spike/TWD argmax preserves mechanical label but is not a valid successful classification.',
            'TWD scores are trusted from the registered classifier; this tool validates decoder/ties, not learned softplus(alpha) replay.',
            'Comparison trusts SHA-bound completed registered classifier audits; it does not replay raw or rerun a model.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('sim-report', 'hardware-report', 'sim-session', 'hardware-session', 'output'): parser.add_argument('--' + flag, required=True)
    args = parser.parse_args(); target = local(args.output)
    require(target.is_relative_to(ROOT / 'reports') and not target.exists(), 'Output must be a new report path')
    sp = local(args.sim_report); hp = local(args.hardware_report)
    sim = read_json(sp); hardware = read_json(hp)
    result = compare(sim, hardware)
    result['binding'] = dict(simulator=bind_session(sim, local(args.sim_session), 'simulator'),
                             hardware=bind_session(hardware, local(args.hardware_session), 'hardware'))
    result['input_reports'] = dict(simulator=dict(path=args.sim_report, sha256=sha(sp)), hardware=dict(path=args.hardware_report, sha256=sha(hp)))
    result['comparison_tool_sha256'] = sha(__file__)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x', encoding='utf-8') as stream: json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(dict(status=result['status'], output=args.output)))
    return 0 if result['status'] == 'pass' else 3


if __name__ == '__main__': raise SystemExit(main())
