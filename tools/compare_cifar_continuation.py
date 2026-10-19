"""Local report/payload comparison: original full SIM versus complete HW continuation.

No raw replay, model execution, training, network or physical I/O. Each schema
keeps its own provenance; a composite is never disguised as a legacy session.
"""
import argparse
import hashlib
import math
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import compare_full_cifar_reports as metrics

PINNED_TOOLS = {
    'tools/compare_full_cifar_reports.py': 'a2d02825ba7501f7bdc87db3e8c79c1747ed9cb90d020597eb8031101895f192',
    'tools/classify_cifar_continuation.py': 'a3d61cbf2ac89811b5aa6f4299a6294e3eed3ce1be8baa0bf614277ac756ec73',
    'tools/audit_cifar_interrupted_prefix.py': '1c19c6e4b0db4af3892a681035b27123633d2ca7c840f1537928b13e35ec3b00',
    'tools/cifar_prefix_replay.py': '0f6bb5743dd8b4218b68701c65749db00874d8e444d6df18feda08f127a39c69',
    'tools/cifar_stage_source_registry.json': '15c6f2729df901c5aa9a92f92645cd7ecd7e58dd4e6a596ac9a8afc1e1632583',
}
SPEC_SHA = '95e0eb4c7a6694486060c7dec5436e3c3d29357d4d22bd7725c4b08685da3560'
WEIGHTS_SHA = '8be312b9eafb9de23f8340b17f848e7e0e2e7884293c826a6d9f6cbb5125d440'
STAGE_BYTES = 12 * 4 * 28 * 28 * 4
FULL_BYTES = 12 * 64 * 28 * 28 * 3 * 4
require = metrics.require
read_json = metrics.read_json
sha = metrics.sha
digest = metrics.digest
local = metrics.local


def verify_tools():
    for name, expected in PINNED_TOOLS.items():
        require(sha(local(name)) == expected, 'Comparison dependency changed: ' + name)


def validate_composite(report):
    require(report.get('schema') == 'cifar-continuation-classification-v1'
            and report.get('source') == 'hardware', 'Expected a hardware continuation report')
    require(report.get('complete_full_tensor') is True
            and report.get('accepted_stage_ids') == list(range(48)), 'Need exactly48 completed stages')
    require(report.get('full_dataset_accuracy_measured') is False
            and report.get('hardware_acquisition_performed') is False, 'Invalid offline/accuracy claim')
    require(report.get('physical_protocol_sha256') == metrics.PROTOCOL_SHA
            and report.get('spec_sha256') == SPEC_SHA, 'Protocol mismatch')
    require(report.get('analyzer_sha256') == PINNED_TOOLS['tools/classify_cifar_continuation.py']
            and report.get('frozen_digital_bridge_sha256') == metrics.PINS['classifier.py']
            and report.get('digital_downstream_sha256') == metrics.DIGITAL_SHA, 'Unreviewed classifier chain')
    require(report.get('case') == metrics.CASE and report.get('output_class_fine_labels') == metrics.LABELS,
            'Wrong fixed case or output classes')
    require(report.get('artifact_sha256', {}).get('weights/best_fs.pt') == metrics.CHECKPOINT_SHA
            and report['artifact_sha256'].get('weights/cifar100_conv1_fixture.json') == metrics.FIXTURE_SHA,
            'Checkpoint/fixture mismatch')
    require(report.get('source_gain_v_per_weight') == .125
            and report.get('gain_confounds_source_comparison') is True, 'Hardware gain mapping mismatch')
    require(set(report.get('predictions', {})) == {'cpu_reference', 'raw', 'calibrated'}
            and set(report.get('conv1_errors', {})) == {'raw', 'calibrated'}, 'Missing complete classifier branches')
    for errors in report['conv1_errors'].values():
        require(set(errors) == {'conv1_rmse', 'conv1_max_abs', 'relative_l2'}
                and all(metrics.finite(v) >= 0 for v in errors.values()), 'Invalid Conv1 error')
    cpu = report['cpu_validation']
    require(0 <= metrics.finite(cpu['direct_forward_max_abs']) <= 1e-6
            and cpu['tiled_forward_max_abs']['spikes'] == 0
            and all(0 <= metrics.finite(v) <= 1e-5 for v in cpu['tiled_forward_max_abs'].values()),
            'CPU forward selfcheck failed')
    require(set(report.get('payload_sha256', {})) == {'raw', 'calibrated'}, 'Missing composite payloads')
    return {name: metrics.decode(value, metrics.CASE['remapped_label'])
            for name, value in report['predictions'].items()}


def validate_stage_report(report, reference):
    require(report.get('schema') == 'cifar-stage-prefix-v1' and report.get('source') == 'hardware'
            and report.get('scope') == 'full' and report.get('protocol_sha256') == metrics.PROTOCOL_SHA,
            'Incompatible stage report')
    require(report.get('status') in ('verified_complete_stage_prefix', 'completed_range')
            and report.get('complete_full_tensor') is False, 'Failed or wrongly promoted prefix report')
    require(report.get('session') == reference['acquisition_session']
            and report.get('session_id') == Path(report['session']).name, 'Stage report/session mismatch')
    stages = [row['stage'] for row in report['accepted_stages']]
    require(all(type(s) is int and 0 <= s < 48 for s in stages) and stages == reference['accepted_stages']
            and stages == list(range(report['first_stage'], report['stop_stage_exclusive'])),
            'Missing, duplicate or reordered stage certificate')
    require(report['native_impacts'] == reference['native_impacts']
            and report['programming_pulses'] == reference['programming_pulses'], 'Stage counters differ')
    expected_sources = {name: PINNED_TOOLS[name] for name in
                        ('tools/audit_cifar_interrupted_prefix.py', 'tools/cifar_prefix_replay.py')}
    if report['provenance'].get('stage_source_registry') is not None:
        expected_sources['tools/cifar_stage_source_registry.json'] = PINNED_TOOLS['tools/cifar_stage_source_registry.json']
    require(report.get('audit_source_sha256') == expected_sources, 'Unreviewed stage audit dependency inventory')
    require(report.get('strict_numeric_replay') in ('PASS', 'FAIL_condition_only'), 'Unqualified numerical replay')
    if report['strict_numeric_replay'] != 'PASS':
        require(report.get('portable_condition_policy') == 'excluded-rank-condition-v1', 'Unknown portable numerical policy')
    diagnostics = report.get('numeric_diagnostics', [])
    require([row.get('stage') for row in diagnostics] == stages, 'Missing or reordered numerical stage diagnostics')
    portable_seen = False
    for row in diagnostics:
        if row.get('strict_numeric_replay') == 'PASS':
            require(row.get('decision_equivalence') == 'exact_within_original_tolerance', 'Contradictory strict numerical decision')
            continue
        portable_seen = True
        require(row.get('strict_numeric_replay') == 'FAIL_condition_only'
                and row.get('decision_equivalence') == 'PASS'
                and row.get('policy') == report.get('portable_condition_policy') == 'excluded-rank-condition-v1',
                'Unqualified portable numerical decision')
        require(row.get('unchanged_other_fields_tolerance') == dict(absolute=1e-10, relative=1e-10),
                'Portable policy changed other-field tolerances')
        for key in ('recorded_condition', 'local_condition'):
            require(key in row and (row[key] is None or metrics.finite(row[key]) > 1e6),
                    'Portable condition outside excluded domain')
        spectra = row.get('spectra', {})
        require(set(spectra) == {'fit', 'reference'}, 'Missing portable rank spectra')
        for spectrum in spectra.values():
            singular = spectrum.get('singular_values', [])
            tol = metrics.finite(spectrum.get('numpy_rank_tolerance'))
            require(spectrum.get('margin_factor') == 16 and spectrum.get('stable') is True
                    and tol > 0 and len(singular) == 8
                    and all(metrics.finite(v) >= 0 and (v >= 16 * tol or v <= tol / 16) for v in singular),
                    'Portable rank stability metadata contradicts margin16')
    require(report['strict_numeric_replay'] == ('FAIL_condition_only' if portable_seen else 'PASS'),
            'Top-level numerical status contradicts stage diagnostics')
    last_task = 0; certificate_pulses = 0
    for certificate in report['accepted_stages']:
        body = {k: v for k, v in certificate.items() if k != 'certificate_sha256'}
        require(digest(body) == certificate['certificate_sha256'], 'Stage certificate changed')
        require(certificate.get('session_id') == report['session_id'], 'Stage certificate session mismatch')
        require(certificate['source'] == 'hardware' and certificate['scope'] == 'full'
                and certificate['shape'] == [12, 4, 28, 28] and certificate['dtype'] == '<f4'
                and certificate['protocol_sha256'] == metrics.PROTOCOL_SHA, 'Wrong stage geometry/source')
        require(certificate['source_provenance_sha256'] == digest(report['provenance'])
                and certificate['raw_inventory_sha256'] == digest(report['raw_inventory']), 'Stage provenance binding mismatch')
        n = (32, 32, 11)[certificate['stage'] % 3]
        require(type(certificate['native_impacts']) is int and 0 < certificate['native_impacts'] <= 15000000
                and type(certificate['programming_pulses']) is int
                and 0 <= certificate['programming_pulses'] <= 4 * n * 24, 'Stage finite budget mismatch')
        require(certificate['task_first'] == last_task + 1
                and certificate['task_last'] - certificate['task_first'] + 1 == certificate['native_impacts'],
                'Stage native task interval differs')
        last_task = certificate['task_last']; certificate_pulses += certificate['programming_pulses']
    require(last_task == report['native_impacts'] and certificate_pulses == report['programming_pulses'],
            'Certificate task/pulse totals differ from range')
    for key in ('heldout', 'retention'):
        require(sorted(row['stage'] for row in report[key]) == stages, 'Missing stage metrics: ' + key)
    require(report['covered_tensor_elements'] == len(stages) * 12 * 4 * 28 * 28,
            'Stage tensor element coverage differs')
    require(report['observed_coefficients'] == sum(4 * (32, 32, 11)[s % 3] for s in stages),
            'Stage coefficient coverage differs')
    require(len(report['weight_observations']) == report['observed_coefficients']
            and all(row['stage'] in stages for row in report['weight_observations']), 'Weight observations belong to another stage range')
    require(metrics.finite(report['maximum_shared_zero_shift_v']) >= 0
            and metrics.finite(report['maximum_position_bracket_duration_s']) >= 0, 'Invalid drift/duration metric')
    return stages


def bind_stage_sidecars(report, reference):
    """Recheck source/receipt/manifest metadata, not raw streams or model math."""
    from tools import audit_cifar_interrupted_prefix as audit
    from tools import classify_cifar_continuation as composer
    session = local(reference['acquisition_session'])
    original = report['status'] == 'verified_complete_stage_prefix'
    expected_namespace = 'cifar_full_case0_20261019' if original else 'cifar_resumable_20261019'
    require(session.parent == ROOT / 'results' / expected_namespace, 'Unexpected acquisition namespace')
    private = report['provenance'].get('private_source_evidence')
    receipt = local(private['receipt_path']) if private else None
    start, actual = audit.verify_sources(session, private_receipt=receipt, stage_session=not original)
    require(report['session_id'] == start['session_id'], 'Stage report/start session mismatch')
    require(actual == report['provenance'], 'Recorded source provenance no longer matches checked sidecars')
    require(start['source_sha256'].get('weights/conv1_weights.npy') == WEIGHTS_SHA, 'Different programmed weights')
    require(sha(session / 'start_manifest.json') == actual['start_manifest_sha256'], 'Start manifest binding differs')
    if original:
        require(reference.get('execution_grant') is None and reference.get('unknown_prior_tail') is True,
                'Original interrupted tail must remain unknown')
        manifest_identity = dict(start_manifest_sha256=sha(session / 'start_manifest.json'), final_manifest_required=False)
    else:
        final = read_json(session / 'manifest.json')
        require(final['status'] == 'completed' and final['scope'] == 'full', 'New stage session is incomplete')
        for key in ('source', 'scope', 'session_id', 'protocol_sha256', 'spec_sha256', 'source_sha256', 'stage_range', 'acquisition_kind'):
            require(final.get(key) == start.get(key), 'Stage start/final mismatch: ' + key)
        require(final['impacts'] == report['native_impacts'] and final['pulses'] == report['programming_pulses'],
                'Stage acquisition counters differ')
        require(composer.verify_stage_grant(session, start) == reference['execution_grant'], 'Execution grant mismatch')
        require(reference.get('unknown_prior_tail') is False, 'Completed stage has unexpected tail ambiguity')
        manifest_identity = dict(start_manifest_sha256=sha(session / 'start_manifest.json'),
                                 final_manifest_sha256=sha(session / 'manifest.json'))
    return dict(session=reference['acquisition_session'], source_verification=actual['status'],
                private_source_evidence=private, execution_grant=reference.get('execution_grant'),
                manifest_identity=manifest_identity, raw_replayed_here=False)


def assemble(stages, branch):
    """Byte interleave all48 measured stage exports, without numerical correction."""
    require(set(stages) == set(range(48)), 'Cannot assemble partial or duplicate stage coverage')
    output = bytearray(FULL_BYTES)
    for stage in range(48):
        data = stages[stage][branch]
        require(len(data) == STAGE_BYTES and all(math.isfinite(v[0]) for v in struct.iter_unpack('<f', data)),
                'Wrong-sized or nonfinite measured stage payload')
        group, tile = divmod(stage, 3)
        for t in range(12):
            for pair in range(4):
                source = (t * 4 + pair) * 784 * 4
                destination = ((t * 64 + group * 4 + pair) * 784 * 3 + tile) * 4
                for byte in range(4):
                    output[destination + byte:destination + byte + 784 * 12:12] = data[source + byte:source + byte + 784 * 4:4]
    return bytes(output)


def bind_composite(report, report_path):
    validate_composite(report)
    rows = []; heldout = []; retention = []; stages = {}; bindings = []; contexts = []
    impacts = pulses = 0; zero_shift = bracket_duration = 0.; unknown_tail = False
    seen_sessions = set()
    for reference in report['stage_reports']:
        require(reference['acquisition_session'] not in seen_sessions, 'Duplicate acquisition session')
        seen_sessions.add(reference['acquisition_session'])
        path = local(reference['path'])
        require(sha(path) == reference['sha256'], 'Stage audit JSON changed')
        audit = read_json(path); ids = validate_stage_report(audit, reference)
        binding = bind_stage_sidecars(audit, reference)
        bindings.append(dict(report_path=reference['path'], report_sha256=reference['sha256'], **binding))
        for certificate in audit['accepted_stages']:
            stage = certificate['stage']; require(stage not in stages, 'Duplicate physical stage across sessions')
            payloads = {}
            for branch in ('raw', 'calibrated'):
                item = certificate['payloads'][branch]; payload = local(item['path'])
                require(item['shape'] == [12, 4, 28, 28] and item['dtype'] == '<f4'
                        and sha(payload) == item['sha256'] == certificate['payload_sha256'][branch], 'Stage payload binding mismatch')
                payloads[branch] = payload.read_bytes()
            stages[stage] = payloads
        rows.extend(audit['weight_observations']); heldout.extend(audit['heldout']); retention.extend(audit['retention'])
        impacts += audit['native_impacts']; pulses += audit['programming_pulses']
        zero_shift = max(zero_shift, metrics.finite(audit['maximum_shared_zero_shift_v']))
        bracket_duration = max(bracket_duration, metrics.finite(audit['maximum_position_bracket_duration_s']))
        unknown_tail |= reference['unknown_prior_tail']
        contexts.append(dict(stages=ids, session=reference['acquisition_session'],
            strict_numeric_replay=audit['strict_numeric_replay'], portable_condition_policy=audit.get('portable_condition_policy'),
            context='original continuous acquisition prefix' if reference['unknown_prior_tail'] else 'fresh Connector/programming/calibration stage'))
    require(sorted(stages) == list(range(48)) and len(rows) == 4800, 'Full48/4800 evidence is missing')
    require(report['validated_stage_impacts'] == impacts and report['validated_stage_pulses'] == pulses
            and report['unknown_prior_lost_tail_operations'] is unknown_tail, 'Composite total/unknown-tail mismatch')
    for branch in ('raw', 'calibrated'):
        expected = hashlib.sha256(assemble(stages, branch)).hexdigest()
        require(expected == report['payload_sha256'][branch]
                and sha(report_path.parent / (branch + '.f32')) == expected, 'Composite tensor differs from exactly48 measured stage payloads')
    aggregate = dict(weight_observations=rows, heldout=heldout, retention=retention, native_impacts=impacts,
                     programming_pulses=pulses, maximum_shared_zero_shift_v=zero_shift,
                     maximum_position_bracket_duration_s=bracket_duration)
    return aggregate, dict(stage_bindings=bindings, stage_contexts=contexts, raw_replayed_here=False,
        payload_assembly_verified=True, unknown_prior_lost_tail_operations=unknown_tail,
        impact_interpretation='Validated accepted-stage lower bound; lost and intentionally discarded prior tail not counted as zero')


def validate_aggregate(audit):
    rows = audit['weight_observations']; seen = set()
    require(len(rows) == 4800, 'Expected4800 measured coefficient observations')
    for row in rows:
        f, tap = row['filter'], row['tap']
        require(type(f) is int and type(tap) is int and 0 <= f < 64 and 0 <= tap < 75 and (f, tap) not in seen,
                'Duplicate/missing/out-of-range coefficient')
        seen.add((f, tap)); pair = f % 4; bl = tap % 32
        require(row['stage'] == f // 4 * 3 + tap // 32 and row['bl'] == bl
                and row['positive_cell'] == [2 * pair, bl] and row['negative_cell'] == [2 * pair + 1, bl]
                and row['per_coefficient_fitting_used'] is False, 'Coefficient layout or fitting mismatch')
    for key in ('heldout', 'retention'):
        require(len(audit[key]) == 48 and sorted(row['stage'] for row in audit[key]) == list(range(48)), 'Missing/duplicate48-stage metrics')
    for row in audit['retention']:
        require(metrics.finite(row['final_minus_initial_basis_rmse_v']) >= 0 and metrics.finite(row['max_abs_v']) >= 0,
                'Invalid retention RMSE/maxabs')


def compare_reports(sim, hardware, hardware_audit):
    decoded = dict(simulator=metrics.validate_report(sim, 'simulator'), hardware=validate_composite(hardware))
    validate_aggregate(hardware_audit)
    require(sim['artifact_sha256'] == hardware['artifact_sha256'], 'Classifier artifacts differ')
    sr = sorted(sim['native_audit']['weight_observations'], key=lambda r: (r['filter'], r['tap']))
    hr = sorted(hardware_audit['weight_observations'], key=lambda r: (r['filter'], r['tap']))
    require(all(s['trained_weight'] == h['trained_weight'] for s, h in zip(sr, hr)), 'Trained coefficient targets differ')
    cpu = metrics.cpu_compare(decoded['simulator']['cpu_reference'], decoded['hardware']['cpu_reference'])
    sources = {}
    for source, classifier, audit in (('simulator', sim, sim['native_audit']), ('hardware', hardware, hardware_audit)):
        sources[source] = dict(weights=metrics.weight_summary(audit['weight_observations'], source),
            heldout=metrics.heldout_summary(audit['heldout']), retention=audit['retention'],
            conv1_errors=classifier['conv1_errors'], downstream=decoded[source],
            accepted_stage_impacts=audit['native_impacts'], accepted_stage_pulses=audit['programming_pulses'],
            maximum_shared_zero_shift_v=audit['maximum_shared_zero_shift_v'],
            maximum_position_bracket_duration_s=audit['maximum_position_bracket_duration_s'])
    return dict(schema='cifar-original-sim-versus-hardware-continuation-v1',
        status='consistent_reports' if cpu['passed'] else 'review_required',
        status_semantics='Report/source/payload consistency and compatible CPU controls; not successful recognition or hardware equivalence',
        coverage=dict(stages=48, coefficients=4800, tensor_elements=1806336, cases=1),
        case=metrics.CASE, protocol_sha256=metrics.PROTOCOL_SHA, cpu_comparator=cpu, sources=sources,
        full_dataset_accuracy_measured=False, calibrated_matrix_repairs_trained_weights=False,
        limitations=['One fixed example, not test accuracy.',
            'Physical Conv1, then digital BN/LIF and remaining network. Three tile contributions sum before one logical LIF step.',
            'Different nominal source gains: nativeSIM .015V/weight, HW .125V/weight; no calibrated physical equivalence claimed.',
            'Original fullSIM used a continuous array session. HW combines an original prefix with fresh per-stage Connector/programming/calibration sessions.',
            'Recovery qualification used a newly initialized SIM array per stage; it is not a replacement full48 SIM baseline.',
            'Silent/tied spike decoders retain mechanical argmax but have no valid_correct success.',
            'TWD scores are trusted from pinned classifier reports; this adapter does not replay learned alpha or any neural model.',
            'This adapter rechecks reports, sidecars, sources/explicit private-source receipts and payload assembly; no raw replay.'])


def main(argv=None):
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('sim-report', 'sim-session', 'hardware-composite', 'output'):
        parser.add_argument('--' + flag, required=True)
    args = parser.parse_args(argv); target = local(args.output)
    require(target.is_relative_to(ROOT / 'reports') and not target.exists(), 'Choose a new repository report path')
    verify_tools()
    sp, hp = local(args.sim_report), local(args.hardware_composite)
    sim, hw = read_json(sp), read_json(hp)
    metrics.validate_report(sim, 'simulator'); validate_composite(hw)
    sim_binding = metrics.bind_session(sim, local(args.sim_session), 'simulator')
    aggregate, hw_binding = bind_composite(hw, hp)
    result = compare_reports(sim, hw, aggregate)
    result['binding'] = dict(simulator=sim_binding, hardware=hw_binding)
    result['input_reports'] = dict(simulator=dict(path=args.sim_report, sha256=sha(sp)),
                                  hardware=dict(path=args.hardware_composite, sha256=sha(hp)))
    result['adapter_sha256'] = sha(__file__)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(dict(status=result['status'], output=args.output)))
    return 0 if result['status'] == 'consistent_reports' else 3


if __name__ == '__main__':
    raise SystemExit(main())
