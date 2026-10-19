"""Offline strict native CIFAR smoke audit; never opens a Connector.

Run on the acquisition checkout when private hardware source files are needed.
Writes a new report outside the immutable session; smoke cannot be classified.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if (ROOT / '.deps').is_dir():
    sys.path.insert(0, str(ROOT / '.deps'))

EXPERIMENT = 'experiments/cifar_full_case0_20261019'
PROTOCOL_SHA = '2cb1d57689508725f7155bf7b1d0d17fec9d4f6b65782171e2b85481bc9aa558'
FROZEN = {
    'audit_native.py': 'd3cd8bf2cf11512f81aabff0300c5fde391dad240fd9008834150e698910e717',
    'classifier.py': '64ba6e85cda19367f93a0c8294f77b6c9b6f5f26cbf40d188685bc47430053e5',
    'collector.py': '03c6121560ca558904c5722228b4c1a6e7e8eb6d7264b6fb6f18cc393b2e5c21',
    'experiment.py': '5e8647da40961558fd76276d7a5b5eef188e57b77dfb44478a86cb2e376a70cf',
    'plan.py': 'd795d6f820eeacca36e6746746bcfd558850bc5d1bedce32cc969820b3ce94d7',
    'reconstruct.py': '516810cdc7d8a23d18d3487c98bf0de53659bd6f82bc126c43d9481a46ade9b7',
    'trace.py': 'c8093a346b46f2124b389fe9bd7f9d1fea6ac161dce7892df70181b058542d37',
}
HELPERS = {
    'cell_calibration.py': 'eff0b2f72586f6a6659a13fba410c6ad0d564bd95d3b83e24b13cf739121c2e8',
    'read_all.py': 'b30ae2f4dd3cd166dbc5cefb07b558127de2bbdba1f6674c40b1b45f4dd74d5f',
    'native_conversions.py': '2ad2e9c057bb14979b083a6e846e8c417c080a6c5273d848e31f893b7970253e',
    'experiment_hardware.py': '059dc4f827943a9df631631f52e776b57182afc9d36db0c1ad5246427f73548c',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def relative(root, name):
    value = Path(name)
    if value.is_absolute() or PureWindowsPath(name).drive or PureWindowsPath(name).root:
        raise ValueError('Repository-relative path required')
    resolved = (root / value).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError('Path escapes its evidence directory')
    return resolved


def verify_evidence(session):
    start = json.loads((session / 'start_manifest.json').read_text())
    final = json.loads((session / 'manifest.json').read_text())
    if final.get('status') != 'completed' or final.get('scope') != 'smoke':
        raise ValueError('Only completed smoke sessions are accepted')
    if final.get('source') not in ('simulator', 'hardware'):
        raise ValueError('Unknown acquisition source')
    for key in ('session_id', 'scope', 'source', 'protocol_sha256', 'source_sha256'):
        if final.get(key) != start.get(key):
            raise ValueError('Start/final provenance mismatch: ' + key)
    if final['protocol_sha256'] != PROTOCOL_SHA:
        raise ValueError('Unreviewed protocol')
    if {p.name for p in (ROOT / EXPERIMENT).glob('*.py')} != set(FROZEN):
        raise ValueError('Frozen Python inventory changed')
    expected = {EXPERIMENT + '/' + name: pin for name, pin in FROZEN.items()}
    expected.update(HELPERS)
    for name, pin in expected.items():
        if sha(relative(ROOT, name)) != pin:
            raise ValueError('Reviewed source changed: ' + name)
    # These reviewed functions read and hash source/INI files only. They do not
    # import a physical driver, create a Connector or access the board.
    from cell_calibration import check_sources
    from experiments.cifar_full_case0_20261019 import collector, plan
    _, board_snapshot, native_sources = check_sources(final['source'])
    expected.update(native_sources)
    expected[EXPERIMENT + '/protocol.json'] = 'fdb1acd5837b21c133f59f8bd67a2031cb9d411f68f83733d87aa00f2a45b2c9'
    expected['weights/conv1_weights.npy'] = collector.WEIGHTS_SHA
    expected[plan.FIXTURE] = plan.FIXTURE_SHA
    if final['source_sha256'] != expected:
        raise ValueError('Acquisition source inventory or hashes differ from reviewed checkout')
    for name, pin in expected.items():
        if sha(relative(ROOT, name)) != pin:
            raise ValueError('Acquisition source bytes differ: ' + name)
    protocol = json.loads((session / 'protocol.json').read_text())
    collector.validate_protocol(protocol, PROTOCOL_SHA)
    if sha(session / 'protocol.json') != expected[EXPERIMENT + '/protocol.json']:
        raise ValueError('Recorded protocol bytes differ')
    if (session / 'board.ini').read_bytes() != board_snapshot:
        raise ValueError('Recorded board INI differs')
    hashes = final['result_sha256']
    names = {p.relative_to(session).as_posix() for p in session.rglob('*')
             if p.is_file() and p != session / 'manifest.json'}
    if set(hashes) != names:
        raise ValueError('Missing or unexpected session evidence file')
    for name, pin in hashes.items():
        if sha(relative(session, name)) != pin:
            raise ValueError('Result SHA mismatch: ' + name)
    return final


def audit_session(session):
    manifest = verify_evidence(session)
    from experiments.cifar_full_case0_20261019 import collector, audit_native, trace
    image, weights = collector.load_assets()
    counters = Counter()
    models = []
    def records():
        for row in trace.records(session / 'raw'):
            counters['events'] += 1
            counters['event:' + row['event']] += 1
            if row['event'] == 'request':
                command = row['command_si']
                counters['mode:' + str(command['mode_flag'])] += 1
                if command['mode_flag'] == 7 and command['vol']:
                    counters['write_sign:' + str(command['sign'])] += 1
            if row['event'] == 'calibration_frozen':
                models.append(dict(stage=row['stage'], **row['matrix']))
            yield row
    payload, replay = audit_native.audit(records(), image, weights, PROTOCOL_SHA)
    expected = dict(source=manifest['source'], scope='smoke', observed_coefficients=300,
                    covered_tensor_elements=144, complete_full_tensor=False,
                    native_impacts=manifest['impacts'], programming_pulses=manifest['pulses'])
    if payload is not None or any(replay.get(k) != v for k, v in expected.items()):
        raise ValueError('Incomplete smoke replay or unexpected classifier payload')
    if replay['programming_pulses'] > 7200:
        raise ValueError('Smoke write-loop bound exceeded')
    summaries = []
    for stage in range(3):
        rows = [r for r in replay['weight_observations'] if r['stage'] == stage]
        summary = dict(stage=stage, coefficients=len(rows),
                       immediate_within_tolerance=sum(r['immediate_within_tolerance'] for r in rows))
        denominator = sum(r['trained_weight'] ** 2 for r in rows)
        for key in ('immediate_weight', 'post_all_raw_weight', 'post_all_calibrated_weight',
                    'final_raw_weight', 'final_calibrated_weight'):
            residuals = [r[key] - r['trained_weight'] for r in rows]
            summary[key] = dict(rmse=math.sqrt(statistics.mean(v*v for v in residuals)),
                                max_abs=max(map(abs, residuals)),
                                relative_l2=math.sqrt(sum(v*v for v in residuals) / denominator))
        summaries.append(summary)
    return dict(schema='strict-native-cifar-smoke-audit-v1', status='pass',
                source=manifest['source'], session_id=manifest['session_id'],
                acquisition_manifest_sha256=sha(session / 'manifest.json'),
                source_files_verified=len(manifest['source_sha256']),
                result_files_verified=len(manifest['result_sha256']),
                source_sha256=manifest['source_sha256'], provenance=manifest['invocation'],
                counters=dict(counters), models=models, replay=replay,
                weight_summary_by_stage=summaries, hardware_access_performed=False,
                classifier_payload_produced=False, full_hardware_qualified=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    session, output = relative(ROOT, args.session), relative(ROOT, args.output)
    if not session.is_relative_to(ROOT / 'results/cifar_full_case0_20261019'):
        raise ValueError('Expected a native CIFAR acquisition session')
    if not output.is_relative_to(ROOT / 'reports') or output.exists():
        raise ValueError('Use a new output under reports; immutable sessions stay unchanged')
    try:
        report = audit_session(session)
    except Exception as exc:
        report = dict(schema='strict-native-cifar-smoke-audit-v1', status='failed',
                      error=type(exc).__name__ + ': ' + str(exc), hardware_access_performed=False,
                      classifier_payload_produced=False, full_hardware_qualified=False)
    report.update(audit_tool_sha256=sha(__file__), session=args.session)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=report['status'], output=args.output,
                         error=report.get('error'), impacts=report.get('replay', {}).get('native_impacts'))))
    return 0 if report['status'] == 'pass' else 2


if __name__ == '__main__':
    raise SystemExit(main())
