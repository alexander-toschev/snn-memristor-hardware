"""Offline case0 classifier bridge: independently audited physical Conv1 only.

No Connector imports or acquisition. Historical hardware-import gates remain
unchanged. CPU reference is an explicitly separate control, never missing-data
replacement. One image prediction is not a dataset accuracy measurement.
"""
import argparse
import hashlib
import json
from pathlib import Path, PureWindowsPath
import sys

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / '.deps-classifier', ROOT / '.deps'):
    if directory.is_dir():
        sys.path.insert(0, str(directory))

DIGITAL_SHA = 'a29140edd81334de22b66d93cd2fbbccc68b524d87a02ecd49ef6f36979201b4'
# Frozen after independent review of audit_native.py. A declaration
# inside a recorded manifest never grants itself permission to import data.
APPROVED_AUDITOR_SHA256 = frozenset({'d3cd8bf2cf11512f81aabff0300c5fde391dad240fd9008834150e698910e717'})
SHAPE = (12, 64, 28, 28, 3)
PAYLOAD_BYTES = 12 * 64 * 28 * 28 * 3 * 4


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def local(root, value):
    path = Path(value)
    if path.is_absolute() or PureWindowsPath(value).drive or PureWindowsPath(value).root:
        raise ValueError('Only repository-relative paths are accepted')
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError('Path escapes repository')
    return resolved


def digital():
    if sha(ROOT / 'full_cifar_classifier.py') != DIGITAL_SHA:
        raise ValueError('Pinned digital downstream source changed')
    import full_cifar_classifier
    return full_cifar_classifier


def decode_payload(payload, declared_sha):
    """Only accept the auditor's complete little-endian float32 dense output."""
    import numpy as np
    import torch
    if not isinstance(payload, bytes) or len(payload) != PAYLOAD_BYTES:
        raise ValueError('Incomplete dense Conv1 payload')
    if hashlib.sha256(payload).hexdigest() != declared_sha:
        raise ValueError('Dense Conv1 SHA mismatch')
    partials = np.frombuffer(payload, dtype='<f4').reshape(SHAPE)
    if not np.isfinite(partials).all():
        raise ValueError('Nonfinite measured contribution; no CPU or zero fill')
    # All three tiles are summed once, BEFORE the checkpoint's BN[t]/LIF.
    return torch.from_numpy(partials.astype(np.float32, copy=True)).sum(-1)


def verify_session(session):
    """Check recorded files against this reviewed checkout before native audit."""
    from experiments.cifar_full_case0_20261019 import collector, trace
    start = json.loads((session / 'start_manifest.json').read_text())
    final = json.loads((session / 'manifest.json').read_text())
    if final.get('status') != 'completed' or final.get('scope') != 'full':
        raise ValueError('Only a completed full session can be classified')
    if final.get('source') not in ('hardware', 'simulator'):
        raise ValueError('Unknown physical source')
    for key in ('session_id', 'scope', 'source', 'protocol_sha256', 'source_sha256'):
        if final.get(key) != start.get(key):
            raise ValueError('Start/final provenance disagreement: ' + key)
    protocol = json.loads((session / 'protocol.json').read_text())
    collector.validate_protocol(protocol, final['protocol_sha256'])
    if sha(session / 'protocol.json') != sha(Path(__file__).with_name('protocol.json')):
        raise ValueError('Recorded protocol differs from reviewed checkout')
    auditor_sha = sha(Path(__file__).with_name('audit_native.py'))
    if auditor_sha not in APPROVED_AUDITOR_SHA256:
        raise ValueError('Native auditor is not registered after review')
    required = {p.relative_to(ROOT).as_posix() for p in Path(__file__).parent.glob('*.py')}
    required.update(('weights/conv1_weights.npy', 'weights/cifar100_conv1_fixture.json'))
    source_hashes = final['source_sha256']
    if not required.issubset(source_hashes):
        raise ValueError('Incomplete collector source fingerprint')
    for name, expected in source_hashes.items():
        if sha(local(ROOT, name)) != expected:
            raise ValueError('Acquisition source changed: ' + name)
    # The raw reader checks each chunk/index, including complete expanded stream.
    # Also bind their acquisition-time identities and all saved sidecar files.
    hashes = final['result_sha256']
    names = {p.relative_to(session).as_posix() for p in session.rglob('*')
             if p.is_file() and p.name != 'manifest.json'}
    if set(hashes) != names:
        raise ValueError('Missing or unexpected session evidence file')
    for name, expected in hashes.items():
        if sha(local(session, name)) != expected:
            raise ValueError('Session evidence SHA mismatch: ' + name)
    return final, auditor_sha, trace.records(session / 'raw')


def cpu_control(state, model, image):
    """Golden test: tiled Conv1 -> downstream versus checkpoint's own forward."""
    import torch
    import torch.nn.functional as F
    ref = digital()
    spikes = ref.encode(image)
    with torch.no_grad():
        original = model(spikes[:, None])
        direct = F.conv2d(spikes, state['conv_layers.0.weight'])
        tiled = ref.cpu_tiles(spikes, state).sum(-1)
        direct_result = ref.downstream(direct, state)
        tiled_result = ref.downstream(tiled, state)
    direct_error = max(float((a - b).abs().max()) for a, b in zip(direct_result, original))
    errors = {k: float((a - b).abs().max()) for k, a, b in
              zip(('membrane', 'spikes', 'twd'), tiled_result, original)}
    if direct_error > 1e-6 or errors['spikes'] != 0 or max(errors.values()) > 1e-5:
        raise ValueError('Digital downstream differs from checkpoint forward')
    return tiled, tiled_result, dict(direct_forward_max_abs=direct_error,
                                    tiled_forward_max_abs=errors,
                                    conv1_tiled_vs_direct_max_abs=float((tiled - direct).abs().max()))


def classify_session(session, state, model, case, image):
    import torch
    from experiments.cifar_full_case0_20261019 import audit_native, collector
    manifest, auditor_sha, records = verify_session(session)
    flat_image, weights = collector.load_assets()
    if flat_image != image.flatten().tolist():
        raise ValueError('Collector/classifier image identity differs')
    if not torch.equal(torch.from_numpy(weights.copy()), state['conv_layers.0.weight'].reshape(64, 75)):
        raise ValueError('Programmed weights are not the verified checkpoint Conv1')
    payloads, audit = audit_native.audit(records, flat_image, weights, manifest['protocol_sha256'])
    if (not audit.get('complete_full_tensor') or audit.get('scope') != 'full'
            or audit.get('source') != manifest['source'] or audit.get('observed_coefficients') != 4800
            or audit.get('covered_tensor_elements') != 12 * 64 * 28 * 28 * 3
            or audit.get('native_impacts') != manifest['impacts']
            or audit.get('programming_pulses') != manifest['pulses']
            or not isinstance(payloads, dict) or set(payloads) != {'raw', 'calibrated'}):
        raise ValueError('Independent full native replay did not establish complete acquisition')
    cpu_conv, cpu_result, validation = cpu_control(state, model, image)
    ref = digital()
    outputs = {'cpu_reference': ref.describe(cpu_result)}
    errors = {}
    for name, payload in payloads.items():
        conv = decode_payload(payload, audit['payload_sha256'][name])
        outputs[name] = ref.describe(ref.downstream(conv, state))
        residual = conv - cpu_conv
        errors[name] = dict(conv1_rmse=float(residual.square().mean().sqrt()),
                            conv1_max_abs=float(residual.abs().max()),
                            relative_l2=float(residual.norm() / cpu_conv.norm()))
    return dict(source=manifest['source'] + '_conv1_digital_remaining_network',
                registered_classifier_import_verified=True,
                acquisition_manifest_sha256=sha(session / 'manifest.json'),
                auditor_sha256=auditor_sha, native_audit=audit,
                cpu_validation=validation, conv1_errors=errors, predictions=outputs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--cpu-reference', action='store_true')
    mode.add_argument('--session', help='Repository-relative completed full acquisition directory')
    parser.add_argument('--output', required=True, help='New repository-relative JSON report')
    args = parser.parse_args(argv)
    output = local(ROOT, args.output)
    if output.exists():
        raise ValueError('Preserve prior report; output already exists')
    import torch
    torch.set_num_threads(2)
    ref = digital()
    state, model, case, image = ref.assets(ROOT, 0)
    if args.cpu_reference:
        _, result, validation = cpu_control(state, model, image)
        report = dict(source='cpu_reference_only', cpu_validation=validation,
                      predictions={'cpu_reference': ref.describe(result)})
    else:
        session = local(ROOT, args.session)
        report = classify_session(session, state, model, case, image)
    report.update(schema='full-case0-classification-v1', classifier_sha256=sha(__file__),
                  digital_downstream_sha256=DIGITAL_SHA, artifact_sha256=ref.PINS,
                  case={k: case[k] for k in ('original_test_index', 'fine_label', 'remapped_label', 'image_chw_uint8_sha256')},
                  output_class_fine_labels=ref.LABELS, hardware_acquisition_performed=False,
                  full_dataset_accuracy_measured=False,
                  interpretation='One case0 prediction; physical Conv1 followed by digital BN/LIF and remaining network. Calibration corrects same-state additivity, not trained weight mismatch.')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(output=args.output, source=report['source'],
                         predictions={k: v['predicted_subset_indices'] for k, v in report['predictions'].items()})))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
