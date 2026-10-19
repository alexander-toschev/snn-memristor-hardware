"""Read-only second review of published prefix certificates; no raw numerical replay."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports/2026-10-19/cifar-full-case0-review/hardware-prefix-diagnostic/PREFIX_AUDIT.json'
OUT = ROOT / 'reports/2026-10-19/cifar-resume-review/HARDWARE_PREFIX_SECOND_REVIEW.json'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def main():
    d = json.loads(REPORT.read_text())
    assert d['status'] == 'verified_complete_stage_prefix'
    assert d['source'] == 'hardware' and d['scope'] == 'full'
    assert [x['stage'] for x in d['accepted_stages']] == list(range(13))
    assert d['stop_stage_exclusive'] == 13
    assert not d['complete_full_tensor'] and not d['frozen_full_auditor_pass_claimed']
    expected = {
        'tools/cifar_prefix_replay.py': '0f6bb5743dd8b4218b68701c65749db00874d8e444d6df18feda08f127a39c69',
        'tools/audit_cifar_interrupted_prefix.py': '1c19c6e4b0db4af3892a681035b27123633d2ca7c840f1537928b13e35ec3b00',
    }
    assert d['audit_source_sha256'] == expected
    for name, pin in expected.items():
        assert sha(ROOT / name) == pin
    private = d['provenance']['private_source_evidence']
    assert sha(ROOT / private['receipt_path']) == private['receipt_sha256']
    receipt = private['receipt']
    assert sha(ROOT / receipt['stdout_evidence_path']) == receipt['stdout_evidence_sha256']
    private_names = {r['path'] for r in receipt['records']}
    checked = 0
    for name, pin in d['provenance']['source_sha256'].items():
        if name not in private_names:
            assert sha(ROOT / name) == pin, name
            checked += 1
    session = ROOT / d['session']
    assert sha(session / 'start_manifest.json') == d['provenance']['start_manifest_sha256']
    inv = d['raw_inventory']
    assert len(inv['parts']) == 56 and len(inv['admitted_parts']) == 55
    assert inv['excluded_crc_valid_tail_records'] == 828468
    assert not inv['parts'][-1]['admitted']
    for part in inv['parts']:
        p = session / 'raw' / part['path']
        assert p.stat().st_size == part['bytes'] and sha(p) == part['sha256']
    previous = 0
    for cert in d['accepted_stages']:
        assert digest({k: v for k, v in cert.items() if k != 'certificate_sha256'}) == cert['certificate_sha256']
        assert cert['source_provenance_sha256'] == digest(d['provenance'])
        assert cert['raw_inventory_sha256'] == digest(inv)
        assert cert['task_first'] == previous + 1
        assert cert['task_last'] - previous == cert['native_impacts']
        previous = cert['task_last']
        assert cert['shape'] == [12, 4, 28, 28] and cert['dtype'] == '<f4'
        for branch, payload in cert['payloads'].items():
            p = ROOT / payload['path']
            assert p.stat().st_size == 12 * 4 * 28 * 28 * 4
            assert sha(p) == payload['sha256'] == cert['payload_sha256'][branch]
    assert previous == d['native_impacts'] == 3909496
    assert sum(x['programming_pulses'] for x in d['accepted_stages']) == d['programming_pulses'] == 18800
    assert len(d['weight_observations']) == d['observed_coefficients'] == 1328
    assert d['covered_tensor_elements'] == 13 * 12 * 4 * 28 * 28 == 489216
    exceptions = [x for x in d['numeric_diagnostics'] if x['strict_numeric_replay'] != 'PASS']
    assert [x['stage'] for x in exceptions] == [0, 2, 11]
    for x in exceptions:
        assert x['decision_equivalence'] == 'PASS'
        for sp in x['spectra'].values():
            tol = sp['numpy_rank_tolerance']
            assert sp['margin_factor'] == 16
            assert all(v >= 16 * tol or v <= tol / 16 for v in sp['singular_values'])
    result = dict(schema='cifar-prefix-second-review-v1', status='reuse_qualified',
        source_report_sha256=sha(REPORT), reviewer_tool_sha256=sha(Path(__file__)),
        accepted_stage_ids=list(range(13)), first_fresh_stage=13,
        local_source_files_checked=checked, remote_private_files=sorted(private_names),
        raw_compressed_files_rehashed=56, stage_payloads_rehashed=26,
        numerical_raw_replay_repeated=False, reviewed_auditor_fake_tests='8/8 PASS, 19.548 s',
        strict_numeric_replay=d['strict_numeric_replay'], condition_only_stages=[0, 2, 11],
        native_impacts_lower_bound=3909496, programming_pulses_lower_bound=18800,
        full_hardware_launch_qualified=False, current_board_state_inferred=False,
        limitations=['Second review checks source, certificates, payloads and compressed raw hashes; the actual raw semantic replay is the separately reviewed prefix auditor.',
                    'Private files have fresh remote operator evidence, not local bytes or signed attestation.',
                    'No original final manifest/close proof; incomplete stage13 and truncated tail are not reusable.',
                    'New hardware continuation still requires native SIM abrupt-loss recovery qualification and preflight.'])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
