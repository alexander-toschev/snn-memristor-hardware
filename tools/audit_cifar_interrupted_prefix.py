"""Offline preservation and independent replay of complete CIFAR stage prefixes.

Never repairs an original gzip, creates a session footer, opens Connector, or
promotes a partial/smoke tensor to a completed classification. No pickle inputs.
"""
import argparse
from configparser import ConfigParser
from datetime import datetime
import gzip
import hashlib
import json
from pathlib import Path, PureWindowsPath
import sys
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / '.deps'), str(ROOT)]
EXP = 'experiments/cifar_full_case0_20261019'
PROTOCOL_SHA = '2cb1d57689508725f7155bf7b1d0d17fec9d4f6b65782171e2b85481bc9aa558'
FROZEN = {
    'audit_native.py': 'd3cd8bf2cf11512f81aabff0300c5fde391dad240fd9008834150e698910e717',
    'classifier.py': '64ba6e85cda19367f93a0c8294f77b6c9b6f5f26cbf40d188685bc47430053e5',
    'collector.py': '03c6121560ca558904c5722228b4c1a6e7e8eb6d7264b6fb6f18cc393b2e5c21',
    'experiment.py': '5e8647da40961558fd76276d7a5b5eef188e57b77dfb44478a86cb2e376a70cf',
    'plan.py': 'd795d6f820eeacca36e6746746bcfd558850bc5d1bedce32cc969820b3ce94d7',
    'reconstruct.py': '516810cdc7d8a23d18d3487c98bf0de53659bd6f82bc126c43d9481a46ade9b7',
    'trace.py': 'c8093a346b46f2124b389fe9bd7f9d1fea6ac161dce7892df70181b058542d37',
    'protocol.json': 'fdb1acd5837b21c133f59f8bd67a2031cb9d411f68f83733d87aa00f2a45b2c9',
}
HELPERS = {
    'cell_calibration.py': 'eff0b2f72586f6a6659a13fba410c6ad0d564bd95d3b83e24b13cf739121c2e8',
    'read_all.py': 'b30ae2f4dd3cd166dbc5cefb07b558127de2bbdba1f6674c40b1b45f4dd74d5f',
    'native_conversions.py': '2ad2e9c057bb14979b083a6e846e8c417c080a6c5273d848e31f893b7970253e',
    'experiment_hardware.py': '059dc4f827943a9df631631f52e776b57182afc9d36db0c1ad5246427f73548c',
}
PRIVATE = {
    'vendor/MemriBoard/MemriCORE/rp5_fpga_python/rpi_FPGAed.py': ('bc717659fa5d4d36d2037eef978bf1cd8dcf0fe92f9bbfad51f1eb1d0d27df9c', 8844),
    'vendor/MemriBoard/MemriCORE/rp5_fpga_python/FPGA_SIM.py': ('5bcb02c04bc185e332b2ec96acabc2d4303d21724c9c052de540ef92d603b634', 4950),
}
# Kept separate to avoid a hash cycle: contract pins this auditor, then root
# freezes the completed contract/source hashes in an independently reviewed
# registration file. No manifest can supply or override this registry.
STAGE_REGISTRY = 'tools/cifar_stage_source_registry.json'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b''):
            value.update(block)
    return value.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def local(root, value):
    path, windows = Path(value), PureWindowsPath(value)
    require(bool(value) and not path.is_absolute() and not windows.drive and not windows.root,
            'Repository-relative path required')
    resolved = (root / path).resolve()
    require(resolved.is_relative_to(root.resolve()), 'Path escapes evidence root')
    return resolved


def decoded_lines(path):
    """Full consumption checks gzip CRC/footer; never admits a recovered tail."""
    with gzip.open(path, 'rb') as stream:
        for line in stream:
            require(line.endswith(b'\n'), 'Non-terminated JSONL line')
            row = json.loads(line)
            require(isinstance(row, dict), 'JSONL record must be an object')
            yield line, row


def inventory(raw):
    """Hash every original part, admit only the continuous CRC-valid sequence."""
    parts = sorted(raw.glob('part-*.jsonl.gz'))
    require(bool(parts), 'No original raw parts')
    result = []
    admitted = []
    stopped = False
    total_records = boundary_records = 0
    boundary_stage = None
    for index, path in enumerate(parts):
        item = dict(path=path.name, bytes=path.stat().st_size, sha256=sha(path), admitted=False)
        result.append(item)
        if stopped:
            item['reason'] = 'after_first_unusable_part'
            continue
        if path.name != f'part-{index:06d}.jsonl.gz':
            item['reason'] = 'noncontiguous_part_number'; stopped = True
            continue
        expanded = hashlib.sha256(); count = 0; last_boundary = None
        try:
            for line, row in decoded_lines(path):
                expanded.update(line); count += 1
                if row.get('event') == 'stage_complete':
                    last_boundary = (total_records + count, row.get('stage'))
        except (EOFError, OSError, ValueError, UnicodeError, zlib.error) as exc:
            item['reason'] = type(exc).__name__ + ': ' + str(exc)
            stopped = True
            continue
        item.update(admitted=True, records=count, expanded_sha256=expanded.hexdigest())
        admitted.append(path.name); total_records += count
        if last_boundary is not None:
            boundary_records, boundary_stage = last_boundary
    return dict(parts=result, admitted_parts=admitted, crc_valid_records=total_records,
                prefix_records=boundary_records, last_candidate_stage=boundary_stage,
                excluded_crc_valid_tail_records=total_records-boundary_records,
                unknown_buffered_or_truncated_tail_operations=True,
                count_interpretation='Verified replay counts are lower bounds, not exact actual hardware totals')


def prefix_records(raw, inv):
    remaining = inv['prefix_records']
    for name in inv['admitted_parts']:
        path = raw / name
        expected = next(p for p in inv['parts'] if p['path'] == name)
        require(sha(path) == expected['sha256'], 'Raw changed between inventory and replay')
        for _, row in decoded_lines(path):
            if remaining == 0:
                return
            remaining -= 1
            yield row
    require(remaining == 0, 'Inventoried prefix disappeared')


def verify_private_receipt(path, session, recorded):
    receipt=json.loads(path.read_text())
    require(receipt.get('schema')=='cifar-private-source-operator-receipt-v1', 'Unknown private receipt schema')
    require(receipt.get('acquisition_session')==session.relative_to(ROOT).as_posix(), 'Private receipt belongs to another acquisition')
    require(receipt.get('verification')=='remote_operator_verified' and receipt.get('operator')=='root', 'Root-observed receipt required')
    require(receipt.get('local_private_bytes_available') is False and receipt.get('cryptographic_remote_attestation') is False,
            'Receipt must disclose its evidential limits')
    require(bool(receipt.get('host')) and len(receipt.get('git_head',''))==40, 'Incomplete remote observation identity')
    require(datetime.fromisoformat(receipt['observed_utc']).tzinfo is not None, 'Timezone missing in receipt')
    evidence_path=local(ROOT,receipt['stdout_evidence_path'])
    require(sha(evidence_path)==receipt['stdout_evidence_sha256'], 'Observed stdout evidence SHA differs')
    evidence=json.loads(evidence_path.read_text())
    require(evidence.get('evidence_type')=='operator_transcription_of_visible_terminal_stdout', 'Unsupported evidence representation')
    require(evidence['records'][0]=={k:receipt[k] for k in ('observed_utc','host','git_head')}, 'Receipt/stdout identity mismatch')
    records=receipt['records']
    require(len(records)==2 and {r['path'] for r in records}==set(PRIVATE), 'Exactly the two approved private files required')
    require(evidence['records'][1:]==records, 'Receipt/stdout file records mismatch')
    for record in records:
        name=record['path'];pin,size=PRIVATE[name]
        require(recorded.get(name)==record['observed_sha256']==record['expected_sha256']==pin, 'Private source SHA differs')
        require(record['size_bytes']==size and record['resolved_path'].endswith('/'+name), 'Private source size/path differs')
    require(bool(receipt.get('limitations')), 'Receipt limitations must be retained')
    return dict(status='remote_operator_verified_local_bytes_unavailable',receipt_path=path.relative_to(ROOT).as_posix(),
                receipt_sha256=sha(path),receipt=receipt, strict_all_source_local_bytes_pass=False)


def verify_sources(session, private_receipt=None, stage_session=False):
    start = json.loads((session / 'start_manifest.json').read_text())
    require(start.get('source') in ('hardware', 'simulator'), 'Unknown source')
    require(start.get('scope') in ('full', 'smoke'), 'Unknown scope')
    require(start.get('session_id') == session.name, 'Session directory identity mismatch')
    require(start.get('protocol_sha256') == PROTOCOL_SHA, 'Unreviewed protocol')
    required = {EXP + '/' + name: pin for name, pin in FROZEN.items()}
    required.update(HELPERS)
    for name, pin in required.items():
        require(sha(local(ROOT, name)) == pin, 'Reviewed source changed: ' + name)
    private_proof=None
    if private_receipt is None:
        # Read-only source/INI check, no Connector instantiated or imported.
        from cell_calibration import check_sources
        _, ini, runtime = check_sources(start['source'])
    else:
        require(start['source']=='hardware', 'Remote private receipt applies only to hardware evidence')
        private_proof=verify_private_receipt(private_receipt,session,start['source_sha256'])
        from read_all import INI, EXPECTED_BOARD, EXPECTED_SOURCES, FPGA_SOURCE, FPGA_SHA256
        ini=(ROOT/INI).read_bytes();config=ConfigParser();config.read_string(ini.decode('utf-8-sig'))
        require(config.get('board','board_type')=='rp5_fpga_python', 'Wrong INI backend')
        require(all(config.getfloat('board',k)==v for k,v in EXPECTED_BOARD.items()), 'INI board setting changed')
        runtime={'vendor/MemriBoard/'+name:pin for name,pin in {**EXPECTED_SOURCES,FPGA_SOURCE:FPGA_SHA256}.items()}
        runtime[INI]=sha(ROOT/INI)
    required.update(runtime)
    required.update({'weights/conv1_weights.npy': '8be312b9eafb9de23f8340b17f848e7e0e2e7884293c826a6d9f6cbb5125d440',
                     'weights/cifar100_conv1_fixture.json': '8cb9ec0fe30242328527073bb076f8019295d73a748343f58fc2a8671bf7c154'})
    recorded = start['source_sha256']
    stage_registry=None
    if stage_session:
        registry_path=ROOT/STAGE_REGISTRY
        require(registry_path.is_file(), 'Completed-stage source registration is closed pending review')
        stage_registry=json.loads(registry_path.read_text())
        require(stage_registry.get('schema')=='cifar-reviewed-stage-source-registry-v1' and stage_registry.get('registration')=='approved_after_review',
                'Stage source registry is not approved after review')
        require(stage_registry.get('physical_protocol_sha256')==PROTOCOL_SHA and stage_registry.get('spec_sha256')==start.get('spec_sha256'),
                'Stage specification is not registered')
        names={p.relative_to(ROOT).as_posix() for p in (ROOT/'experiments/cifar_resumable_20261019').glob('*.py')}
        require({'experiments/cifar_resumable_20261019/'+name for name in ('collector.py','contract.py','durable.py','experiment.py')}.issubset(names),
                'Incomplete local resumed route')
        require(set(stage_registry['source_sha256'])==names, 'Stage registry source set differs')
        required.update(stage_registry['source_sha256'])
    require(all(recorded.get(k) == v for k, v in required.items()), 'Missing/mismatched required source pin')
    for name, pin in recorded.items():
        if private_proof is not None and name in PRIVATE and not local(ROOT,name).exists():
            continue
        require(sha(local(ROOT, name)) == pin, 'Acquisition source bytes missing or changed: ' + name)
    require((session / 'board.ini').read_bytes() == ini, 'Recorded board INI mismatch')
    require(sha(session / 'protocol.json') == FROZEN['protocol.json'], 'Recorded protocol bytes mismatch')
    return start, dict(status='public_local_private_remote_operator_verified' if private_proof else 'all_recorded_source_bytes_verified_locally',
                       private_source_evidence=private_proof, source_sha256=recorded,
                       stage_source_registry=stage_registry,
                       stage_source_registry_sha256=sha(ROOT/STAGE_REGISTRY) if stage_registry else None,
                       start_manifest_sha256=sha(session / 'start_manifest.json'),
                       board_ini_sha256=sha(session / 'board.ini'), protocol_file_sha256=sha(session / 'protocol.json'))


def audit_session(session, output, portable_condition=False, private_receipt=None, stage_session=False):
    require(not output.exists(), 'Preserve previous report: output directory exists')
    start, provenance = verify_sources(session,private_receipt,stage_session)
    inv = inventory(session / 'raw')
    require(inv['prefix_records'] > 0, 'No complete CRC-backed stage candidate')
    from experiments.cifar_full_case0_20261019.collector import load_assets
    from tools.cifar_prefix_replay import audit_range
    image, weights = load_assets()
    if stage_session:
        final=json.loads((session/'manifest.json').read_text())
        require(final.get('status')=='completed', 'Only completed new stage sessions accepted')
        index=json.loads((session/'raw/index.json').read_text())
        require(index.get('stream_finalized') is True, 'Durable stage stream not finalized')
        for key in ('source','scope','session_id','protocol_sha256','source_sha256','stage_range','acquisition_kind',
                    'spec_sha256','continuation_plan_sha256','launcher_sha256'):
            require(final.get(key)==start.get(key), 'Stage start/final differs: '+key)
        require(start.get('acquisition_kind')=='stage_segment' and len(start.get('stage_range',[]))==2, 'Literal stage session identity required')
        require(start.get('launcher_sha256')==start['source_sha256'].get('experiments/cifar_resumable_20261019/experiment.py'),
                'Stage launcher SHA differs from registered source')
        require(sha(session/'continuation_plan.json')==start.get('continuation_plan_sha256'), 'Stage continuation plan SHA differs')
        names={p.relative_to(session).as_posix() for p in session.rglob('*') if p.is_file() and p.name!='manifest.json'}
        require(set(final['result_sha256'])==names, 'Completed-stage result inventory differs')
        for name,pin in final['result_sha256'].items():require(sha(local(session,name))==pin,'Stage evidence SHA differs: '+name)
        from experiments.cifar_full_case0_20261019.trace import records
        payloads,report=audit_range(records(session/'raw'),image,weights,PROTOCOL_SHA,
            first_stage=start['stage_range'][0],stop_stage=start['stage_range'][1],portable_condition=portable_condition)
        require(report['native_impacts']==final['impacts'] and report['programming_pulses']==final['pulses'],'Stage manifest counters differ')
        inv['unknown_buffered_or_truncated_tail_operations']=False
        inv['count_interpretation']='Completed stage stream and final manifest counts independently matched'
    else:
        payloads, report = audit_range(prefix_records(session / 'raw', inv), image, weights, PROTOCOL_SHA,
                                       stop_stage=48 if start['scope'] == 'full' else 3, allow_prefix_eof=True,
                                       portable_condition=portable_condition)
    require(report['source'] == start['source'] and report['scope'] == start['scope'], 'Raw/manifest source mismatch')
    require(report['stop_stage_exclusive'] == inv['last_candidate_stage'] + 1, 'Prefix boundary disagreement')
    report.update(session=session.relative_to(ROOT).as_posix(), session_id=start['session_id'],
                  provenance=provenance, raw_inventory=inv, original_final_manifest_present=(session / 'manifest.json').exists(),
                  original_raw_index_present=(session / 'raw/index.json').exists(),
                  audit_source_sha256={name: sha(ROOT / name) for name in
                    ('tools/cifar_prefix_replay.py', 'tools/audit_cifar_interrupted_prefix.py')},
                  physical_state_now_inferred=False, hardware_acquisition_performed=False,
                  incomplete_tail_not_reused=True, unknown_actual_tail_operations=not stage_session)
    if stage_session:
        report['audit_source_sha256'][STAGE_REGISTRY]=provenance['stage_source_registry_sha256']
    output.mkdir(parents=True)
    for stage in report['accepted_stages']:
        stage['session_id'] = start['session_id']
        stage['source'] = start['source']; stage['scope'] = start['scope']
        stage['protocol_sha256'] = PROTOCOL_SHA
        stage['source_provenance_sha256'] = digest(provenance)
        stage['raw_inventory_sha256'] = digest(inv)
        stage['payloads'] = {}
        for label, data in payloads[stage['stage']].items():
            path = output / f"stage-{stage['stage']:02d}-{label}.f32"
            path.write_bytes(data)
            stage['payloads'][label] = dict(path=path.relative_to(ROOT).as_posix(), sha256=sha(path),
                                           shape=stage['shape'], dtype=stage['dtype'])
        stage['certificate_sha256'] = digest(stage)
    (output / 'PREFIX_AUDIT.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--inventory-only', action='store_true', help='Preservation only; grants no reusable stage certificate')
    parser.add_argument('--portable-condition', action='store_true', help='Reviewed rank-deficient excluded-condition policy; records strict numeric failure separately')
    parser.add_argument('--private-source-receipt', help='Explicit root-observed proof for the two allowlisted private files only')
    parser.add_argument('--stage-session', action='store_true', help='Audit a completed literal one-stage resumed session, not an interrupted prefix')
    args = parser.parse_args(argv)
    session, output = local(ROOT, args.session), local(ROOT, args.output_dir)
    if args.inventory_only:
        require(not output.exists(), 'Output exists')
        report = dict(schema='cifar-interrupted-preservation-v1', reusable_stages=[],
                      semantic_audit_performed=False, raw_inventory=inventory(session / 'raw'))
        output.mkdir(parents=True)
        (output / 'INVENTORY.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    else:
        report = audit_session(session, output,args.portable_condition,
                               local(ROOT,args.private_source_receipt) if args.private_source_receipt else None,args.stage_session)
    print(json.dumps(dict(output_dir=args.output_dir, accepted_stages=[v['stage'] for v in report.get('accepted_stages', [])],
                          complete_full_tensor=False)))


if __name__ == '__main__':
    main()
