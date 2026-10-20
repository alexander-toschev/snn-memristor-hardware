"""Frozen-physics continuation contract. Registration stays closed until review."""
import hashlib
import json
from pathlib import Path, PureWindowsPath

ROOT=Path(__file__).resolve().parents[2]
from tools import runtime_provenance as provenance_policy
PHYSICAL_PROTOCOL_SHA='2cb1d57689508725f7155bf7b1d0d17fec9d4f6b65782171e2b85481bc9aa558'
PINS={
 'experiments/cifar_full_case0_20261019/audit_native.py':'d3cd8bf2cf11512f81aabff0300c5fde391dad240fd9008834150e698910e717',
 'experiments/cifar_full_case0_20261019/classifier.py':'64ba6e85cda19367f93a0c8294f77b6c9b6f5f26cbf40d188685bc47430053e5',
 'experiments/cifar_full_case0_20261019/experiment.py':'5e8647da40961558fd76276d7a5b5eef188e57b77dfb44478a86cb2e376a70cf',
 'experiments/cifar_full_case0_20261019/trace.py':'c8093a346b46f2124b389fe9bd7f9d1fea6ac161dce7892df70181b058542d37',
 'experiments/cifar_full_case0_20261019/collector.py':'03c6121560ca558904c5722228b4c1a6e7e8eb6d7264b6fb6f18cc393b2e5c21',
 'experiments/cifar_full_case0_20261019/plan.py':'d795d6f820eeacca36e6746746bcfd558850bc5d1bedce32cc969820b3ce94d7',
 'experiments/cifar_full_case0_20261019/reconstruct.py':'516810cdc7d8a23d18d3487c98bf0de53659bd6f82bc126c43d9481a46ade9b7',
 'cell_calibration.py':'eff0b2f72586f6a6659a13fba410c6ad0d564bd95d3b83e24b13cf739121c2e8',
 'read_all.py':'b30ae2f4dd3cd166dbc5cefb07b558127de2bbdba1f6674c40b1b45f4dd74d5f',
 'native_conversions.py':'2ad2e9c057bb14979b083a6e846e8c417c080a6c5273d848e31f893b7970253e',
 'experiment_hardware.py':'059dc4f827943a9df631631f52e776b57182afc9d36db0c1ad5246427f73548c',
}
SPEC=dict(schema='cifar-resumable-v1',physical_protocol_sha256=PHYSICAL_PROTOCOL_SHA,
    scope='full',positions_per_stage=784,stages=48,checkpoint_positions=16,
    checkpoint_coefficients=16,raw_expanded_stage_limit_bytes=2*1024**3,
    checkpoint_reuse='whole independently audited stage only',
    finite_batch_stop='Explicit stop_stage_exclusive up to target48; completed batch is not full classifier coverage',
    new_stage='fresh Connector, programming, R, basis, C, heldout, all784 positions and final basis',
    resume_inside_stage_allowed=False,automatic_retry=False,
    native_sim_qualification=dict(stages=[0,1,2],interrupted_stage=1,interrupt_after_positions=17,
                                  fault='SIM-only os._exit86 before closing position17 chunk; durableindex16',repeat_interrupted_stage_in_full=True),
    hardware_duration_s=None,physical_io='public MemriBoard Connector/native converters/INI only')
GRANTS={source:'reports/2026-10-20/runtime-provenance-v2/'+source.upper()+'_EXECUTION_GRANT.json' for source in ('simulator','hardware')}
APPROVED_PREFIX_AUDITOR_SHA256=frozenset({'1c19c6e4b0db4af3892a681035b27123633d2ca7c840f1537928b13e35ec3b00'})
PREFIX_REPLAY_SHA='0f6bb5743dd8b4218b68701c65749db00874d8e444d6df18feda08f127a39c69'
STAGE_REGISTRY='tools/cifar_stage_source_registry_v2.json'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024**2),b''):h.update(chunk)
    return h.hexdigest()


def local(value,root=ROOT):
    p=Path(value)
    if p.is_absolute() or PureWindowsPath(value).drive or PureWindowsPath(value).root:raise ValueError('Repository-relative path required')
    p=(root/p).resolve()
    if not p.is_relative_to(root.resolve()):raise ValueError('Path escapes repository')
    return p


def verify_pins():
    for path,pin in PINS.items():
        if sha(local(path))!=pin:raise ValueError('Reviewed physical helper changed: '+path)
    return dict(PINS)


def execution_grant(source, historical_path=None):
    """Separate immutable reviewed grant per source; never edit a completed SIM grant."""
    grant_name=historical_path or GRANTS[source]
    path=local(grant_name)
    if historical_path is not None:
        approved=provenance_policy.versions(ROOT)['historical_grants']
        if approved.get(grant_name)!=sha(path):raise ValueError('Unreviewed historical grant')
    if not path.exists():return None
    grant=json.loads(path.read_text())
    if grant.get('schema')!='cifar-resumable-execution-grant-v1' or grant.get('registration')!='approved_after_review' or grant.get('source')!=source:
        raise ValueError('Invalid explicit source execution grant')
    if grant.get('spec_sha256')!=digest(SPEC) or grant.get('physical_protocol_sha256')!=PHYSICAL_PROTOCOL_SHA:
        raise ValueError('Grant refers to another protocol')
    names={p.relative_to(ROOT).as_posix() for p in Path(__file__).parent.glob('*.py')}
    names.update(('tools/plan_cifar_continuation.py','tools/classify_cifar_continuation.py',
                  'tools/audit_cifar_interrupted_prefix.py','tools/cifar_prefix_replay.py',
                  provenance_policy.OLD_REGISTRY if historical_path else STAGE_REGISTRY))
    if historical_path is None:names.update(('tools/runtime_provenance.py',provenance_policy.REGISTRATION))
    if set(grant.get('source_sha256',{}))!=names:raise ValueError('Execution grant source inventory incomplete')
    for name,pin in grant['source_sha256'].items():
        if historical_path:provenance_policy.verify_source(name,pin,ROOT)
        elif sha(local(name))!=pin:raise ValueError('Execution grant source changed: '+name)
    if source=='hardware':
        ref=grant.get('native_sim_qualification',{});report_path=local(ref['path'])
        if sha(report_path)!=ref['sha256']:raise ValueError('Qualifying nativeSIM report changed')
        report=json.loads(report_path.read_text())
        if report.get('status')!='qualification_pass' or report.get('source')!='simulator' or report.get('accepted_stage_ids')!=[0,1,2] or report.get('classifier_executed') is not False:
            raise ValueError('Missing complete nativeSIM recovery qualification')
        if report.get('analyzer_sha256')!=provenance_policy.versions(ROOT)['historical_sources']['tools/classify_cifar_continuation.py'] or report.get('spec_sha256')!=digest(SPEC):
            raise ValueError('Qualification analyzer/protocol differs')
        fault=report.get('controlled_interruption',{})
        if fault.get('durable_completed_positions')!=16 or fault.get('incomplete_tail_verified') is not True or fault.get('failed_stage_payload_used') is not False:
            raise ValueError('Missing abrupt-process-loss evidence')
        for stage_report in report['stage_reports']:
            if sha(local(stage_report['path']))!=stage_report['sha256']:raise ValueError('Qualified stage report changed')
    elif grant.get('native_sim_qualification') is not None:
        raise ValueError('SIM acquisition grant must not claim existing native qualification')
    return dict(path=grant_name,sha256=sha(path),document=grant)


def validate_plan(plan,source):
    if plan.get('schema')!='cifar-continuation-plan-v1' or plan.get('source')!=source:raise ValueError('Wrong continuation plan/source')
    if plan.get('spec_sha256')!=digest(SPEC) or plan.get('physical_protocol_sha256')!=PHYSICAL_PROTOCOL_SHA:raise ValueError('Wrong continuation protocol')
    first,stop=plan.get('first_stage'),plan.get('stop_stage_exclusive')
    if type(first) is not int or type(stop) is not int or not 0<=first<stop<=48:raise ValueError('Invalid finite stage range')
    if plan.get('purpose')=='qualification':
        if source!='simulator' or stop!=3 or plan.get('target_stop_stage_exclusive')!=3:raise ValueError('Qualification only SIM stages0..2')
    elif plan.get('purpose')!='full' or plan.get('target_stop_stage_exclusive')!=48:raise ValueError('Invalid acquisition purpose/target')
    if plan.get('accepted_stage_ids')!=list(range(first)):raise ValueError('Only continuous audited complete prefix can be reused')
    refs=plan.get('prefix_reports')
    if not isinstance(refs,list) or (first>0 and not refs):raise ValueError('Missing prefix provenance')
    accepted=[]
    for ref in refs:
        path=local(ref['path'])
        if sha(path)!=ref['sha256']:raise ValueError('Prefix report changed')
        report=json.loads(path.read_text())
        if report.get('schema')!='cifar-stage-prefix-v1' or report.get('scope')!='full' or report.get('source')!=source or report.get('protocol_sha256')!=PHYSICAL_PROTOCOL_SHA:
            raise ValueError('Incompatible prefix report')
        if report.get('status') not in ('verified_complete_stage_prefix','completed_range') or report.get('complete_full_tensor') is not False:
            raise ValueError('Failed diagnostic is not an accepted stage certificate')
        provenance_policy.audit_sources(report,ROOT)
        for cert in report['accepted_stages']:
            body={k:v for k,v in cert.items() if k!='certificate_sha256'}
            if digest(body)!=cert['certificate_sha256']:raise ValueError('Stage certificate changed')
            accepted.append(cert['stage'])
    if sorted(accepted)!=list(range(first)) or len(set(accepted))!=len(accepted):raise ValueError('Duplicate/missing accepted stage')
    return first,stop
