"""Supplemental hardware launcher attestation plus unchanged full native-STDP replay."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
LAUNCHER=ROOT/'experiments/conductance_stdp_hardware_20261019/experiment.py'
OLD_AUDITOR=ROOT/'tools/audit_conductance_stdp_calibrated.py'
OLD_AUDITOR_SHA='f1fab366818f554e90dea185c0a23073074f0e2984bd5af3c15ff415cac8bcfa'
HARDWARE_FREEZE=ROOT/'reports/2026-10-19/conductance-stdp-hardware-review/SOURCE_FREEZE.json'
APPROVED_LAUNCHER_SHA256=frozenset({'cef6db7a176d2264c12be077852d56a0dc62bc7e696f351bc49ebfeb328ae5ed'})


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def require(condition,message):
    if not condition:raise ValueError(message)


def compare_start_final(start,manifest):
    fields=('session_id','source','source_sha256','protocol_sha256','launcher_sha256',
            'launcher_source_sha256','launcher_spec','launcher_spec_sha256','qualification',
            'qualifying_sim_audit_sha256','invocation','physical_pulse_duration_s','seed','environment')
    for field in fields:require(start.get(field)==manifest.get(field) and field in start,'Start/final mismatch: '+field)


def launcher_attestation(session):
    launcher_sha=sha(LAUNCHER)
    require(launcher_sha in APPROVED_LAUNCHER_SHA256,'Hardware launcher has not been independently registered')
    require(sha(OLD_AUDITOR)==OLD_AUDITOR_SHA,'Frozen native replay auditor changed')
    manifest=json.loads((session/'manifest.json').read_text(encoding='utf-8'))
    start=json.loads((session/'start_manifest.json').read_text(encoding='utf-8'))
    require(manifest.get('source')=='hardware','Supplemental audit accepts hardware only')
    require(manifest.get('status') in ('completed','uninformative_calibration'),'Valid terminal hardware outcome required')
    compare_start_final(start,manifest)
    require(manifest['launcher_sha256']==launcher_sha,'Launcher bytes mismatch')
    expected={p.relative_to(ROOT).as_posix():sha(p) for p in LAUNCHER.parent.iterdir()
              if p.is_file() and p.suffix in ('.py','.md')}
    require(manifest['launcher_source_sha256']==expected,'Launcher inventory/bytes mismatch')
    sys.path[:0]=[str(ROOT/'.deps'),str(ROOT)]
    from experiments.conductance_stdp_hardware_20261019 import experiment as launcher
    require(Path(launcher.__file__).resolve()==LAUNCHER,'Wrong launcher import')
    require(manifest['launcher_spec']==launcher.LAUNCH_SPEC,'Wrong reviewed launcher specification')
    spec_sha=launcher.digest(launcher.LAUNCH_SPEC)
    require(spec_sha==manifest['launcher_spec_sha256'] and spec_sha in launcher.APPROVED_LAUNCH_SPEC_SHA256,'Unregistered launcher protocol')
    freeze,qualification=launcher.qualification()  # File-only; checks actual SIM artifacts and all qualification refs.
    require(manifest['qualification']==qualification,'Hardware qualification references changed')
    require(manifest['qualifying_sim_audit_sha256']==launcher.SIM_AUDIT_SHA,'Wrong qualifying SIM audit')
    require(manifest['protocol_sha256']==launcher.PROTOCOL_SHA==freeze['protocol_sha256'],'Wrong acquisition protocol')
    require(manifest['physical_pulse_duration_s'] is None,'Unknown hardware duration cannot become claimed known')
    reviewed=json.loads(HARDWARE_FREEZE.read_text(encoding='utf-8'))
    expected_freeze=dict(launcher_sha256=launcher_sha,launcher_source_sha256=expected,
                        supplemental_auditor_sha256=sha(__file__),launcher_spec_sha256=spec_sha,
                        protocol_sha256=launcher.PROTOCOL_SHA,qualification_reference_sha256=qualification['reference_sha256'])
    for key,value in expected_freeze.items():require(reviewed.get(key)==value,'Hardware review freeze mismatch: '+key)
    argv=manifest['invocation']['hardware_argv']
    require(argv==['--run-hardware'],'Unreviewed hardware invocation arguments')
    process=manifest['invocation']['process_argv']
    require(len(process)==2 and process[1]=='--run-hardware','Wrong physical launch command')
    require(Path(process[0]).as_posix().endswith('experiments/conductance_stdp_hardware_20261019/experiment.py'),'Wrong hardware entrypoint')
    return dict(launcher_sha256=launcher_sha,launcher_spec_sha256=spec_sha,launcher_source_sha256=expected,
                qualification=qualification,source_protocol_sha256=manifest['protocol_sha256'])


def audit(session):
    session=Path(session).resolve()
    require(session.is_relative_to(ROOT/'results/conductance_stdp_hardware_20261019'),'Session outside hardware result namespace')
    attestation=launcher_attestation(session)
    from tools import audit_conductance_stdp_calibrated as native
    require(Path(native.__file__).resolve()==OLD_AUDITOR and sha(native.__file__)==OLD_AUDITOR_SHA,'Wrong native replay import')
    replay=native.audit(session)  # Exact old checker; no patched source inventories, functions or gates.
    return dict(status='pass',session=session.relative_to(ROOT).as_posix(),auditor_sha256=sha(__file__),
                launcher_attestation=attestation,native_replay=replay,
                physical_learning_improvement_proven=False,classification_accuracy=None,
                interpretation='Hardware launcher and full measured raw protocol replay passed; efficacy remains descriptive.')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--session',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    for value in (args.session,args.output):
        if Path(value).is_absolute() or '..' in Path(value).parts:raise ValueError('Use repository-relative paths')
    output=(ROOT/args.output).resolve()
    require(output.is_relative_to(ROOT/'reports'),'Write audit reports only under reports/')
    result=audit(ROOT/args.session);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status=result['status'],output=output.relative_to(ROOT).as_posix())))


if __name__=='__main__':main()
