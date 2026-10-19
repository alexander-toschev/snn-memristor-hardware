"""Reviewed-protocol hardware STDP v2 launcher; MemriBoard only; execution gate closed."""
import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import platform
import shutil
import sys
import uuid

ROOT=Path(__file__).resolve().parents[2]
FROZEN=ROOT/'experiments/conductance_stdp_calibrated_20261019'
REVIEW=ROOT/'reports/2026-10-19/conductance-stdp-calibrated-review'
SIM=ROOT/'results/conductance_stdp_calibrated_20261019/d27da87e-9928-4303-971e-215afbc63158'
PROTOCOL_SHA='6d86180c0a73311753bfa420b66e8f6727d779106025ad23ac9d8cc6a3b414c5'
FREEZE_SHA='e6c33fc2da25b14f8653832e94398a2fa6275ee6ca101825226af256d7827ccd'
SIM_MANIFEST_SHA='d34b2b03d88c7d01e49e4817ce1dbfeca425853d28674a05343aee43e073c51c'
SIM_AUDIT_SHA='19ebfdbb29e4196a7780521743684f309939917310ce2543d7be508353b5ae53'
SIM_AUDITOR_SHA='f1fab366818f554e90dea185c0a23073074f0e2984bd5af3c15ff415cac8bcfa'
CONDITIONAL_SHA='3a019ee587a24fb0bd821b07655c5aefdd470c34326a8a1e775853e58a06948a'
LAUNCH_SPEC=dict(schema='memriboard-conductance-stdp-v2-hardware-launch-v1',protocol_sha256=PROTOCOL_SHA,
    source_freeze_sha256=FREEZE_SHA,qualifying_sim_manifest_sha256=SIM_MANIFEST_SHA,
    qualifying_sim_audit_sha256=SIM_AUDIT_SHA,conditional_qualification_sha256=CONDITIONAL_SHA,
    source='hardware',driver_selection='INI board.board_type through public manager.board.Connector',
    maximum_impacts=6712,maximum_pulses=128,selected_cells=8,hardware_pulse_duration_s=None,
    fresh_fit=True,reuse_sim_gain=False,after_cifar_release=True)
APPROVED_LAUNCH_SPEC_SHA256=frozenset({'ef9d61315c7a248732c0f56a877e92776db92746a8d2e4a24f02a2b1979635e9'})


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def verified_json(path,pin):
    if sha(path)!=pin:raise ValueError('Qualification/source hash changed: '+str(path.relative_to(ROOT)))
    return json.loads(path.read_text(encoding='utf-8'))


def qualification():
    freeze=verified_json(REVIEW/'SOURCE_FREEZE.json',FREEZE_SHA)
    audit=verified_json(REVIEW/'NATIVE_SIM_INDEPENDENT_AUDIT.json',SIM_AUDIT_SHA)
    conditional=verified_json(REVIEW/'HARDWARE_CONDITIONAL_QUALIFICATION.json',CONDITIONAL_SHA)
    manifest=verified_json(SIM/'manifest.json',SIM_MANIFEST_SHA)
    if sha(ROOT/'tools/audit_conductance_stdp_calibrated.py')!=SIM_AUDITOR_SHA:raise ValueError('Frozen qualifying auditor changed')
    if freeze['protocol_sha256']!=PROTOCOL_SHA or digest(freeze['protocol'])!=PROTOCOL_SHA:raise ValueError('Protocol changed')
    expected_files=set(freeze['file_sha256'])
    if {p.name for p in FROZEN.iterdir() if p.is_file() and p.suffix in ('.py','.md')}!=expected_files:
        raise ValueError('Frozen SIM source inventory changed')
    for name,pin in freeze['file_sha256'].items():
        if Path(name).name!=name or sha(FROZEN/name)!=pin:raise ValueError('Frozen SIM component changed')
    if audit['status']!='pass' or audit['manifest_sha256']!=SIM_MANIFEST_SHA or audit['auditor_sha256']!=SIM_AUDITOR_SHA:
        raise ValueError('Wrong independent SIM audit')
    if audit['protocol_sha256']!=PROTOCOL_SHA or audit['training_pulses']<=0 or audit['selected_gain'] is None:
        raise ValueError('SIM did not qualify active gain-calibrated learning branch')
    if conditional['status']!='conditional_go' or conditional['qualifying_audit_sha256']!=SIM_AUDIT_SHA or conditional['protocol_sha256']!=PROTOCOL_SHA:
        raise ValueError('Hardware conditions not reviewed')
    if manifest['source']!='simulator' or manifest['status']!='completed' or manifest['protocol_sha256']!=PROTOCOL_SHA:
        raise ValueError('Wrong qualifying SIM session')
    actual={p.relative_to(SIM).as_posix() for p in SIM.rglob('*') if p.is_file()}
    if actual!=set(manifest['result_sha256'])|{'manifest.json'}:raise ValueError('SIM result inventory changed')
    for name,pin in manifest['result_sha256'].items():
        path=(SIM/name).resolve()
        if not path.is_relative_to(SIM.resolve()) or sha(path)!=pin:raise ValueError('Qualifying SIM artifact changed')
    references={
        (REVIEW/'SOURCE_FREEZE.json').relative_to(ROOT).as_posix():FREEZE_SHA,
        (REVIEW/'NATIVE_SIM_INDEPENDENT_AUDIT.json').relative_to(ROOT).as_posix():SIM_AUDIT_SHA,
        (REVIEW/'HARDWARE_CONDITIONAL_QUALIFICATION.json').relative_to(ROOT).as_posix():CONDITIONAL_SHA,
        (SIM/'manifest.json').relative_to(ROOT).as_posix():SIM_MANIFEST_SHA,
        'tools/audit_conductance_stdp_calibrated.py':SIM_AUDITOR_SHA}
    return freeze,dict(session=SIM.relative_to(ROOT).as_posix(),reference_sha256=references,
        conditional_hardware_go=True,sim_gain_not_reused=True)


def preflight():
    freeze,qualified=qualification()
    sys.path[:0]=[str(ROOT/'.deps'),str(ROOT)]
    from experiments.conductance_stdp_calibrated_20261019 import experiment as native
    if Path(native.__file__).resolve()!=FROZEN/'experiment.py' or native.protocol_sha()!=PROTOCOL_SHA:
        raise ValueError('Wrong reviewed native entry/core')
    for name,pin in native.PINS.items():
        if sha(ROOT/name)!=pin:raise ValueError('Pinned public-Connector helper changed: '+name)
    from cell_calibration import check_sources,LOCK
    config,snapshot,hashes=check_sources('hardware')  # Strict private native files + INI; never opens Connector.
    hashes.update(native.PINS)
    hashes.update({(FROZEN/name).relative_to(ROOT).as_posix():pin for name,pin in freeze['file_sha256'].items()})
    if config.get('board','board_type')!='rp5_fpga_python':raise ValueError('Unreviewed INI driver')
    if shutil.disk_usage(ROOT).free<2*1024**3:raise ValueError('Need2GiB initial free-space reserve')
    launcher_sources={p.relative_to(ROOT).as_posix():sha(p) for p in Path(__file__).parent.iterdir()
                      if p.is_file() and p.suffix in ('.py','.md')}
    return native,config,snapshot,hashes,launcher_sources,qualified,ROOT/LOCK


def execute(native,config,snapshot,hashes,launcher_sources,qualified,lock,argv):
    from read_all import now,save_json
    from experiment_hardware import launch_provenance
    sid=str(uuid.uuid4());token='STDP-v2 hardware '+sid+' '+now()+'\n'
    lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('x',encoding='utf-8') as stream:stream.write(token)
    try:
        output=ROOT/'results/conductance_stdp_hardware_20261019'/sid;output.mkdir(parents=True,exist_ok=False)
        save_json(output/'protocol.json',native.core.SPEC);(output/'board.ini').write_bytes(snapshot)
        manifest=dict(session_id=sid,source='hardware',status='failed',started_utc=now(),protocol_sha256=PROTOCOL_SHA,
            source_sha256=hashes,launcher_sha256=sha(__file__),launcher_source_sha256=launcher_sources,
            launcher_spec=LAUNCH_SPEC,launcher_spec_sha256=digest(LAUNCH_SPEC),qualification=qualified,
            qualifying_sim_audit_sha256=SIM_AUDIT_SHA,invocation=launch_provenance(argv),
            scientific_scope='fit-calibrated boundary-updated conductance STDP; not intrinsic delta-t STDP or classification accuracy',
            physical_pulse_duration_s=None,seed=native.core.SPEC['seed'])
        packages={}
        for name in ('numpy','SQLAlchemy','greenlet'):
            try:packages[name]=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:packages[name]=None
        manifest['environment']=dict(python=sys.version,platform=platform.platform(),packages=packages)
        save_json(output/'start_manifest.json',manifest)
        writer=None;runner=None
        try:
            writer=native.raw.Writer(output/'raw')
            # Ordinary public API import, with actual driver selected from reviewed INI.
            sys.path.insert(0,str(ROOT/'vendor/MemriBoard'))
            from manager.board import Connector
            from manager.service import converters
            if Path(sys.modules['manager.board'].__file__).resolve()!=ROOT/'vendor/MemriBoard/manager/board.py':
                raise ValueError('Public Connector imported outside reviewed MemriBoard')
            if Path(converters.__file__).resolve()!=ROOT/'vendor/MemriBoard/manager/service/converters.py':
                raise ValueError('Native converters imported outside reviewed MemriBoard')
            connector=Connector(silent=True,logger=logging.getLogger(__name__),cb_type='real',
                board_type=config.get('board','board_type'),task_format='SI',config=config)
            runner=native.core.Collector(connector,converters,writer,'hardware')
            result=runner.run();save_json(output/'observations.json',result)
            manifest.update(status=result['status'],impacts=result['impacts'],pulses=result['pulses'],decisions=result['decisions'])
        except (Exception,KeyboardInterrupt) as exc:manifest['error']=type(exc).__name__+': '+str(exc)
        finally:
            if writer is not None:
                try:writer.finish()
                except Exception as exc:manifest.update(status='failed',raw_finish_error=repr(exc))
            if runner is not None:
                manifest.update(impacts=runner.calls,pulses=runner.pulses,decisions=runner.decisions)
                if manifest['status'] not in ('completed','uninformative_calibration'):
                    save_json(output/'partial_observations.json',dict(checkpoints=runner.checkpoints,windows=runner.windows,updates=runner.updates))
            manifest['finished_utc']=now()
            manifest['result_sha256']={p.relative_to(output).as_posix():sha(p) for p in output.rglob('*') if p.is_file()}
            save_json(output/'manifest.json',manifest)
        print(json.dumps(dict(session=output.relative_to(ROOT).as_posix(),status=manifest['status'],error=manifest.get('error'))))
        return 0 if manifest['status']=='completed' else 3 if manifest['status']=='uninformative_calibration' else 2
    finally:
        # Never remove a lock owned by some other experiment.
        if lock.read_text(encoding='utf-8')!=token:raise RuntimeError('Hardware lock ownership changed')
        lock.unlink()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-run',action='store_true');mode.add_argument('--run-hardware',action='store_true');args=parser.parse_args(argv)
    if args.run_hardware and digest(LAUNCH_SPEC) not in APPROVED_LAUNCH_SPEC_SHA256:
        raise ValueError('Separate hardware launcher/auditor review and registration required')
    prepared=preflight()
    if args.dry_run:
        native,config,snapshot,hashes,launcher_sources,qualified,lock=prepared
        print(json.dumps(dict(source='hardware',protocol=native.core.SPEC,protocol_sha256=PROTOCOL_SHA,
            launcher_spec=LAUNCH_SPEC,launcher_spec_sha256=digest(LAUNCH_SPEC),launcher_sha256=sha(__file__),
            source_sha256=hashes,launcher_source_sha256=launcher_sources,qualification=qualified,
            registered=digest(LAUNCH_SPEC) in APPROVED_LAUNCH_SPEC_SHA256,hardware_lock_present=lock.exists(),
            connector_opened=False,board_access=False)));return 0
    return execute(*prepared,argv)


if __name__=='__main__':raise SystemExit(main())
