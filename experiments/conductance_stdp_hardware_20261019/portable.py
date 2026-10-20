"""STDP8 hardware launcher: unchanged numeric protocol, public MemriBoard/INI only."""
import argparse
import contextlib
import json
import logging
from pathlib import Path
import shutil
import sys
import uuid

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'.deps')]
from experiments.conductance_stdp_calibrated_20261019 import core
from experiments.conductance_stdp_20261019 import raw
from experiments.stable_cell_calibration_20261020 import experiment as calibration
from experiments.cifar_resumable_20261019.durable import atomic_json,sha
from experiments.unn_rev3_20261014.paths import use_local_dependencies
from read_all import now
from experiment_hardware import launch_provenance

EVIDENCE='reports/2026-10-19/conductance-stdp-calibrated-review/NATIVE_SIM_INDEPENDENT_AUDIT.json'
STATUS='private_runtime_not_independently_reverified'


def protocol_sha():return calibration.plan.digest(core.SPEC)


def preflight():
    config,hashes=calibration.preflight('hardware')
    paths=[Path(core.__file__),Path(core.previous.__file__),Path(raw.__file__),Path(__file__),
           ROOT/'native_conversions.py',ROOT/'experiment_hardware.py']
    hashes.update({p.relative_to(ROOT).as_posix():sha(p) for p in paths})
    evidence=ROOT/EVIDENCE
    return config,hashes,dict(path=EVIDENCE,sha256=sha(evidence) if evidence.is_file() else None,
                             usage='Historical SIM evidence; gain is newly fitted on this board')


def execute(argv=None):
    config,hashes,evidence=preflight()
    if shutil.disk_usage(ROOT).free<2*1024**3:raise ValueError('Need2GiB initial free space')
    sid=str(uuid.uuid4());out=ROOT/'results/conductance_stdp_hardware_20261019'/sid
    lock=ROOT/'results/read_all/.hardware.lock';lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('x',encoding='utf-8') as stream:stream.write(sid+'\n')
    writer=runner=None
    manifest=dict(schema='stdp8-public-launch-v2',session_id=sid,source='hardware',status='failed',
        started_utc=now(),protocol_sha256=protocol_sha(),source_sha256=hashes,
        launcher_sha256=sha(__file__),qualifying_sim_evidence=evidence,
        invocation=launch_provenance(argv),python=sys.version,
        private_runtime_status=STATUS,physical_pulse_duration_s=None,seed=core.SPEC['seed'],
        maximum_impacts=6712,maximum_pulses=128,
        scientific_scope='Algorithmic conductance STDP with digital LIF; not intrinsic delta-t STDP or accuracy')
    try:
        out.mkdir(parents=True,exist_ok=False)
        atomic_json(out/'protocol.json',core.SPEC)
        (out/'board.ini').write_bytes((ROOT/calibration.INI).read_bytes())
        atomic_json(out/'start_manifest.json',manifest)
        with (out/'lifecycle_output.txt').open('w',encoding='utf-8',newline='\n') as log:
            with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                use_local_dependencies()
                from manager.board import Connector
                from manager.service import converters
                writer=raw.Writer(out/'raw')
                connector=Connector(silent=True,logger=logging.getLogger(__name__),cb_type='real',
                    board_type=config.get('board','board_type'),task_format='SI',config=config)
                runner=core.Collector(connector,converters,writer,'hardware')
                result=runner.run();atomic_json(out/'observations.json',result)
                manifest.update(status=result['status'],impacts=result['impacts'],pulses=result['pulses'],
                                decisions=result['decisions'],error=result.get('error'))
    except (Exception,KeyboardInterrupt) as exc:manifest.update(status='failed',error=type(exc).__name__+': '+str(exc))
    finally:
        if writer:
            try:writer.finish()
            except Exception as exc:manifest.update(status='failed',raw_finish_error=repr(exc))
        if out.exists():
            if runner:
                manifest.update(impacts=runner.calls,pulses=runner.pulses,decisions=runner.decisions)
                if not (out/'observations.json').exists():
                    atomic_json(out/'partial_observations.json',dict(checkpoints=runner.checkpoints,windows=runner.windows,updates=runner.updates))
            manifest['finished_utc']=now()
            manifest['result_sha256']={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()}
            atomic_json(out/'manifest.json',manifest)
        if lock.read_text(encoding='utf-8').strip()!=sid:raise RuntimeError('Hardware lock owner changed')
        lock.unlink()
    return dict(session=out.relative_to(ROOT).as_posix(),status=manifest['status'],error=manifest.get('error'))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);m=parser.add_mutually_exclusive_group(required=True)
    m.add_argument('--dry-run',action='store_true');m.add_argument('--run-hardware',action='store_true')
    args=parser.parse_args(argv)
    if args.dry_run:
        config,hashes,evidence=preflight()
        result=dict(source='hardware',protocol=core.SPEC,protocol_sha256=protocol_sha(),
                    driver=config.get('board','board_type'),source_sha256=hashes,evidence=evidence,
                    private_runtime_status=STATUS,connector_opened=False,board_access=False)
    else:result=execute(argv)
    print(json.dumps(result,allow_nan=False));return 2 if result.get('status')=='failed' else 3 if result.get('status')=='uninformative_calibration' else 0

if __name__=='__main__':raise SystemExit(main())
