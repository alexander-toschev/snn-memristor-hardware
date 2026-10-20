"""MNIST128 conductance hardware pilot: public MemriBoard/INI only; no accuracy claim."""
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
from experiments.mnist_conductance_20261019 import core,experiment as original
from experiments.cifar_resumable_20261019 import durable
from experiments.stable_cell_calibration_20261020 import experiment as calibration
from experiments.cifar_resumable_20261019.durable import atomic_json,sha
from experiments.unn_rev3_20261014.paths import use_local_dependencies
from read_all import now
from experiment_hardware import launch_provenance

EVIDENCE=('reports/2026-10-20/mnist-native-sim/MAIN_AUDIT.json',
          'reports/2026-10-20/mnist-native-sim/REFERENCE_AUDIT.json')
STATUS='private_runtime_not_independently_reverified'


def protocol_sha():return calibration.plan.digest(core.SPEC)


def preflight():
    config,hashes=calibration.preflight('hardware')
    fixture=original.load_fixture()
    paths=[Path(core.__file__),Path(original.__file__),Path(durable.__file__),Path(__file__),
           ROOT/original.FIXTURE,ROOT/'tools/prepare_mnist_conductance_fixture.py',
           ROOT/'experiments/conductance_stdp_20261019/core.py',
           ROOT/'native_conversions.py',ROOT/'experiment_hardware.py']
    hashes.update({p.relative_to(ROOT).as_posix():sha(p) for p in paths})
    evidence=[dict(path=name,sha256=sha(ROOT/name) if (ROOT/name).is_file() else None) for name in EVIDENCE]
    return config,hashes,fixture,evidence


def execute(argv=None):
    config,hashes,fixture,evidence=preflight()
    if shutil.disk_usage(ROOT).free<core.SPEC['initial_minimum_free_bytes']:raise ValueError('Need6GiB initial free space')
    sid=str(uuid.uuid4());out=ROOT/'results/mnist_conductance_20261019'/sid
    lock=ROOT/'results/read_all/.hardware.lock';lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('x',encoding='utf-8') as stream:stream.write(sid+'\n')
    writer=runner=connector=None;lifecycle_assumed=False
    manifest=dict(schema='mnist128-public-hardware-launch-v1',session_id=sid,source='hardware',status='failed',
        started_utc=now(),protocol_sha256=protocol_sha(),source_sha256=hashes,
        launcher_sha256=sha(__file__),qualifying_sim_evidence=evidence,
        invocation=launch_provenance(argv),python=sys.version,
        private_runtime_status=STATUS,physical_pulse_duration_s=None,seed=core.SPEC['seed'],
        maximum_impacts=core.SPEC['maximum_impacts'],maximum_pulses=core.SPEC['maximum_pulses'],
        fixture_sha256=original.FIXTURE_SHA,reference=None,classification_accuracy_measured=False,full10k_executed=False,
        environment=dict(python=sys.version),
        scientific_scope='128 training images, measured conductance STDP mechanism; not classification accuracy or full10k')
    try:
        out.mkdir(parents=True,exist_ok=False)
        atomic_json(out/'protocol.json',core.SPEC)
        (out/'board.ini').write_bytes((ROOT/calibration.INI).read_bytes())
        (out/'fixture.json').write_bytes((ROOT/original.FIXTURE).read_bytes())
        atomic_json(out/'start_manifest.json',manifest)
        with (out/'lifecycle_output.txt').open('w',encoding='utf-8',newline='\n') as log:
            with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                use_local_dependencies()
                from manager.board import Connector
                from manager.service import converters
                writer=durable.Writer(out/'raw',limit=core.SPEC['raw_chunk_bytes'],
                    maximum_expanded_bytes=core.SPEC['maximum_raw_expanded_bytes'],
                    minimum_free_bytes=core.SPEC['raw_minimum_free_bytes'])
                connector=Connector(silent=True,logger=logging.getLogger(__name__),cb_type='real',
                    board_type=config.get('board','board_type'),task_format='SI',config=config)
                runner=core.Collector(connector,converters,writer,'hardware',fixture)
                lifecycle_assumed=True
                result=runner.run();atomic_json(out/'observations.json',result)
                manifest.update(status=result['status'],impacts=result['impacts'],pulses=result['pulses'],
                                decisions=result['decisions'],measured_images=result['measured_images'],gain=result['gain'])
    except (Exception,KeyboardInterrupt) as exc:manifest.update(status='failed',error=type(exc).__name__+': '+str(exc))
    finally:
        if connector is not None and not lifecycle_assumed:
            try:
                if not connector.close_port():manifest.update(status='failed',cleanup_error='Public close returned false')
            except Exception as exc:manifest.update(status='failed',cleanup_error=repr(exc))
        if writer:
            try:writer.finish()
            except Exception as exc:manifest.update(status='failed',raw_finish_error=repr(exc))
        if out.exists():
            if runner:
                manifest.update(impacts=runner.calls,pulses=runner.pulses,decisions=runner.decisions,measured_images=runner.ordinal,gain=runner.gain)
                if not (out/'observations.json').exists():
                    atomic_json(out/'partial_observations.json',runner.summary('failed'))
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
        config,hashes,fixture,evidence=preflight()
        result=dict(source='hardware',protocol=core.SPEC,protocol_sha256=protocol_sha(),
                    driver=config.get('board','board_type'),source_sha256=hashes,evidence=evidence,
                    fixture_sha256=original.FIXTURE_SHA,fixture_unique_images=len(fixture['images']),
                    private_runtime_status=STATUS,connector_opened=False,board_access=False)
    else:result=execute(argv)
    print(json.dumps(result,allow_nan=False));return 2 if result.get('status')=='failed' else 3 if result.get('status')=='uninformative_calibration' else 0

if __name__=='__main__':raise SystemExit(main())
