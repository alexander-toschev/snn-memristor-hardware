"""Frozen post-STDP MNIST feature acquisition, public MemriBoard only."""
import argparse
import contextlib
import json
import logging
from pathlib import Path
import random
import shutil
import sys
import uuid
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'.deps')]
from experiments.mnist_accuracy_20261020 import core,inputs
from experiments.cifar_resumable_20261019.durable import Writer,atomic_json,sha
from experiments.stable_cell_calibration_20261020 import experiment as calibration
from experiments.unn_rev3_20261014.paths import use_local_dependencies,repo_path
from experiment_hardware import launch_provenance
from read_all import now

PARENTS={'hardware':'a4327210-2cc4-4c19-bb7b-b8820d2e84e4',
         'simulator':'ee0db984-cd00-4052-95e6-069e71c720eb'}
BASE=ROOT/'results/mnist_accuracy_20261020'


def preflight(source):
    config,hashes=calibration.preflight(source)
    fixture=json.loads((inputs.HERE/'fixture.json').read_text());inputs.verify(fixture)
    parent_dir=ROOT/'results/mnist_diagonal_20261020'/PARENTS[source]
    manifest=json.loads((parent_dir/'manifest.json').read_text())
    if manifest['status']!='completed' or manifest['source']!=source or manifest['session_id']!=PARENTS[source]:
        raise ValueError('Completed parent source/session required')
    obs=parent_dir/'observations.json'
    if sha(obs)!=manifest['result_sha256']['observations.json']:raise ValueError('Parent observations differ')
    observations=json.loads(obs.read_text())
    if manifest['gain']!=observations['gain']:raise ValueError('Parent gain differs')
    for folder in ('mnist_accuracy_20261020','mnist_diagonal_20261020','mnist_conductance_20261019'):
        for p in (ROOT/'experiments'/folder).glob('*.py'):hashes[p.relative_to(ROOT).as_posix()]=sha(p)
    for name in ('experiments/cifar_resumable_20261019/durable.py','tools/prepare_mnist_conductance_fixture.py',
                 'experiments/conductance_stdp_20261019/core.py','native_conversions.py','experiment_hardware.py'):
        hashes[name]=sha(ROOT/name)
    binding=dict(session_id=PARENTS[source],manifest_sha256=sha(parent_dir/'manifest.json'),
                 observations_sha256=sha(obs),gain=manifest['gain'],final_array_sha256=None)
    if source=='simulator':
        final=parent_dir/'array.cb'
        if sha(final)!=manifest['result_sha256']['array.cb']:raise ValueError('Final parent SIM array differs')
        binding['final_array_sha256']=sha(final)
    return config,hashes,fixture,parent_dir,binding


def execute(source,argv):
    config,hashes,fixture,parent_dir,parent=preflight(source)
    if shutil.disk_usage(ROOT).free<6*1024**3:raise ValueError('Need6GiB initial free space')
    sid=str(uuid.uuid4());out=BASE/sid
    lock=ROOT/'results/read_all/.hardware.lock';lock.parent.mkdir(parents=True,exist_ok=True)
    locked=False;writer=runner=connector=None;lifecycle=False
    if source=='hardware':
        with lock.open('x',encoding='utf-8') as f:f.write(sid+'\n')
        locked=True
    manifest=dict(schema='mnist-frozen-accuracy-acquisition-v1',session_id=sid,source=source,
        status='failed',started_utc=now(),protocol_sha256=inputs.digest(core.SPEC),source_sha256=hashes,
        invocation=launch_provenance(argv),parent=parent,gain=parent['gain'],
        fixture_sha256=sha(inputs.HERE/'fixture.json'),private_runtime_status='private_runtime_not_independently_reverified',
        maximum_impacts=128064,exact_impacts=inputs.impact_count(fixture),maximum_pulses=0,
        labels_loaded=False,classification_accuracy_measured=False,python=sys.version)
    try:
        out.mkdir(parents=True,exist_ok=False)
        atomic_json(out/'protocol.json',core.SPEC)
        shutil.copyfile(inputs.HERE/'fixture.json',out/'fixture.json')
        shutil.copyfile(ROOT/calibration.INI,out/'board.ini')
        shutil.copyfile(parent_dir/'manifest.json',out/'parent_manifest.json')
        shutil.copyfile(parent_dir/'observations.json',out/'parent_observations.json')
        with (out/'lifecycle_output.txt').open('w',encoding='utf-8',newline='\n') as log:
            with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                use_local_dependencies()
                from manager.board import Connector
                from manager.service import converters
                writer=Writer(out/'raw',limit=8*1024**2,maximum_expanded_bytes=2*1024**3,minimum_free_bytes=2*1024**3)
                kwargs={}
                if source=='simulator':
                    import numpy as np
                    random.seed(20261020);np.random.seed(20261020)
                    shutil.copyfile(parent_dir/'array.cb',out/'array.cb')
                    shutil.copyfile(parent_dir/'array.cb',out/'array_initial.cb')
                    if sha(out/'array_initial.cb')!=parent['final_array_sha256']:raise ValueError('Array copy differs')
                    shutil.copyfile(ROOT/calibration.SIM_INI,out/'simulator_settings.ini')
                    kwargs['crossbar_serial']=str(out/'array')
                atomic_json(out/'start_manifest.json',manifest)
                connector=Connector(silent=True,logger=logging.getLogger(__name__),
                    cb_type='real' if source=='hardware' else 'simulator',board_type=config.get('board','board_type'),
                    task_format='SI',config=config,**kwargs)
                runner=core.Collector(connector,converters,writer,source,fixture,parent['gain'],parent)
                lifecycle=True;result=runner.run()
                atomic_json(out/'observations.json',result);manifest.update(status='completed')
    except (Exception,KeyboardInterrupt) as exc:
        manifest.update(status='failed',error=type(exc).__name__+': '+str(exc))
    finally:
        if connector is not None and not lifecycle:
            try:
                if not connector.close_port():manifest['cleanup_error']='Public close returned false'
            except Exception as exc:manifest['cleanup_error']=repr(exc)
        if writer:
            try:writer.finish()
            except Exception as exc:manifest.update(status='failed',raw_finish_error=repr(exc))
        if out.exists():
            if runner:
                manifest.update(impacts=runner.calls,pulses=runner.pulses,decisions=runner.decisions,measured_images=runner.ordinal)
                if not (out/'observations.json').exists():atomic_json(out/'partial_observations.json',runner.summary('failed'))
            manifest['finished_utc']=now()
            manifest['result_sha256']={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()}
            atomic_json(out/'manifest.json',manifest)
        if locked:
            if lock.read_text().strip()!=sid:raise ValueError('Lock ownership changed')
            lock.unlink()
    return dict(session=out.relative_to(ROOT).as_posix(),status=manifest['status'],error=manifest.get('error'))


def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--dry-run',action='store_true');g.add_argument('--run-simulator',action='store_true');g.add_argument('--run-hardware',action='store_true')
    p.add_argument('--preflight-source',choices=['hardware','simulator'],default='hardware')
    a=p.parse_args();source=a.preflight_source if a.dry_run else 'hardware' if a.run_hardware else 'simulator'
    if a.dry_run:
        _,_,fixture,_,parent=preflight(source)
        result=dict(source=source,parent=parent,protocol=core.SPEC,exact_impacts=inputs.impact_count(fixture),
                    pulses=0,connector_opened=False)
    else:result=execute(source,sys.argv[1:])
    print(json.dumps(result));return 2 if result.get('status')=='failed' else 0

if __name__=='__main__':raise SystemExit(main())
