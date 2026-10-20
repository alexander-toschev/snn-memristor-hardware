"""One public MemriBoard pipeline: STDP128, frozen train128/test100, ridge accuracy."""
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
from experiments.mnist64_20261020 import core,inputs,readout
from experiments.cifar_resumable_20261019.durable import Writer,atomic_json,sha
from experiments.stable_cell_calibration_20261020 import experiment as calibration
from experiments.unn_rev3_20261014.paths import use_local_dependencies
from experiment_hardware import launch_provenance
from read_all import now
BASE=ROOT/'results/mnist64_20261020'
SEED=20261019


def preflight(source):
    config,hashes=calibration.preflight(source)
    fixtures={s:json.loads((inputs.HERE/('fixture_'+s+'.json')).read_text()) for s in ('train','accuracy')}
    for f in fixtures.values():inputs.verify(f)
    for folder in ('mnist64_20261020','mnist_conductance_20261019'):
        for p in (ROOT/'experiments'/folder).glob('*.py'):hashes[p.relative_to(ROOT).as_posix()]=sha(p)
    for name in ('experiments/cifar_resumable_20261019/durable.py','experiments/conductance_stdp_20261019/core.py',
                 'experiments/mnist_accuracy_20261020/analyze.py','native_conversions.py','experiment_hardware.py'):
        hashes[name]=sha(ROOT/name)
    return config,hashes,fixtures


def stage(source,scope,config,hashes,fixture,argv,parent_dir=None):
    sid=str(uuid.uuid4());out=BASE/sid;writer=runner=connector=None;lifecycle=False
    parent=None;gain=None
    if parent_dir:
        pm=json.loads((parent_dir/'manifest.json').read_text())
        if pm['status']!='completed' or pm['source']!=source:raise ValueError('Completed same-source parent required')
        parent=dict(session_id=parent_dir.name,manifest_sha256=sha(parent_dir/'manifest.json'),
            observations_sha256=sha(parent_dir/'observations.json'),gain=pm['gain'],
            final_array_sha256=sha(parent_dir/'array.cb') if source=='simulator' else None)
        gain=parent['gain']
    manifest=dict(schema='mnist64-pipeline-stage-v1',session_id=sid,source=source,scope=scope,status='failed',
        started_utc=now(),protocol_sha256=inputs.digest(core.SPEC),source_sha256=hashes,seed=SEED,
        invocation=launch_provenance(argv),parent=parent,initial_gain=gain,
        fixture_sha256=sha(inputs.HERE/('fixture_'+scope+'.json')),limits=inputs.budget(fixture),
        private_runtime_status='private_runtime_not_independently_reverified',python=sys.version,
        physical_pulse_duration_s=None,labels_in_acquisition=False,separate_control_bank=False)
    progress=lambda text:print(text,flush=True)
    try:
        out.mkdir(parents=True,exist_ok=False);progress('Session %s %s %s'%(sid,source,scope))
        atomic_json(out/'protocol.json',core.SPEC);atomic_json(out/'fixture.json',fixture)
        # Preserve fixture bytes so its acquired binding is directly comparable.
        shutil.copyfile(inputs.HERE/('fixture_'+scope+'.json'),out/'fixture.json')
        shutil.copyfile(ROOT/calibration.INI,out/'board.ini')
        if parent_dir:
            for name in ('manifest.json','observations.json'):shutil.copyfile(parent_dir/name,out/('parent_'+name))
        with (out/'lifecycle_output.txt').open('w',encoding='utf-8',newline='\n') as log:
            with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                use_local_dependencies()
                from manager.board import Connector
                from manager.service import converters
                writer=Writer(out/'raw',limit=8*1024**2,maximum_expanded_bytes=2*1024**3,minimum_free_bytes=2*1024**3)
                kwargs={}
                if source=='simulator':
                    import numpy as np
                    random.seed(SEED);np.random.seed(SEED)
                    if parent_dir:shutil.copyfile(parent_dir/'array.cb',out/'array.cb')
                    else:
                        from simulator.src import create_crossbar_array
                        create_crossbar_array(str(out/'array'),32,8)
                    shutil.copyfile(out/'array.cb',out/'array_initial.cb')
                    manifest['array_initial_sha256']=sha(out/'array_initial.cb')
                    if parent and manifest['array_initial_sha256']!=parent['final_array_sha256']:raise ValueError('Parent final array copy differs')
                    shutil.copyfile(ROOT/calibration.SIM_INI,out/'simulator_settings.ini')
                    kwargs['crossbar_serial']=str(out/'array')
                atomic_json(out/'start_manifest.json',manifest)
                connector=Connector(silent=True,logger=logging.getLogger(__name__),cb_type='real' if source=='hardware' else 'simulator',
                    board_type=config.get('board','board_type'),task_format='SI',config=config,**kwargs)
        runner=core.Collector(connector,converters,writer,source,fixture,gain,parent,progress)
        lifecycle=True;result=runner.run();atomic_json(out/'observations.json',result)
        manifest.update(status=result['status'])
        if scope=='accuracy' and result['status']=='completed':
            scores=readout.classify(result,out,progress)
            manifest['accuracy']={k:scores[k] for k in ('strict_accuracy','conventional_argmax_accuracy','coverage','silent','tied')}
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
                manifest.update(impacts=runner.calls,pulses=runner.pulses,decisions=runner.decisions,measured_images=runner.ordinal,gain=runner.gain)
                if not (out/'observations.json').exists():atomic_json(out/'partial_observations.json',runner.summary('failed'))
            manifest['finished_utc']=now()
            manifest['result_sha256']={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()}
            atomic_json(out/'manifest.json',manifest)
    return out,manifest


def execute(source,argv):
    config,hashes,fixtures=preflight(source)
    if shutil.disk_usage(ROOT).free<6*1024**3:raise ValueError('Need6GiB initial free space')
    lock=ROOT/'results/read_all/.hardware.lock';owner=str(uuid.uuid4());locked=False
    lock.parent.mkdir(parents=True,exist_ok=True)
    if source=='hardware':
        with lock.open('x',encoding='utf-8') as f:f.write(owner+'\n')
        locked=True
    sessions=[]
    try:
        folder,m=stage(source,'train',config,hashes,fixtures['train'],argv)
        sessions.append(folder.relative_to(ROOT).as_posix())
        if m['status']=='completed':
            folder,m=stage(source,'accuracy',config,hashes,fixtures['accuracy'],argv,folder)
            sessions.append(folder.relative_to(ROOT).as_posix())
        return dict(sessions=sessions,status=m['status'],error=m.get('error'),accuracy=m.get('accuracy'))
    finally:
        if locked:
            if lock.read_text().strip()!=owner:raise ValueError('Lock ownership differs')
            lock.unlink()


def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--dry-run',action='store_true');g.add_argument('--run-simulator',action='store_true');g.add_argument('--run-hardware',action='store_true')
    p.add_argument('--preflight-source',choices=['hardware','simulator'],default='hardware');a=p.parse_args()
    source=a.preflight_source if a.dry_run else 'hardware' if a.run_hardware else 'simulator'
    if a.dry_run:
        _,_,f=preflight(source);caps={s:inputs.budget(v) for s,v in f.items()}
        result=dict(source=source,seed=SEED,limits=caps,combined_maximum_impacts=sum(v['maximum_impacts'] for v in caps.values()),
            maximum_pulses=2560,connector_opened=False,labels_loaded=False)
    else:result=execute(source,sys.argv[1:])
    print(json.dumps(result),flush=True);return 2 if result.get('status')=='failed' else 3 if result.get('status')=='uninformative_calibration' else 0

if __name__=='__main__':raise SystemExit(main())
