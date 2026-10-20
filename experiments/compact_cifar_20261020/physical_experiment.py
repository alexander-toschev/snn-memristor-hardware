"""Prepare compact physical mapping, acquire via MemriBoard, or classify locally."""
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
for p in (ROOT,ROOT/'.deps'):
    if p.is_dir():sys.path.insert(0,str(p))
import numpy as np
from experiments.compact_cifar_20261020 import data,hardware
from experiments.stable_cell_calibration_20261020 import experiment as cal_entry,plan as cal_plan
from experiments.cifar_resumable_20261019.durable import Writer,atomic_json,sha
from experiments.unn_rev3_20261014.paths import repo_path,use_local_dependencies
from read_all import now


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def load_inputs(cpu):
    manifest=read(cpu/'manifest.json');freeze=read(cpu/'checkpoint_freeze.json')
    if manifest['status']!='completed':raise ValueError('CPU checkpoint not completed')
    if sha(cpu/'checkpoint_freeze.json')!=manifest['result_sha256']['checkpoint_freeze.json']:
        raise ValueError('Checkpoint freeze changed')
    for name in ('W','A','b'):
        if sha(cpu/(name+'.npy'))!=freeze[name+'_sha256']:raise ValueError('Frozen weight bytes changed')
    inp=read(cpu/'input_freeze.json')
    if sha(cpu/'input_freeze.json')!=freeze['input_freeze_sha256']:
        raise ValueError('Pre-training input freeze changed')
    if sha(cpu/'acquisition_fixture.json')!=inp['acquisition_fixture_sha256']:
        raise ValueError('Input fixture differs from pre-training freeze')
    fixture=read(cpu/'acquisition_fixture.json');cases=fixture['cases']
    if len(cases)!=100 or any('label' in row or 'fine_label' in row for row in cases):
        raise ValueError('Expected100 unlabeled acquisition cases')
    images=np.asarray([r['image_chw_uint8'] for r in cases],dtype=np.uint8)
    sums,times,spikes=data.encode(images)
    for i,row in enumerate(cases):
        if row['pool_sums']!=sums[i].tolist() or row['ttfs_times']!=times[i].tolist():
            raise ValueError('Input pooling/schedule mismatch')
    w=np.load(cpu/'W.npy',allow_pickle=False)
    if w.shape!=(8,48) or w.dtype!=np.float32 or not np.isfinite(w).all():raise ValueError('Invalid physical weights')
    return w,times


def prepare(cpu,map_session,output):
    weights,times=load_inputs(cpu);manifest=read(map_session/'manifest.json')
    if manifest['status']!='completed':raise ValueError('Map acquisition not completed')
    value=read(map_session/'map.json');selected=value['selection']['selected']
    gain=hardware.select_gain(value,weights)
    stages=hardware.stage_layout(selected['pairs'],selected['bls'])
    protocol=dict(schema='compact-cifar-physical-v1',source=manifest['source'],seed=20261020,
        cpu_session=cpu.relative_to(ROOT).as_posix(),map_session=map_session.relative_to(ROOT).as_posix(),
        map_sha256=sha(map_session/'map.json'),checkpoint_sha256=sha(cpu/'selected.pt'),
        fixture_sha256=sha(cpu/'acquisition_fixture.json'),weights_sha256=sha(cpu/'W.npy'),
        gain_v_per_weight=gain,gain_rule='0.8 * min B signed immediate/delayed(mean-3SE)*ADC_LSB / max(abs W)',
        amplitude_v=.05,maximum_active_bl=4,stages=stages,limits=hardware.bounds(stages,times.tolist()),
        reset_levels_v=cal_plan.LEVELS,set_levels_v=cal_plan.LEVELS,
        programming='8 RESET each leg, up to8 SET one leg, fresh3-repeat differential feedback; approximate continue',
        hardware_submitted_width_s=0.,physical_pulse_duration_s=None,simulator_width_s=.000125,
        current_limits_a={'reset':.04,'set':.002,'MVM':.005},output_prediction_limit_v=3.3,
        measured_limit_v=4.,matrix_target='measured singleton superposition; never CPU target weights',
        retention='fresh3repeat singleton basis after each20 images; fixed C, no test refit',
        cpu_fill=False,labels_loaded_by_acquisition=False,
        map_quality_qualified=value['qualified'],
        private_runtime_status='private_runtime_not_independently_reverified')
    if output.exists():raise FileExistsError('Use a new plan path, do not overwrite prior plan')
    output.parent.mkdir(parents=True,exist_ok=True);atomic_json(output,protocol)
    return dict(plan=output.relative_to(ROOT).as_posix(),stages=len(stages),gain=gain,limits=protocol['limits'])


def validate_plan(path,source):
    p=read(path);cpu=repo_path(p['cpu_session']);weights,times=load_inputs(cpu)
    if p['schema']!='compact-cifar-physical-v1' or p['source']!=source:raise ValueError('Plan/source mismatch')
    if (p['amplitude_v']!=.05 or p['maximum_active_bl']!=4
        or p['reset_levels_v']!=cal_plan.LEVELS or p['set_levels_v']!=cal_plan.LEVELS
        or p['hardware_submitted_width_s']!=0. or p['simulator_width_s']!=.000125
        or p['physical_pulse_duration_s'] is not None or p['cpu_fill'] is not False):
        raise ValueError('Physical protocol constants differ from the executor')
    if p['fixture_sha256']!=sha(cpu/'acquisition_fixture.json') or p['weights_sha256']!=sha(cpu/'W.npy'):
        raise ValueError('Plan/frozen inputs changed')
    map_session=repo_path(p['map_session']);value=read(map_session/'map.json')
    if sha(map_session/'map.json')!=p['map_sha256']:raise ValueError('Map changed')
    selected=value['selection']['selected'];stages=hardware.stage_layout(selected['pairs'],selected['bls'])
    if stages!=p['stages'] or hardware.bounds(stages,times.tolist())!=p['limits']:
        raise ValueError('Geometry/budget mismatch')
    if hardware.select_gain(value,weights)!=p['gain_v_per_weight']:raise ValueError('Frozen global gain changed')
    return p,weights,times


def acquire(path,source):
    p,weights,times=validate_plan(path,source)
    config,hashes=cal_entry.preflight(source)
    hashes.update({f.relative_to(ROOT).as_posix():sha(f) for f in Path(__file__).parent.iterdir() if f.suffix=='.py'})
    out=ROOT/'results/compact_cifar_physical_20261020'/str(uuid.uuid4());out.mkdir(parents=True)
    if shutil.disk_usage(out).free<4*1024**3:raise ValueError('Insufficient disk reserve')
    manifest=dict(status='failed',source=source,session_id=out.name,started_utc=now(),
                  protocol_sha256=cal_plan.digest(p),source_sha256=hashes,
                  private_runtime_status='private_runtime_not_independently_reverified')
    atomic_json(out/'protocol.json',p);(out/'board.ini').write_bytes((ROOT/cal_entry.INI).read_bytes())
    lock=ROOT/'results/read_all/.hardware.lock';locked=False;writer=runner=None
    try:
        if source=='hardware':
            lock.parent.mkdir(parents=True,exist_ok=True)
            with lock.open('x',encoding='utf-8') as f:f.write(out.name+'\n')
            locked=True
        use_local_dependencies()
        from manager.board import Connector
        from manager.service import converters
        writer=Writer(out/'raw',limit=8*1024**2,maximum_expanded_bytes=2*1024**3,minimum_free_bytes=1024**3)
        with (out/'lifecycle_output.txt').open('w',encoding='utf-8',newline='\n') as log:
            with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                kwargs={}
                if source=='simulator':
                    from simulator.src import create_crossbar_array
                    random.seed(p['seed']);np.random.seed(p['seed'])
                    serial=out/'array';create_crossbar_array(str(serial),32,8)
                    shutil.copyfile(serial.with_suffix('.cb'),out/'array_initial.cb')
                    manifest['array_initial_sha256']=sha(out/'array_initial.cb')
                    kwargs['crossbar_serial']=str(serial)
                atomic_json(out/'start_manifest.json',manifest)
                connector=Connector(silent=True,logger=logging.getLogger(__name__),
                    cb_type='real' if source=='hardware' else 'simulator',board_type=config.get('board','board_type'),
                    task_format='SI',config=config,**kwargs)
                runner=hardware.Collector(connector,converters,writer,source,p,weights,times.tolist())
                result=runner.run_compact()
        atomic_json(out/'measured_tiles.json',result)
        manifest.update(status=result['status'],error=result['error'],impacts=result['impacts'],pulses=result['pulses'])
    except (Exception,KeyboardInterrupt) as exc:manifest.update(error=type(exc).__name__+': '+str(exc))
    finally:
        if writer:
            try:writer.finish()
            except Exception as exc:manifest.update(status='failed',raw_finish_error=repr(exc))
        if runner and not (out/'measured_tiles.json').exists():
            atomic_json(out/'measured_tiles.json',dict(status='failed',stages=runner.outputs,error=manifest.get('error')))
        manifest['finished_utc']=now()
        manifest['result_sha256']={f.relative_to(out).as_posix():sha(f) for f in out.rglob('*') if f.is_file()}
        atomic_json(out/'manifest.json',manifest)
        if locked:
            if lock.read_text(encoding='utf-8').strip()!=out.name:raise ValueError('Lock owner changed')
            lock.unlink()
    return dict(session=out.relative_to(ROOT).as_posix(),status=manifest['status'],error=manifest.get('error'))


def classify(session,output):
    manifest=read(session/'manifest.json')
    if manifest['status']!='completed':raise ValueError('Cannot classify partial physical acquisition')
    for name,digest in manifest['result_sha256'].items():
        path=(session/name).resolve()
        if not path.is_relative_to(session.resolve()) or sha(path)!=digest:
            raise ValueError('Physical result bytes changed: '+name)
    p=read(session/'protocol.json');rows=read(session/'measured_tiles.json')['stages']
    cpu=repo_path(p['cpu_session']);load_inputs(cpu)
    evaluation=read(cpu/'sealed_evaluation.json');fixture=read(cpu/'acquisition_fixture.json')
    if [r['official_test_index'] for r in evaluation['cases']]!=[r['official_test_index'] for r in fixture['cases']]:
        raise ValueError('Evaluation/acquisition ID mismatch')
    truth=np.asarray([r['label'] for r in evaluation['cases']])
    A=np.load(cpu/'A.npy',allow_pickle=False);b=np.load(cpu/'b.npy',allow_pickle=False)
    report=dict(source=manifest['source'],physical_session=session.relative_to(ROOT).as_posix(),
                denominator=100,branches={},cpu_fill=False,hardware_accuracy=manifest['source']=='hardware')
    for branch in ('raw','calibrated'):
        try:currents=hardware.aggregate(p,rows,branch)
        except ValueError as exc:
            if branch=='raw':raise
            report['branches'][branch]=dict(status='unavailable',reason=str(exc));continue
        logits,z=hardware.readout(currents,A,b);pred=logits.argmax(1)
        margin=np.sort(logits,axis=1)[:,-1]-np.sort(logits,axis=1)[:,-2]
        silent=z.sum(axis=(1,2))==0;near=margin<=1e-6
        confusion=np.zeros((10,10),dtype=int)
        for a,c in zip(truth,pred):confusion[a,c]+=1
        report['branches'][branch]=dict(status='completed',correct=int((pred==truth).sum()),
            accuracy=float((pred==truth).mean()),informative_correct=int(((pred==truth)&~silent&~near).sum()),
            all_silent=int(silent.sum()),exact_ties=int((margin==0).sum()),near_ties=int(near.sum()),
            confusion=confusion.tolist(),predictions=pred.tolist(),logits=logits.tolist(),spike_counts=z.sum(1).tolist())
    if output.exists():raise FileExistsError('Do not overwrite a prior evaluation')
    output.parent.mkdir(parents=True,exist_ok=True);atomic_json(output,report)
    return dict(report=output.relative_to(ROOT).as_posix(),source=manifest['source'],
                correct={k:v.get('correct') for k,v in report['branches'].items()})


def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare-plan','run-hardware','run-simulator','classify'):mode.add_argument('--'+name,action='store_true')
    for name in ('cpu-session','map-session','plan','session','output'):parser.add_argument('--'+name)
    args=parser.parse_args()
    if args.prepare_plan:result=prepare(repo_path(args.cpu_session),repo_path(args.map_session),repo_path(args.output))
    elif args.classify:result=classify(repo_path(args.session),repo_path(args.output))
    else:result=acquire(repo_path(args.plan),'hardware' if args.run_hardware else 'simulator')
    print(json.dumps(result,allow_nan=False));return 2 if result.get('status')=='failed' else 0


if __name__=='__main__':raise SystemExit(main())
