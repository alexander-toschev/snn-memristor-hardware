"""Foreground finite stage continuation; only public MemriBoard performs physical I/O."""
import argparse
import contextlib
import gzip
import importlib.metadata
import json
import logging
from pathlib import Path
import platform
import random
import shutil
import sys
import uuid

ROOT=Path(__file__).resolve().parents[2]
for folder in (ROOT,ROOT/'.deps'):
    if folder.is_dir():sys.path.insert(0,str(folder))
from experiments.cifar_resumable_20261019 import contract as c,durable,collector
from experiments.cifar_full_case0_20261019 import collector as physical,plan


def run_segment(source,stage,config,snapshot,hashes,protocol,plan_path,interrupt,remaining_raw_bytes,grant,process_loss=False):
    from read_all import native_connector,now
    from experiment_hardware import launch_provenance
    for name,pin in hashes.items():
        if c.sha(c.local(name))!=pin:raise ValueError('Acquisition source changed before stage: '+name)
    sid=str(uuid.uuid4());output=ROOT/'results/cifar_resumable_20261019'/sid
    output.mkdir(parents=True,exist_ok=False)
    protocol_bytes=(ROOT/'experiments/cifar_full_case0_20261019/protocol.json').read_bytes()
    (output/'protocol.json').write_bytes(protocol_bytes)
    (output/'board.ini').write_bytes(snapshot)
    (output/'continuation_plan.json').write_bytes(plan_path.read_bytes())
    if c.sha(c.local(grant['path']))!=grant['sha256']:raise ValueError('Execution grant changed between stages')
    (output/'execution_grant.json').write_bytes(c.local(grant['path']).read_bytes())
    manifest=dict(schema='cifar-stage-segment-v1',session_id=sid,source=source,scope='full',
        acquisition_kind='stage_segment',stage_range=[stage,stage+1],status='failed',
        started_utc=now(),spec_sha256=c.digest(c.SPEC),protocol_sha256=c.PHYSICAL_PROTOCOL_SHA,
        continuation_plan_sha256=c.sha(plan_path),source_sha256=hashes,
        execution_grant_path=grant['path'],execution_grant_sha256=grant['sha256'],
        launcher_sha256=c.sha(__file__),invocation=launch_provenance(None),
        execution_environment=dict(python=sys.version,platform=platform.platform(),
            packages={name:importlib.metadata.version(name) for name in ('numpy','SQLAlchemy','greenlet')}),
        fresh_connector_each_stage=True,simulator_initial_state_recreated_each_stage=source=='simulator',
        controlled_interruption_after_positions=17 if interrupt else None,
        controlled_process_loss=process_loss,
        minimum_reusable_unit='whole independently audited stage')
    durable.atomic_json(output/'start_manifest.json',manifest)
    writer=durable.Writer(output/'raw',maximum_expanded_bytes=min(2*1024**3,remaining_raw_bytes))
    connector=None;runner=None
    try:
        # Capture native driver's per-impact printing losslessly, without terminal flooding.
        with gzip.open(output/'native_stdout.txt.gz','wt',encoding='utf-8',compresslevel=1) as driverlog:
            with contextlib.redirect_stdout(driverlog):
                if source=='hardware':connector=native_connector(config)
                else:
                    from experiments.unn_rev3_20261014.paths import use_local_dependencies
                    use_local_dependencies()
                    import numpy as np
                    from manager.board import Connector
                    from simulator.src import create_crossbar_array
                    random.seed(protocol['seed']);np.random.seed(protocol['seed'])
                    serial=output/'array';create_crossbar_array(str(serial),32,8)
                    manifest['array_initial_sha256']=c.sha(serial.with_suffix('.cb'))
                    from cell_calibration import SIM_INI
                    (output/'simulator_settings.ini').write_bytes(c.local(SIM_INI).read_bytes())
                    connector=Connector(silent=True,logger=logging.getLogger(__name__),cb_type='simulator',
                        board_type=config.get('board','board_type'),task_format='SI',config=config,crossbar_serial=str(serial))
                from manager.service import converters
                image,weights=physical.load_assets()
                runner=collector.Collector(connector,converters,writer,protocol,c.PHYSICAL_PROTOCOL_SHA,source,
                    stage=stage,interrupt_after_positions=17 if interrupt else None,process_loss=process_loss)
                # Only progress uses original stdout; native Connector stays captured.
                runner.progress_stream=sys.__stdout__
                manifest.update(runner.run_stage(image,weights))
    except (Exception,KeyboardInterrupt) as exc:
        manifest['status']='failed';manifest['error']=type(exc).__name__+': '+str(exc)
    finally:
        if connector is not None and not (runner is not None and runner.lifecycle_assumed):
            try:
                closed=connector.close_port()
                manifest['pre_run_cleanup_close_result']=physical.json_safe(closed)
                if not closed:raise ConnectionError('Pre-run Connector.close_port failed')
            except Exception as exc:
                manifest['status']='failed';manifest['pre_run_cleanup_error']=repr(exc)
        try:writer.finish()
        except Exception as exc:
            manifest['status']='failed';manifest['raw_finalize_error']=repr(exc)
        manifest['finished_utc']=now()
        manifest['raw_expanded_bytes']=writer.expanded
        manifest['result_sha256']={p.relative_to(output).as_posix():c.sha(p) for p in output.rglob('*') if p.is_file()}
        durable.atomic_json(output/'manifest.json',manifest)
    print(json.dumps(dict(session=output.relative_to(ROOT).as_posix(),stage=stage,status=manifest['status'],error=manifest.get('error'))),flush=True)
    return output,manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-run',action='store_true');mode.add_argument('--run-simulator',action='store_true');mode.add_argument('--run-hardware',action='store_true')
    parser.add_argument('--plan',required=True,help='Repository-relative reviewed finite continuation plan')
    parser.add_argument('--preflight-source',choices=('simulator','hardware'),default='hardware')
    faults=parser.add_mutually_exclusive_group()
    faults.add_argument('--qualify-interruption',action='store_true',help='Software graceful-failure check only, not abrupt qualification')
    faults.add_argument('--qualify-process-loss',action='store_true',help='SIM only: os._exit86 at stage1 after17positions, durable16, no close/finalmanifest')
    args=parser.parse_args(argv)
    source=args.preflight_source if args.dry_run else 'hardware' if args.run_hardware else 'simulator'
    plan_path=c.local(args.plan);continuation=json.loads(plan_path.read_text())
    first,stop=c.validate_plan(continuation,source)
    if (args.qualify_interruption or args.qualify_process_loss) and (source!='simulator' or continuation['purpose']!='qualification' or first>1):raise ValueError('Invalid qualification interruption request')
    hashes=c.verify_pins()
    from cell_calibration import check_sources,LOCK
    config,snapshot,native_hashes=check_sources(source);hashes.update(native_hashes)
    protocol=json.loads((ROOT/'experiments/cifar_full_case0_20261019/protocol.json').read_text())
    physical.validate_protocol(protocol,c.PHYSICAL_PROTOCOL_SHA)
    image,weights=physical.load_assets()
    bounds=[collector.stage_bounds(image,s) for s in range(first,stop)]
    # Reserve two copies (working tree + eventual Git LFS), plus 8GiB free.
    raw_budget=min(2*(stop-first),32)*1024**3
    required=2*raw_budget+8*1024**3
    grant=c.execution_grant(source)
    checked=dict(source=source,stage_range=[first,stop],spec_sha256=c.digest(c.SPEC),registered=grant is not None,
        maximum_new_impacts=sum(b['maximum_impacts'] for b in bounds),maximum_new_pulses=sum(b['maximum_pulses'] for b in bounds),
        required_free_bytes=required,available_bytes=shutil.disk_usage(ROOT).free,source_sha256=hashes,
        connector_opened=False,prior_attempt_impacts='See prefix verified lower bound; lost buffered tail unknown, prior protocol upper15000000')
    if args.dry_run:print(json.dumps(checked,sort_keys=True));return 0
    if not checked['registered']:raise ValueError('Resumable protocol not registered after independent review/nativeSIM qualification')
    if checked['available_bytes']<required:raise ValueError('Insufficient disk for bounded stage raw plus LFS duplicate/reserve')
    for path in Path(__file__).parent.glob('*.py'):hashes[path.relative_to(ROOT).as_posix()]=c.sha(path)
    hashes['experiments/cifar_full_case0_20261019/protocol.json']=c.sha(ROOT/'experiments/cifar_full_case0_20261019/protocol.json')
    hashes['weights/conv1_weights.npy']=physical.WEIGHTS_SHA;hashes[plan.FIXTURE]=plan.FIXTURE_SHA
    lock=c.local(LOCK);token=str(uuid.uuid4());locked=False
    if source=='hardware':
        lock.parent.mkdir(parents=True,exist_ok=True)
        with lock.open('x',encoding='utf-8') as f:f.write(token)
        locked=True
    try:
        remaining_raw_bytes=raw_budget
        for stage in range(first,stop):
            fault=(args.qualify_interruption or args.qualify_process_loss) and stage==1
            output,result=run_segment(source,stage,config,snapshot,hashes,protocol,plan_path,fault,remaining_raw_bytes,grant,args.qualify_process_loss and stage==1)
            remaining_raw_bytes-=result['raw_expanded_bytes']
            if result['status']!='completed':return 2
        return 0
    finally:
        if locked and lock.read_text()==token:lock.unlink()


if __name__=='__main__':raise SystemExit(main())
