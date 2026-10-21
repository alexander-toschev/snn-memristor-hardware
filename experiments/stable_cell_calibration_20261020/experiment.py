"""Fresh stable-cell calibration: bounded A/B/C on Pi; readout and independent audit on PC."""
import argparse
from configparser import ConfigParser
import contextlib
import importlib.metadata
import json
import logging
from pathlib import Path
import random
import shutil
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT/'.deps'):
    if directory.is_dir():sys.path.insert(0,str(directory))
from experiments.stable_cell_calibration_20261020 import core, metrics, plan
from experiments.cifar_resumable_20261019.durable import Writer, atomic_json, sha
from read_all import EXPECTED_BOARD, EXPECTED_SOURCES, local_path, now

PUBLIC_PINS = {f'vendor/MemriBoard/{k}':v for k,v in EXPECTED_SOURCES.items()
               if not k.startswith('MemriCORE/')}
PUBLIC_PINS.update({
    'vendor/MemriBoard/simulator/src.py':'d41edc5c0464e15d44a23a1bb628e8025194e518d914635c1a14dd46ea2d4cc2',
    'vendor/MemriBoard/simulator/memristor.py':'bb9c43d0f254c512c21902a64b5414e05f63f1deecf10016e5b97bd086050aca',
})
INI='configs/raspberry_board.ini'
SIM_INI='vendor/MemriBoard/simulator/simulator_settings.ini'
DESIGN='docs/plans/FRESH_STABLE_CELL_CALIBRATION_20261020.md'
HELPERS=('read_all.py','experiments/cifar_resumable_20261019/durable.py',
         'experiments/unn_rev3_20261014/paths.py')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def preflight(source):
    config=ConfigParser();config.read_string((ROOT/INI).read_text(encoding='utf-8-sig'))
    if config.get('board','board_type')!='rp5_fpga_python':raise ValueError('Unexpected INI board driver')
    other=ConfigParser();other.read_string((ROOT/SIM_INI).read_text(encoding='utf-8-sig'))
    for k,v in EXPECTED_BOARD.items():
        if config.getfloat('board',k)!=v or other.getfloat('board',k)!=v:
            raise ValueError('Board/native simulator setting mismatch: '+k)
    # Record public bytes for reproducibility, without approval registries or private-driver inspection.
    paths=list(PUBLIC_PINS)+list(HELPERS)+[INI,SIM_INI,DESIGN]
    paths += [p.relative_to(ROOT).as_posix() for p in Path(__file__).parent.iterdir()
              if p.suffix in ('.py','.md')]
    # Old durable import is now a compatibility alias; record actual shared
    # implementations for new runs, never alter previously saved manifests.
    paths += [p.relative_to(ROOT).as_posix() for p in (ROOT/'experiment_common').glob('*.py')]
    return config,{name:sha(ROOT/name) for name in sorted(set(paths))}


def acquire(source):
    config,hashes=preflight(source)
    parent=ROOT/'results/stable_cell_calibration_20261020';parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(parent).free<plan.SPEC['initial_free_bytes']:raise ValueError('Insufficient free disk space')
    sid=str(uuid.uuid4());out=parent/sid;out.mkdir()
    lock=ROOT/'results/read_all/.hardware.lock';locked=False
    writer=runner=None
    manifest=dict(session_id=sid,source=source,status='failed',
                  started_utc=now(),protocol_sha256=plan.digest(plan.SPEC),source_sha256=hashes,
                  private_runtime_status=plan.SPEC['private_runtime_status'],physical_pulse_duration_s=None,
                  python=sys.version,packages={},source_attestation='public local bytes; private runtime unverified')
    try:
        if source=='hardware':
            lock.parent.mkdir(parents=True,exist_ok=True)
            with lock.open('x',encoding='utf-8') as stream:stream.write(sid+'\n')
            locked=True
        atomic_json(out/'protocol.json',plan.SPEC);(out/'board.ini').write_bytes((ROOT/INI).read_bytes())
        for name in ('numpy','SQLAlchemy','greenlet'):
            try:manifest['packages'][name]=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:manifest['packages'][name]=None
        from experiments.unn_rev3_20261014.paths import use_local_dependencies
        use_local_dependencies()
        from manager.board import Connector
        from manager.service import converters
        writer=Writer(out/'raw',limit=plan.SPEC['raw_chunk_bytes'],
                      maximum_expanded_bytes=plan.SPEC['raw_maximum_bytes'],minimum_free_bytes=plan.SPEC['reserve_bytes'])
        with (out/'lifecycle_output.txt').open('w',encoding='utf-8',newline='\n') as native_log:
            with contextlib.redirect_stdout(native_log),contextlib.redirect_stderr(native_log):
                kwargs={}
                if source=='simulator':
                    import numpy as np
                    from simulator.src import create_crossbar_array
                    random.seed(plan.SPEC['seed']);np.random.seed(plan.SPEC['seed'])
                    serial=out/'array'
                    create_crossbar_array(str(serial),32,8)
                    shutil.copyfile(serial.with_suffix('.cb'),out/'array_initial.cb')
                    manifest['array_initial_sha256']=sha(out/'array_initial.cb')
                    (out/'simulator_settings.ini').write_bytes((ROOT/SIM_INI).read_bytes())
                    kwargs['crossbar_serial']=str(serial)
                atomic_json(out/'start_manifest.json',manifest)
                connector=Connector(silent=True,logger=logging.getLogger(__name__),
                    cb_type='real' if source=='hardware' else 'simulator',
                    board_type=config.get('board','board_type'),task_format='SI',config=config,**kwargs)
                runner=core.Collector(connector,converters,writer,source)
                result=runner.run()
        atomic_json(out/'observations.json',result)
        for split in ('before','fit','selection','heldout','after'):
            if split in runner.calibration:atomic_json(out/('C_'+split+'.json'),runner.calibration[split])
        atomic_json(out/'map.json',dict(cells=result['cells'],selection=result['selection'],
                    pair_qualification=result['pair_qualification'],calibration=result['calibration'],
                    qualified=result['qualified'],diagnostic_best_available=result['diagnostic_best_available'],
                    state_context='checkerboard recipes at .05V; not CIFAR learned weights'))
        manifest.update(status=result['status'],error=result['error'],impacts=result['impacts'],pulses=result['pulses'])
    except (Exception,KeyboardInterrupt) as exc:
        manifest.update(status='failed',error=type(exc).__name__+': '+str(exc))
    finally:
        if writer:
            try:writer.finish()
            except Exception as exc:manifest.update(status='failed',raw_finish_error=repr(exc))
        if runner:
            manifest.update(impacts=runner.calls,pulses=runner.pulses)
            if not (out/'observations.json').exists():atomic_json(out/'observations.json',runner.result())
            if not (out/'map.json').exists():
                result=runner.result()
                atomic_json(out/'map.json',dict(cells=result['cells'],status=result['status'],
                                              qualified=False,error=result['error']))
        manifest['finished_utc']=now()
        manifest['result_sha256']={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()}
        atomic_json(out/'manifest.json',manifest)
        if locked:
            if lock.read_text(encoding='utf-8').strip()!=sid:raise ValueError('Hardware lock ownership changed')
            lock.unlink()
    return dict(session=out.relative_to(ROOT).as_posix(),status=manifest['status'],error=manifest.get('error'))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group(required=True)
    for name in ('dry-run','run-simulator','run-hardware'):
        modes.add_argument('--'+name,action='store_true')
    parser.add_argument('--preflight-source',choices=('hardware','simulator'),default='hardware')
    args=parser.parse_args(argv)
    if args.dry_run:
        _,hashes=preflight(args.preflight_source)
        result=dict(protocol=plan.SPEC,protocol_sha256=plan.digest(plan.SPEC),source_sha256=hashes,
                    connector_opened=False,source_approval_required=False)
    else:result=acquire('hardware' if args.run_hardware else 'simulator')
    print(json.dumps(result,allow_nan=False))
    return 2 if result.get('status')=='failed' else 0


if __name__=='__main__':raise SystemExit(main())
