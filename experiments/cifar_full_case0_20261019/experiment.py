"""Foreground MemriBoard collector entrypoint. Registered protocol required."""
import argparse
import hashlib
import json
import logging
from pathlib import Path
import random
import shutil
import sys
import uuid
import platform
import importlib.metadata

ROOT=Path(__file__).resolve().parents[2]
for directory in (ROOT,ROOT/'.deps'):
    if directory.is_dir():sys.path.insert(0,str(directory))

from experiments.cifar_full_case0_20261019 import collector,plan,trace


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--dry-run',action='store_true')
    modes.add_argument('--run-simulator',action='store_true')
    modes.add_argument('--run-hardware',action='store_true')
    parser.add_argument('--scope',choices=('smoke','full'),default='smoke')
    parser.add_argument('--preflight-source',choices=('hardware','simulator'),default='hardware',help='Source inspected by --dry-run; no port is opened')
    args=parser.parse_args(argv)
    protocol_path=Path(__file__).with_name('protocol.json')
    protocol_bytes=protocol_path.read_bytes();protocol=json.loads(protocol_bytes)
    frozen_sha=collector.digest(protocol)
    collector.validate_protocol(protocol,frozen_sha,require_approval=not args.dry_run)
    image,weights=collector.load_assets()
    source=args.preflight_source if args.dry_run else 'hardware' if args.run_hardware else 'simulator'
    from cell_calibration import check_sources,SIM_INI,LOCK
    from read_all import native_connector,local_path,now,sha256,save_json
    from experiment_hardware import launch_provenance
    pins={'cell_calibration.py':'eff0b2f72586f6a6659a13fba410c6ad0d564bd95d3b83e24b13cf739121c2e8',
          'read_all.py':'b30ae2f4dd3cd166dbc5cefb07b558127de2bbdba1f6674c40b1b45f4dd74d5f',
          'native_conversions.py':'2ad2e9c057bb14979b083a6e846e8c417c080a6c5273d848e31f893b7970253e',
          'experiment_hardware.py':'059dc4f827943a9df631631f52e776b57182afc9d36db0c1ad5246427f73548c'}
    for name,pin in pins.items():
        if sha256(ROOT/name)!=pin:raise ValueError('Unreviewed helper: '+name)
    config,snapshot,hashes=check_sources(source)
    hashes.update(pins)
    if args.dry_run:
        print(json.dumps(dict(protocol_sha256=frozen_sha,registered=frozen_sha in collector.APPROVED_PROTOCOL_SHA256,
            scope=args.scope,source=source,hardware_access=False,full_mode10=5133568,
            full_inference_mode7=6742400,calibration_mode10=502272,calibration_mode7=466608,
            maximum_programming_impacts=1612800,whole_run_maximum_impacts=14457648,
            raw_expanded_budget_bytes=32*1024**3,minimum_free_before_full_bytes=72*1024**3,
            native_sources_verified=hashes,connector_opened=False)))
        return 0
    free=shutil.disk_usage(ROOT).free
    required=(72 if args.scope=='full' else 12)*1024**3
    if free<required:raise ValueError('Insufficient disk for raw, Git LFS copy and reserve')
    for path in Path(__file__).parent.glob('*.py'):
        hashes[path.relative_to(ROOT).as_posix()]=sha256(path)
    hashes[protocol_path.relative_to(ROOT).as_posix()]=sha256(protocol_path)
    hashes['weights/conv1_weights.npy']=collector.WEIGHTS_SHA
    hashes[plan.FIXTURE]=plan.FIXTURE_SHA
    lock=local_path(LOCK);locked=False
    if source=='hardware':
        lock.parent.mkdir(parents=True,exist_ok=True)
        with lock.open('x',encoding='utf-8') as stream:stream.write('full CIFAR '+now()+'\n')
        locked=True
    try:
        sid=str(uuid.uuid4());output=ROOT/'results/cifar_full_case0_20261019'/sid
        output.mkdir(parents=True,exist_ok=False)
        (output/'protocol.json').write_bytes(protocol_bytes);(output/'board.ini').write_bytes(snapshot)
        manifest=dict(session_id=sid,source=source,scope=args.scope,status='failed',started_utc=now(),
            protocol_sha256=frozen_sha,source_sha256=hashes,invocation=launch_provenance(argv))
        packages={}
        for name in ('numpy','SQLAlchemy','greenlet'):
            try:packages[name]=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:packages[name]=None
        manifest['execution_environment']=dict(python=sys.version,platform=platform.platform(),packages=packages)
        save_json(output/'start_manifest.json',manifest)
        try:
            if source=='hardware':connector=native_connector(config)
            else:
                from experiments.unn_rev3_20261014.paths import use_local_dependencies
                use_local_dependencies()
                import numpy as np
                from manager.board import Connector
                from simulator.src import create_crossbar_array
                random.seed(protocol['seed']);np.random.seed(protocol['seed'])
                serial=output/'array';create_crossbar_array(str(serial),32,8)
                (output/'simulator_settings.ini').write_bytes(local_path(SIM_INI).read_bytes())
                manifest['array_initial_sha256']=sha256(serial.with_suffix('.cb'))
                connector=Connector(silent=True,logger=logging.getLogger(__name__),cb_type='simulator',
                    board_type=config.get('board','board_type'),task_format='SI',config=config,crossbar_serial=str(serial))
            from manager.service import converters
            writer=trace.Writer(output/'raw')
            result=collector.Collector(connector,converters,writer,protocol,frozen_sha,source).run(image,weights,args.scope)
            manifest.update(result)
        except (Exception,KeyboardInterrupt) as exc:
            manifest['error']=type(exc).__name__+': '+str(exc)
        finally:
            manifest['finished_utc']=now()
            manifest['result_sha256']={p.relative_to(output).as_posix():sha256(p) for p in output.rglob('*') if p.is_file()}
            save_json(output/'manifest.json',manifest)
        print(json.dumps(dict(session=output.relative_to(ROOT).as_posix(),status=manifest['status'],error=manifest.get('error'))))
        return 0 if manifest['status']=='completed' else 2
    finally:
        if locked:lock.unlink()


if __name__=='__main__':
    raise SystemExit(main())
