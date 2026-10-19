"""Separate bounded conductance STDP pilot. Explicit driver selection; review gated."""
import argparse
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
import platform
import random
import sys
import uuid

ROOT=Path(__file__).resolve().parents[2]
for directory in (ROOT,ROOT/'.deps'):
    if directory.is_dir():sys.path.insert(0,str(directory))
from experiments.conductance_stdp_20261019 import core,raw

APPROVED_PROTOCOL_SHA256={
    'simulator':frozenset({'72f6b41aec3a22ed7286ec6e063d125ed2c23cc3220b3b87a118b0f8a954eb68'}),
    'hardware':frozenset(),  # Requires separate review of actual native SIM evidence.
}
REVIEWED_COMPONENT_SHA256={
    'core.py':'005586394be54aad0ac7d116c729d7b733628161ef7a442975ee9b94209aa86f',
    'raw.py':'bb52dad606db21cf57cea6aa908882cac07fa5955b47c4c9b1e2c85329bba8a7',
}
PINS={
    'cell_calibration.py':'eff0b2f72586f6a6659a13fba410c6ad0d564bd95d3b83e24b13cf739121c2e8',
    'read_all.py':'b30ae2f4dd3cd166dbc5cefb07b558127de2bbdba1f6674c40b1b45f4dd74d5f',
    'native_conversions.py':'2ad2e9c057bb14979b083a6e846e8c417c080a6c5273d848e31f893b7970253e',
    'experiment_hardware.py':'059dc4f827943a9df631631f52e776b57182afc9d36db0c1ad5246427f73548c',
}


def protocol_sha():return hashlib.sha256(json.dumps(core.SPEC,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-run',action='store_true')
    mode.add_argument('--run-simulator',action='store_true')
    mode.add_argument('--run-hardware',action='store_true')
    parser.add_argument('--preflight-source',choices=('hardware','simulator'),default='hardware')
    args=parser.parse_args(argv)
    source=args.preflight_source if args.dry_run else 'hardware' if args.run_hardware else 'simulator'
    for name,pin in REVIEWED_COMPONENT_SHA256.items():
        if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()!=pin:
            raise ValueError('Reviewed STDP component changed: '+name)
    if not args.dry_run and protocol_sha() not in APPROVED_PROTOCOL_SHA256[source]:
        raise ValueError('STDP code review and protocol registration required before native launch')
    from cell_calibration import check_sources,SIM_INI,LOCK
    from read_all import native_connector,local_path,now,sha256,save_json
    from experiment_hardware import launch_provenance
    for name,pin in PINS.items():
        if sha256(ROOT/name)!=pin:raise ValueError('Pinned helper changed: '+name)
    config,snapshot,hashes=check_sources(source);hashes.update(PINS)
    for path in Path(__file__).parent.iterdir():
        if path.is_file() and path.suffix in ('.py','.md'):
            hashes[path.relative_to(ROOT).as_posix()]=sha256(path)
    if args.dry_run:
        print(json.dumps(dict(source=source,protocol=core.SPEC,protocol_sha256=protocol_sha(),
            registered=protocol_sha() in APPROVED_PROTOCOL_SHA256[source],source_sha256=hashes,
            connector_opened=False,board_access=False)))
        return 0
    lock=local_path(LOCK);locked=False
    if source=='hardware':
        lock.parent.mkdir(parents=True,exist_ok=True)
        with lock.open('x',encoding='utf-8') as stream:stream.write('STDP '+now()+'\n')
        locked=True
    try:
        sid=str(uuid.uuid4());output=ROOT/'results/conductance_stdp_20261019'/sid
        output.mkdir(parents=True,exist_ok=False)
        save_json(output/'protocol.json',core.SPEC);(output/'board.ini').write_bytes(snapshot)
        manifest=dict(session_id=sid,source=source,status='failed',started_utc=now(),
            protocol_sha256=protocol_sha(),source_sha256=hashes,invocation=launch_provenance(argv),
            scientific_scope='algorithmic boundary-updated STDP; not intrinsic delta-t STDP or classification accuracy',
            physical_pulse_duration_s=None,seed=core.SPEC['seed'])
        packages={}
        for name in ('numpy','SQLAlchemy','greenlet'):
            try:packages[name]=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:packages[name]=None
        manifest['environment']=dict(python=sys.version,platform=platform.platform(),packages=packages)
        writer=None;runner=None
        try:
            if source=='hardware':connector=native_connector(config)
            else:
                from experiments.unn_rev3_20261014.paths import use_local_dependencies
                use_local_dependencies()
                import numpy as np
                from manager.board import Connector
                from simulator.src import create_crossbar_array
                random.seed(core.SPEC['seed']);np.random.seed(core.SPEC['seed'])
                serial=output/'array';create_crossbar_array(str(serial),32,8)
                (output/'simulator_settings.ini').write_bytes(local_path(SIM_INI).read_bytes())
                manifest['array_initial_sha256']=sha256(serial.with_suffix('.cb'))
                connector=Connector(silent=True,logger=logging.getLogger(__name__),cb_type='simulator',
                    board_type=config.get('board','board_type'),task_format='SI',config=config,crossbar_serial=str(serial))
            save_json(output/'start_manifest.json',manifest)
            from manager.service import converters
            writer=raw.Writer(output/'raw');runner=core.Collector(connector,converters,writer,source)
            result=runner.run();save_json(output/'observations.json',result)
            manifest.update(status=result['status'],impacts=result['impacts'],pulses=result['pulses'],decisions=result['decisions'])
        except (Exception,KeyboardInterrupt) as exc:
            manifest['error']=type(exc).__name__+': '+str(exc)
        finally:
            if writer is not None:
                try:writer.finish()
                except Exception as exc:manifest.update(status='failed',raw_finish_error=repr(exc))
            if runner is not None:
                manifest.update(impacts=runner.calls,pulses=runner.pulses,decisions=runner.decisions)
                if manifest['status']!='completed':
                    save_json(output/'partial_observations.json',dict(checkpoints=runner.checkpoints,windows=runner.windows,updates=runner.updates))
            manifest['finished_utc']=now()
            manifest['result_sha256']={p.relative_to(output).as_posix():sha256(p) for p in output.rglob('*') if p.is_file()}
            save_json(output/'manifest.json',manifest)
        print(json.dumps(dict(session=output.relative_to(ROOT).as_posix(),status=manifest['status'],error=manifest.get('error'))))
        return 0 if manifest['status']=='completed' else 2
    finally:
        if locked:lock.unlink()


if __name__=='__main__':raise SystemExit(main())
