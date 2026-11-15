"""Read-only MemriBoard study: basis-only pairing, frozen map, independent heldout."""
from __future__ import annotations
import argparse
from configparser import ConfigParser
import contextlib
from datetime import datetime,timezone
import html
import json
import logging
from pathlib import Path
import random
import shutil
import sys
import time
from typing import TYPE_CHECKING
import uuid

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from experiment_common.durable import Writer,atomic_json,sha
from experiment_common.pairing_readout import SPEC,WideCollector,candidates,diagnose
from experiment_common.runtime import INI,SIM_INI,use_local_dependencies
from experiment_common.selective_metrology import Source,object_dict
from experiment_common import terminal_publication
from read_all import EXPECTED_BOARD,EXPECTED_SOURCES
if TYPE_CHECKING:
    from manager.board import Connector

BASE=ROOT/'results/mnist_pairing_readout_20261115'
CANDIDATES=Path(__file__).with_name('candidates.json')
ASSETS=ROOT/'experiments/mnist_tiled_transfer_20261022/assets/learned_conductances.json'


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(source: Source,smoke: bool,publish: bool) -> tuple[dict[str,object],int]:
    cells=candidates(CANDIDATES)
    if len(cells)!=163:
        raise ValueError('Pinned full candidate scope must be163')
    if smoke:
        addresses={(0,0),(1,0),(0,2),(1,2),(6,30),(7,30)}
        cells=tuple(c for c in cells if (c.wl,c.bl) in addresses)
        if len(cells)!=6:
            raise ValueError('Pinned smoke scope changed')
    config=ConfigParser()
    config.read_string((ROOT/INI).read_text(encoding='utf-8-sig'))
    if config.get('board','board_type')!='rp5_fpga_python':
        raise ValueError('INI driver mismatch')
    for key,value in EXPECTED_BOARD.items():
        if config.getfloat('board',key)!=value:
            raise ValueError('Unexpected INI electrical setting: '+key)
    use_local_dependencies(prefer_environment=True)
    import numpy as np
    from manager.board import Connector
    from manager.service import converters
    payload=object_dict(json.loads(ASSETS.read_text(encoding='utf-8')))
    goals=np.asarray(payload['conductances_s'],dtype=np.float64)
    if goals.shape!=(64,25,2) or not np.isfinite(goals).all() or np.any(goals<=0):
        raise ValueError('Frozen target scale invalid')
    if shutil.disk_usage(ROOT).free<3*1024**3:
        raise ValueError('Need3GiB free')
    sid=str(uuid.uuid4());out=BASE/sid;out.mkdir(parents=True)
    log=ROOT/'.tmp'/f'publish-pairing-{sid}.log'
    if publish:
        terminal_publication.preflight(ROOT,log)
    protocol={**SPEC,'smoke':smoke,'candidate_count':len(cells),'planned_basis_calls':len(cells)*3*3*44,
              'candidate_sha256':sha(CANDIDATES),'source_targets_sha256':sha(ASSETS)}
    paths=[Path(__file__),CANDIDATES,ASSETS,ROOT/INI,ROOT/SIM_INI,*sorted((ROOT/'experiment_common').glob('*.py'))]
    paths.extend(ROOT/'vendor/MemriBoard'/p for p in EXPECTED_SOURCES if not p.startswith('MemriCORE/'))
    paths.extend(ROOT/'vendor/MemriBoard/simulator'/p for p in ('src.py','memristor.py'))
    manifest: dict[str,object]=dict(session_id=sid,source=source,status='failed',error=None,started_utc=now(),
        protocol=protocol,pulses=0,source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in paths})
    atomic_json(out/'start_manifest.json',manifest)
    atomic_json(out/'protocol.json',protocol)
    shutil.copyfile(CANDIDATES,out/'candidates.json')
    shutil.copyfile(ROOT/INI,out/'board.ini')
    print(f'Read-only pairing session {sid}; source={source}; cells={len(cells)}',flush=True)
    connector: Connector|None=None
    reader: WideCollector|None=None
    writer: Writer|None=None
    lock=ROOT/'results/read_all/.hardware.lock';locked=False
    started=time.monotonic();metrics: dict[str,object]={}
    def progress(message: str) -> None:
        atomic_json(out/'phase_progress.json',dict(message=message,calls=reader.calls if reader else 0,
            elapsed_seconds=time.monotonic()-started,updated_utc=now()))
        print(message,flush=True)
    try:
        if source=='hardware':
            lock.parent.mkdir(parents=True,exist_ok=True)
            with lock.open('x',encoding='utf-8') as stream:
                stream.write(sid+'\n')
            locked=True
        writer=Writer(out/'raw',maximum_expanded_bytes=512*1024**2,minimum_free_bytes=2*1024**3)
        with (out/'lifecycle_output.txt').open('w',encoding='utf-8') as lifecycle:
            with contextlib.redirect_stdout(lifecycle),contextlib.redirect_stderr(lifecycle):
                serial=''
                if source=='simulator':
                    from simulator.src import create_crossbar_array
                    random.seed(20261115);np.random.seed(20261115)
                    serial=str(out/'array');create_crossbar_array(serial,32,8)
                    shutil.copyfile(Path(serial+'.cb'),out/'array_initial.cb')
                    shutil.copyfile(ROOT/SIM_INI,out/'simulator_settings.ini')
                connector=Connector(silent=True,logger=logging.getLogger(__name__),cb_type='real' if source=='hardware' else 'simulator',
                    board_type=config.get('board','board_type'),task_format='SI',config=config,crossbar_serial=serial)
                if not connector.open_port():
                    raise ValueError('Connector open_port failed')
            reader=WideCollector(connector,converters,writer,source)
            metrics=diagnose(reader,cells,out,float(goals.min()),float(goals.max()),
                config.getfloat('board','vol_ref_adc')/(2**config.getint('board','adc_bit')-1),
                config.getfloat('board','sum_gain'),progress)
            manifest['status']='completed'
    except (Exception,KeyboardInterrupt) as exc:
        manifest.update(status='failed',error=type(exc).__name__+': '+str(exc))
    finally:
        if connector is not None:
            try:
                with (out/'lifecycle_output.txt').open('a',encoding='utf-8') as lifecycle:
                    with contextlib.redirect_stdout(lifecycle),contextlib.redirect_stderr(lifecycle):
                        if not connector.close_port():
                            raise ValueError('Connector close_port failed')
                manifest['port_closed']=True
            except Exception as exc:
                manifest.update(status='failed',close_error=str(exc))
        if writer is not None:
            try:
                writer.finish()
            except Exception as exc:
                manifest.update(status='failed',raw_finish_error=str(exc))
        if locked:
            if lock.read_text(encoding='utf-8').strip()!=sid:
                raise ValueError('Lock ownership changed')
            lock.unlink()
    summary: dict[str,object]={**metrics,'session_id':sid,'source':source,'status':manifest['status'],'error':manifest['error'],
        'impacts':reader.calls if reader else 0,'pulses':0,'smoke':smoke,'elapsed_seconds':time.monotonic()-started}
    atomic_json(out/'summary.json',summary)
    (out/'map.html').write_text('<!doctype html><meta charset="utf-8"><h1>Read-only pairing</h1><pre>'+
        html.escape(json.dumps(summary,indent=2,ensure_ascii=False))+'</pre>',encoding='utf-8')
    manifest.update(finished_utc=now(),impacts=summary['impacts'],result_sha256={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()})
    atomic_json(out/'manifest.json',manifest)
    code=0 if manifest['status']=='completed' else 2
    if publish:
        try:
            summary['publication']=terminal_publication.publish(ROOT,out,log=log)
        except Exception as exc:
            summary['publication']=dict(status='pending',error_type=type(exc).__name__);code=code or 3
    return summary,code


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--dry-run',action='store_true')
    modes.add_argument('--run-simulator',action='store_true')
    modes.add_argument('--run-hardware',action='store_true')
    modes.add_argument('--publish-session')
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--publish',action='store_true')
    args=parser.parse_args()
    if args.dry_run:
        print(json.dumps(SPEC,indent=2));return 0
    if args.publish_session:
        sid=str(uuid.UUID(args.publish_session))
        print(json.dumps(terminal_publication.publish(ROOT,BASE/sid,log=ROOT/'.tmp'/f'publish-pairing-{sid}.log')));return 0
    result,code=run('hardware' if args.run_hardware else 'simulator',args.smoke,args.publish)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return code


if __name__=='__main__':
    raise SystemExit(main())
