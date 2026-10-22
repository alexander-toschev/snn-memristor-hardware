"""74518 read-only calls: all binary masks of 15 exact blocks, mean4 readout."""
from __future__ import annotations

import argparse
from configparser import ConfigParser
import contextlib
from datetime import datetime, timezone
from dataclasses import asdict
import json
import logging
from pathlib import Path
import random
import shutil
import sys
import time
from typing import TYPE_CHECKING
import uuid

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiment_common.durable import Writer, atomic_json, sha
from experiment_common.block_readout_validation import Collector, SPEC, summarize, render
from experiment_common.runtime import INI, SIM_INI, use_local_dependencies
from experiment_common.selective_metrology import Source
from experiment_common import terminal_publication
from read_all import EXPECTED_BOARD, EXPECTED_SOURCES

if TYPE_CHECKING:
    from manager.board import Connector

BASE = ROOT / 'results/block_readout_validation_20261022'


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def preflight() -> tuple[ConfigParser, dict[str, str]]:
    config = ConfigParser()
    config.read_string((ROOT / INI).read_text(encoding='utf-8-sig'))
    if config.get('board', 'board_type') != 'rp5_fpga_python':
        raise ValueError('Unexpected INI driver')
    for key, expected in EXPECTED_BOARD.items():
        if config.getfloat('board', key) != expected:
            raise ValueError('Unexpected electrical setting: ' + key)
    paths = [ROOT / INI, ROOT / SIM_INI, ROOT / 'read_all.py', ROOT / 'reports/2026-10-22/mvm-address-diagnostic/BLOCK_SELECTION.json']
    paths += list(Path(__file__).parent.glob('*.py'))
    paths += list((ROOT / 'experiment_common').glob('*.py'))
    paths += [ROOT / 'vendor/MemriBoard' / name for name in EXPECTED_SOURCES if not name.startswith('MemriCORE/')]
    paths += [ROOT / 'vendor/MemriBoard/simulator' / name for name in ('src.py', 'memristor.py')]
    return config, {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


def run(source: Source, publish: bool) -> tuple[dict[str, object], int]:
    config, hashes = preflight()
    BASE.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(BASE).free < 1024**3:
        raise ValueError('Need1GiBfree')
    sid = str(uuid.uuid4())
    log = ROOT / '.tmp' / ('publish-block-readout-' + sid + '.log')
    if publish:
        terminal_publication.preflight(ROOT, log)
    out = BASE / sid
    out.mkdir()
    lock = ROOT / 'results/read_all/.hardware.lock'
    locked = False
    connector: Connector | None = None
    writer: Writer | None = None
    collector: Collector | None = None
    started = time.monotonic()
    manifest: dict[str, object] = dict(session_id=sid, source=source, status='failed', error=None,
        started_utc=now(), protocol=SPEC, source_sha256=hashes, pulses=0,
        private_runtime_status='private_runtime_not_independently_reverified')
    print(f'Session {out.relative_to(ROOT).as_posix()}', flush=True)
    try:
        if source == 'hardware':
            lock.parent.mkdir(parents=True, exist_ok=True)
            with lock.open('x', encoding='utf-8') as stream:
                stream.write(sid + '\n')
            locked = True
        atomic_json(out / 'protocol.json', SPEC)
        atomic_json(out / 'start_manifest.json', manifest)
        shutil.copyfile(ROOT / INI, out / 'board.ini')
        use_local_dependencies(prefer_environment=True)
        from manager.board import Connector
        from manager.service import converters
        writer = Writer(out / 'raw', maximum_expanded_bytes=256*1024**2, minimum_free_bytes=512*1024**2)
        serial = ''
        with (out / 'lifecycle_output.txt').open('w', encoding='utf-8') as lifecycle:
            with contextlib.redirect_stdout(lifecycle), contextlib.redirect_stderr(lifecycle):
                if source == 'simulator':
                    import numpy as np
                    from simulator.src import create_crossbar_array
                    random.seed(20261022)
                    np.random.seed(20261022)
                    serial = str(out / 'array')
                    create_crossbar_array(serial, 32, 8)
                    shutil.copyfile(Path(serial + '.cb'), out / 'array_initial.cb')
                    shutil.copyfile(ROOT / SIM_INI, out / 'simulator_settings.ini')
                # Публичный GUI-путь: Connector выбирает real/simulator,
                # электрические параметры и аппаратный driver заданы INI.
                connector = Connector(silent=True, logger=logging.getLogger(__name__),
                    cb_type='real' if source == 'hardware' else 'simulator',
                    board_type=config.get('board', 'board_type'), task_format='SI', config=config,
                    crossbar_serial=serial)
                if not connector.open_port():
                    raise ValueError('Connector open_port failed')
            collector = Collector(connector, converters, writer, source)
            collector.diagnose(lambda message: print(message, flush=True))
            manifest['status'] = 'completed'
    except (Exception, KeyboardInterrupt) as exc:
        manifest.update(status='failed', error=type(exc).__name__ + ': ' + str(exc))
    finally:
        if connector is not None:
            try:
                with (out / 'lifecycle_output.txt').open('a', encoding='utf-8') as lifecycle:
                    with contextlib.redirect_stdout(lifecycle), contextlib.redirect_stderr(lifecycle):
                        if not connector.close_port():
                            raise ValueError('Connector close_port failed')
            except Exception as exc:
                manifest.update(status='failed', close_error=str(exc))
        if writer is not None:
            try:
                writer.finish()
            except Exception as exc:
                manifest.update(status='failed', raw_finish_error=str(exc))
        rows = collector.pairs if collector else []
        assessment = summarize(rows)
        atomic_json(out / 'observations.json', dict(pairs=[asdict(r) for r in rows], assessment=assessment, basis_sha256=collector.basis_sha256 if collector else None))
        summary: dict[str, object] = dict(session_id=sid, source=source, status=manifest['status'],
            error=manifest['error'], impacts=collector.calls if collector else 0, pulses=0,
            complete_pairs=len(rows), **assessment,
            cause_identified=False, programmable_range_validated=False,
            note='Stable current-state readout only; no programming/MNIST qualification; all selected block inputs retained.')
        atomic_json(out / 'summary.json', summary)
        (out / 'map.html').write_text(render(summary), encoding='utf-8')
        manifest.update(finished_utc=now(), elapsed_seconds=time.monotonic()-started, impacts=summary['impacts'])
        manifest['result_sha256'] = {p.relative_to(out).as_posix(): sha(p) for p in out.rglob('*') if p.is_file()}
        atomic_json(out / 'manifest.json', manifest)
        if locked:
            if lock.read_text(encoding='utf-8').strip() != sid:
                raise ValueError('Lock ownership changed')
            lock.unlink()
    code = 2 if manifest['status'] == 'failed' else 0
    publication: dict[str, object] | None = None
    if publish:
        try:
            publication = terminal_publication.publish(ROOT, out, log=log)
        except Exception as exc:
            publication = dict(status='pending', error_type=type(exc).__name__, local_log=log.relative_to(ROOT).as_posix())
            atomic_json(ROOT / '.tmp' / ('publication-pending-' + sid + '.json'), publication)
            code = code or 3
    return dict(session=out.relative_to(ROOT).as_posix(), status=manifest['status'], error=manifest['error'],
                impacts=summary['impacts'], pulses=0, complete_pairs=summary['complete_pairs'],
                elapsed_seconds=manifest['elapsed_seconds'], publication=publication), code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--dry-run', action='store_true')
    modes.add_argument('--run-simulator', action='store_true')
    modes.add_argument('--run-hardware', action='store_true')
    modes.add_argument('--publish-session', metavar='UUID')
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args(argv)
    if args.publish_session:
        sid = str(uuid.UUID(args.publish_session))
        result = terminal_publication.publish(ROOT, BASE/sid, log=ROOT/'.tmp'/('publish-block-readout-'+sid+'.log'))
        print(json.dumps(result))
        return 2 if result['acquisition_status'] == 'failed' else 0
    if args.dry_run:
        _, hashes = preflight()
        print(json.dumps(dict(protocol=SPEC, source_sha256=hashes, connector_opened=False)))
        return 0
    result, code = run('hardware' if args.run_hardware else 'simulator', args.publish)
    print(json.dumps(result, ensure_ascii=False))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
