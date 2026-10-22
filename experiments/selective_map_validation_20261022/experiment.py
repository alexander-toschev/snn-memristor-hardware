"""40 выбранных ячеек: только повторные чтения через публичный MemriBoard."""
from __future__ import annotations

import argparse
from configparser import ConfigParser
import contextlib
from datetime import datetime, timezone
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
from experiment_common.runtime import INI, SIM_INI, use_local_dependencies
from experiment_common.selective_metrology import Collector, Source, load_sample
from read_all import EXPECTED_BOARD, EXPECTED_SOURCES

if TYPE_CHECKING:
    from manager.board import Connector

BASE = ROOT / 'results/selective_map_validation_20261022'
SAMPLE = Path(__file__).parent / 'sample.json'
SPEC: dict[str, object] = dict(
    schema='selective-map-readonly-v1', sample_cells=40, repeats=10,
    roles=['initial', 'delay_0', 'delay_10', 'delay_60'],
    delay_origin='end of initial complete sample sweep', amplitude_v=.05,
    maximum_impacts=6400, maximum_pulses=0, maximum_wall_seconds=600,
    noise_limit='max(2ADC,0.05*historical_span)',
    drift_limit='max(3ADC,0.05*historical_span); relative own current initial',
    zero_limit='max(2ADC,3*SD(zero_after-zero_before)/sqrt(10))',
    programmable_range_validated=False, full_array_validated=False,
    old_quality_grade_promoted=False, private_runtime_status='private_runtime_not_independently_reverified',
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def preflight() -> tuple[ConfigParser, dict[str, str]]:
    # Проверяем электрические параметры, записываем public-source SHA только
    # как provenance. Нет private receipts/ручных разрешений на source hash.
    config = ConfigParser()
    config.read_string((ROOT / INI).read_text(encoding='utf-8-sig'))
    if config.get('board', 'board_type') != 'rp5_fpga_python':
        raise ValueError('Unexpected board_type in INI')
    for key, expected in EXPECTED_BOARD.items():
        if config.getfloat('board', key) != expected:
            raise ValueError('Unexpected electrical setting: ' + key)
    paths = [ROOT / INI, ROOT / SIM_INI, SAMPLE]
    paths += list(Path(__file__).parent.glob('*.py'))
    paths += list((ROOT / 'experiment_common').glob('*.py'))
    paths += [ROOT / 'read_all.py']
    paths += [ROOT / 'vendor/MemriBoard' / name for name in EXPECTED_SOURCES if not name.startswith('MemriCORE/')]
    paths += [ROOT / 'vendor/MemriBoard/simulator' / name for name in ('src.py', 'memristor.py')]
    return config, {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


def run(source: Source) -> dict[str, object]:
    config, hashes = preflight()
    sample, cells = load_sample(SAMPLE)
    BASE.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(BASE).free < 1024**3:
        raise ValueError('Need1GiBfree')
    sid = str(uuid.uuid4())
    out = BASE / sid
    out.mkdir()
    lock = ROOT / 'results/read_all/.hardware.lock'
    locked = False
    connector: Connector | None = None
    writer: Writer | None = None
    collector: Collector | None = None
    started = time.monotonic()
    manifest: dict[str, object] = dict(
        session_id=sid, source=source, status='failed', started_utc=now(),
        protocol=SPEC, sample_sha256=sha(SAMPLE), source_sha256=hashes,
        parent_session=sample['parent_session'], parent_map_sha256=sample['parent_map_sha256'],
        parent_quality_qualified=sample['parent_qualified'],
        parent_pair_mandatory_pass=sample['parent_pair_mandatory_pass'],
        private_runtime_status='private_runtime_not_independently_reverified',
        pulses=0, error=None, python=sys.version,
    )
    print(f'Session {out.relative_to(ROOT).as_posix()}', flush=True)
    try:
        if source == 'hardware':
            lock.parent.mkdir(parents=True, exist_ok=True)
            with lock.open('x', encoding='utf-8') as stream:
                stream.write(sid + '\n')
            locked = True
        # Выборка и старые grades сохранены ДО первого измерения.
        atomic_json(out / 'protocol.json', SPEC)
        shutil.copyfile(SAMPLE, out / 'sample.json')
        shutil.copyfile(ROOT / INI, out / 'board.ini')
        atomic_json(out / 'start_manifest.json', manifest)
        # Пакеты venv должны предшествовать repo .deps: на Windows последняя
        # содержит расширения Python3.12, а native SIM запускается в venv3.11.
        use_local_dependencies(prefer_environment=True)
        from manager.board import Connector
        from manager.service import converters
        writer = Writer(out / 'raw', maximum_expanded_bytes=64*1024**2, minimum_free_bytes=512*1024**2)
        serial = ''
        with (out / 'lifecycle_output.txt').open('w', encoding='utf-8') as lifecycle:
            # Driver init/lifecycle не засоряет прогресс; raw impact stdout
            # Collector сохраняет отдельно. Все вызовы остаются публичными.
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
                connector = Connector(silent=True, logger=logging.getLogger(__name__),
                                      cb_type='real' if source == 'hardware' else 'simulator',
                                      board_type=config.get('board', 'board_type'), task_format='SI',
                                      config=config, crossbar_serial=serial)
                if not connector.open_port():
                    raise ValueError('Connector open_port failed')
            collector = Collector(connector, converters, writer, source)
            summaries = collector.run(cells, lambda message: print(message, flush=True))
            metrology_pass = all(s['noise_pass'] and s['zero_shift_pass'] and s['drift_pass'] for s in summaries)
            manifest.update(status='completed', sampled_metrology_pass=metrology_pass,
                            programmable_range_validated=False, full_array_validated=False,
                            map_reuse_status='read_only_metrology_only_programmable_range_unverified',
                            original_diagnostic_grade_preserved=True)
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
        if collector is not None:
            atomic_json(out / 'observations.json', dict(checkpoints=collector.checkpoints, impacts=collector.calls, pulses=0))
            manifest['impacts'] = collector.calls
        if writer is not None:
            try:
                writer.finish()
            except Exception as exc:
                manifest.update(status='failed', raw_finish_error=str(exc))
        manifest.update(finished_utc=now(), elapsed_seconds=time.monotonic() - started)
        manifest['result_sha256'] = {p.relative_to(out).as_posix(): sha(p) for p in out.rglob('*') if p.is_file()}
        atomic_json(out / 'manifest.json', manifest)
        if locked:
            if lock.read_text(encoding='utf-8').strip() != sid:
                raise ValueError('Hardware lock ownership changed')
            lock.unlink()
    return dict(session=out.relative_to(ROOT).as_posix(), **manifest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--dry-run', action='store_true')
    modes.add_argument('--run-simulator', action='store_true')
    modes.add_argument('--run-hardware', action='store_true')
    args = parser.parse_args(argv)
    if args.dry_run:
        _, hashes = preflight()
        sample, cells = load_sample(SAMPLE)
        result: dict[str, object] = dict(protocol=SPEC, sample=sample, count=len(cells),
                                         source_sha256=hashes, connector_opened=False)
    else:
        result = run('hardware' if args.run_hardware else 'simulator')
    if args.dry_run:
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    else:
        # Все hashes/protocol остаются в manifest; терминалу только итог.
        compact = {key: result.get(key) for key in (
            'session', 'status', 'error', 'impacts', 'pulses',
            'elapsed_seconds', 'sampled_metrology_pass', 'map_reuse_status')}
        print(json.dumps(compact, ensure_ascii=False, allow_nan=False))
    return 2 if result.get('status') == 'failed' else 0


if __name__ == '__main__':
    raise SystemExit(main())
