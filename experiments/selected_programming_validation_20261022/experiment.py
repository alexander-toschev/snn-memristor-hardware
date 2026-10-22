"""Ограниченная A/B проверка16cells; публичный MemriBoard, без нейросети."""
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
from experiment_common.programming_validation import BLS, CELLS, LEVELS, Collector
from experiment_common.runtime import INI, SIM_INI, use_local_dependencies
from experiment_common.selective_metrology import Source, load_sample
from read_all import EXPECTED_BOARD, EXPECTED_SOURCES

if TYPE_CHECKING:
    from manager.board import Connector

BASE = ROOT / 'results/selected_programming_validation_20261022'
SAMPLE = ROOT / 'experiments/selective_map_validation_20261022/sample.json'
SPEC: dict[str, object] = dict(
    schema='selected16-programming-validation-v1',
    parent_map_session='4c8c70ca-f819-4c82-83c8-8fed7f9fe3b0',
    readonly_predecessor='75299d0c-c8dc-4043-b282-e43c15778ac5',
    cells=[list(cell) for cell in CELLS], pairs=[[0, 7]], bls=list(BLS),
    cycles=2, states=['R', 'M', 'H'], repeats=10, delays_s=[0, 10, 60],
    levels_v=list(LEVELS), amplitude_v=.05, maximum_active_bl=1,
    maximum_impacts=24320, maximum_pulses=1280,
    phase_a_maximum_impacts=17664, phase_a_maximum_pulses=512,
    phase_b_maximum_impacts=6656, phase_b_maximum_pulses=768,
    phase_a_checkpoint_count=416, phase_b_pair_checkpoint_count=64,
    explicit_wait_limit_s=600., soft_deadline_s=1200.,
    simulator_pulse_width_s=.000125, hardware_submitted_pulse_width_s=0.,
    physical_pulse_duration_s=None, read_current_limit_a=.005,
    output_prediction_limit_v=3.3, output_measured_limit_v=4.,
    write_current_limits_a={'0': .04, '1': .002},
    span_min_adc=10., span_snr=5., span_se_margin=3.,
    sd_floor_adc=2., sd_span_fraction=.05, drift_floor_adc=3., drift_span_fraction=.05,
    cycle_floor_adc=4., cycle_span_fraction=.10, zero_shift_floor_adc=2., zero_shift_se_margin=3.,
    pair_sign_min_adc=5., pair_sign_snr=5.,
    selection='Same16originalselectedcells; no rerank, no automatic expansion',
    a_mandatory_failure='no_stable_rectangle; no B, no replacement cells',
    b_mandatory_failure='pair_qualification_failed; no replacement cells',
    quality_failure='preserve diagnostic grade; no silent threshold expansion',
    input_labels_used=False, network_run=False, full_array_validated=False,
    private_runtime_status='private_runtime_not_independently_reverified',
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def preflight() -> tuple[ConfigParser, dict[str, str], dict[str, object]]:
    sample, all_cells = load_sample(SAMPLE)
    selected = [(c.wl, c.bl) for c in all_cells if c.sample_reason == 'original_selected_rectangle']
    if tuple(selected) != CELLS or sample['parent_session'] != SPEC['parent_map_session']:
        raise ValueError('Fixed original16cell selection mismatch')
    config = ConfigParser()
    config.read_string((ROOT / INI).read_text(encoding='utf-8-sig'))
    if config.get('board', 'board_type') != 'rp5_fpga_python':
        raise ValueError('Unexpected INI board_type')
    for key, expected in EXPECTED_BOARD.items():
        if config.getfloat('board', key) != expected:
            raise ValueError('Unexpected electrical setting: ' + key)
    paths = [ROOT / INI, ROOT / SIM_INI, SAMPLE, ROOT / 'read_all.py']
    paths += list(Path(__file__).parent.glob('*.py'))
    paths += list((ROOT / 'experiment_common').glob('*.py'))
    paths += [ROOT / 'vendor/MemriBoard' / name for name in EXPECTED_SOURCES if not name.startswith('MemriCORE/')]
    paths += [ROOT / 'vendor/MemriBoard/simulator' / name for name in ('src.py', 'memristor.py')]
    return config, {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}, sample


def run(source: Source) -> dict[str, object]:
    config, hashes, sample = preflight()
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
    runner: Collector | None = None
    started = time.monotonic()
    manifest: dict[str, object] = dict(
        session_id=sid, source=source, status='failed', started_utc=now(), python=sys.version,
        protocol=SPEC, sample_sha256=sha(SAMPLE), source_sha256=hashes,
        parent_session=sample['parent_session'], parent_map_sha256=sample['parent_map_sha256'],
        parent_quality_qualified=sample['parent_qualified'],
        private_runtime_status='private_runtime_not_independently_reverified',
        physical_pulse_duration_s=None, error=None,
    )
    print(f'Session {out.relative_to(ROOT).as_posix()}', flush=True)
    try:
        if source == 'hardware':
            lock.parent.mkdir(parents=True, exist_ok=True)
            with lock.open('x', encoding='utf-8') as stream:
                stream.write(sid + '\n')
            locked = True
        # Состав16cells/напряжения/пороги публикуются и фиксируются ДО записи.
        atomic_json(out / 'protocol.json', SPEC)
        shutil.copyfile(SAMPLE, out / 'sample.json')
        shutil.copyfile(ROOT / INI, out / 'board.ini')
        atomic_json(out / 'start_manifest.json', manifest)
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
                connector = Connector(silent=True, logger=logging.getLogger(__name__),
                                      cb_type='real' if source == 'hardware' else 'simulator',
                                      board_type=config.get('board', 'board_type'), task_format='SI',
                                      config=config, crossbar_serial=serial)
                if not connector.open_port():
                    raise ValueError('Connector open_port failed')
            runner = Collector(connector, converters, writer, source, lambda s: print(s, flush=True))
            result = runner.run()
            manifest.update(status=result['status'], qualified=result['qualified'],
                            diagnostic_best_available=result['diagnostic_best_available'])
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
        if runner is not None:
            result = runner.result()
            result['status'] = manifest['status']
            result['error'] = manifest['error']
            if manifest['status'] == 'failed':
                result['qualified'] = False
                result['diagnostic_best_available'] = False
            atomic_json(out / 'observations.json', result)
            atomic_json(out / 'map.json', {key: result[key] for key in (
                'cells', 'pair_qualification', 'qualified', 'diagnostic_best_available',
                'status', 'full_array_validated', 'calibration_matrix_reused')})
            manifest.update(impacts=runner.calls, pulses=runner.pulses,
                            phase_impacts=runner.phase_calls, phase_pulses=runner.phase_pulses)
        if writer is not None:
            try:
                writer.finish()
            except Exception as exc:
                manifest.update(status='failed', raw_finish_error=str(exc))
        manifest.update(finished_utc=now(), elapsed_seconds=time.monotonic()-started)
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
        _, hashes, _ = preflight()
        result: dict[str, object] = dict(protocol=SPEC, source_sha256=hashes, connector_opened=False)
    else:
        result = run('hardware' if args.run_hardware else 'simulator')
    keys = ('session', 'status', 'error', 'impacts', 'pulses', 'elapsed_seconds', 'qualified', 'diagnostic_best_available')
    print(json.dumps(result if args.dry_run else {k: result.get(k) for k in keys}, ensure_ascii=False))
    return 2 if result.get('status') == 'failed' else 0


if __name__ == '__main__':
    raise SystemExit(main())
