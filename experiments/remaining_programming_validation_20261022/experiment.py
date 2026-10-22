"""Одна foreground-команда: A240 → проверка прежних16 → карта → archive/push.

Ни фаза B, ни MNIST не запускаются. Публикация по явному --publish; повтор
--publish-session работает только с завершёнными данными, без обращения к плате.
"""
from __future__ import annotations

import argparse
from configparser import ConfigParser
import contextlib
from dataclasses import asdict
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
from experiment_common.calibration_inventory import REMAINING, SMOKE, inventory, row_map, render_map
from experiment_common.durable import Writer, atomic_json, sha
from experiment_common.programming_validation import CELLS, LEVELS, Collector, Schedule, assess_cells
from experiment_common.runtime import INI, SIM_INI, use_local_dependencies
from experiment_common.selective_metrology import Source, object_dict
from experiment_common import terminal_publication
from read_all import EXPECTED_BOARD, EXPECTED_SOURCES

if TYPE_CHECKING:
    from manager.board import Connector

BASE = ROOT / 'results/remaining_programming_validation_20261022'
PRIOR = Path(__file__).parent / 'prior16'


class FunctionalClock:
    """SIM smoke: виртуальные ожидания, поэтому retention не квалифицируется."""
    def __init__(self) -> None:
        self.offset = 0.

    def now(self) -> float:
        return time.monotonic() + self.offset

    def sleep(self, seconds: float) -> None:
        self.offset += seconds


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read(path: Path) -> dict[str, object]:
    decoded: object = json.loads(path.read_text(encoding='utf-8'))
    return object_dict(decoded)


def protocol(smoke: bool) -> tuple[Schedule, dict[str, object]]:
    schedule = Schedule(cells=SMOKE if smoke else REMAINING, include_b=False,
                        final_recheck_cells=(CELLS[0], CELLS[-1]) if smoke else CELLS, wall_seconds=3600.)
    spec: dict[str, object] = dict(
        schema='remaining240-programmability-v1', cells=[list(c) for c in schedule.cells],
        excluded_from_programming=[list(c) for c in CELLS],
        final_readonly_recheck=[list(c) for c in schedule.final_recheck_cells],
        phase='A only; no B', cycles=2, states=['R', 'M', 'H'], repeats=10,
        delays_s=[0, 10, 60], levels_v=list(LEVELS), amplitude_v=.05,
        maximum_impacts=sum(schedule.calls_by_phase.values()),
        maximum_pulses=sum(schedule.pulses_by_phase.values()),
        phase_call_caps=schedule.calls_by_phase, phase_pulse_caps=schedule.pulses_by_phase,
        explicit_wait_limit_s=360., wall_seconds=3600., raw_maximum_bytes=512*1024**2,
        simulator_smoke=smoke, virtual_waits=smoke, retention_qualification_performed=not smoke,
        simulator_pulse_width_s=.000125, hardware_submitted_pulse_width_s=0., physical_pulse_duration_s=None,
        read_current_limit_a=.005, write_current_limits_a={'0': .04, '1': .002},
        predicted_output_limit_v=3.3, measured_output_limit_v=4.,
        span_min_adc=10., span_snr=5., span_se_margin=3.,
        sd_floor_adc=2., sd_span_fraction=.05, drift_floor_adc=3., drift_span_fraction=.05,
        cycle_floor_adc=4., cycle_span_fraction=.1, zero_shift_floor_adc=2., zero_shift_se_margin=3.,
        thresholds_changed=False, repeat_failed_cells=False, automatic_neural_run=False,
        prior16_session='6fa70f93-2ad8-4cc0-ab2a-d009cdd47886',
        private_runtime_status='private_runtime_not_independently_reverified')
    return schedule, spec


def preflight() -> tuple[ConfigParser, dict[str, str], dict[str, object], dict[str, object]]:
    config = ConfigParser()
    config.read_string((ROOT / INI).read_text(encoding='utf-8-sig'))
    if config.get('board', 'board_type') != 'rp5_fpga_python':
        raise ValueError('Unexpected INI board driver')
    for key, expected in EXPECTED_BOARD.items():
        if config.getfloat('board', key) != expected:
            raise ValueError('Unexpected electrical setting: ' + key)
    prior_manifest, prior = read(PRIOR / 'manifest.json'), read(PRIOR / 'observations.json')
    if prior_manifest['session_id'] != '6fa70f93-2ad8-4cc0-ab2a-d009cdd47886' or prior_manifest['source'] != 'hardware':
        raise ValueError('Wrong prior16 source session')
    if object_dict(prior_manifest['result_sha256'])['observations.json'] != sha(PRIOR / 'observations.json') or set(row_map(prior['cells'])) != set(CELLS):
        raise ValueError('Prior16 data identity mismatch')
    paths = [ROOT / INI, ROOT / SIM_INI, ROOT / 'read_all.py', ROOT / 'tools/archive_calibration.py',
             ROOT / 'tools/run_cifar_batch.py', ROOT / 'tools/security/scan_staged_lfs.py']
    paths += list(Path(__file__).parent.glob('*.py')) + list(PRIOR.glob('*.json'))
    paths += list((ROOT / 'experiment_common').glob('*.py'))
    paths += [ROOT / 'vendor/MemriBoard' / name for name in EXPECTED_SOURCES if not name.startswith('MemriCORE/')]
    paths += [ROOT / 'vendor/MemriBoard/simulator' / name for name in ('src.py', 'memristor.py')]
    return config, {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}, prior_manifest, prior


def run(source: Source, smoke: bool, publish: bool) -> tuple[dict[str, object], int]:
    if source == 'hardware' and smoke:
        raise ValueError('Hardware must execute exact240 targets with real waits')
    config, hashes, prior_manifest, prior = preflight()
    schedule, spec = protocol(smoke)
    BASE.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(BASE).free < 2*1024**3:
        raise ValueError('Need2GiBfree for logs and archive')
    sid = str(uuid.uuid4())
    log = ROOT / '.tmp' / ('publish-remaining-' + sid + '.log')
    launch_head: str | None = None
    if publish:
        launch_head = terminal_publication.preflight(ROOT, log)
    out = BASE / sid
    out.mkdir()
    lock = ROOT / 'results/read_all/.hardware.lock'
    locked = False
    connector: Connector | None = None
    writer: Writer | None = None
    runner: Collector | None = None
    started = time.monotonic()
    manifest: dict[str, object] = dict(
        session_id=sid, source=source, status='failed', error=None, started_utc=now(), python=sys.version,
        protocol=spec, source_sha256=hashes, launch_git_head=launch_head,
        prior16_session=prior_manifest['session_id'], prior16_finished_utc=prior_manifest['finished_utc'],
        private_runtime_status='private_runtime_not_independently_reverified', publish_requested=publish)
    print(f'Session {out.relative_to(ROOT).as_posix()}', flush=True)
    try:
        if source == 'hardware':
            lock.parent.mkdir(parents=True, exist_ok=True)
            with lock.open('x', encoding='utf-8') as stream:
                stream.write(sid + '\n')
            locked = True
        atomic_json(out / 'protocol.json', spec)
        atomic_json(out / 'start_manifest.json', manifest)
        shutil.copyfile(ROOT / INI, out / 'board.ini')
        for name in ('manifest.json', 'observations.json'):
            shutil.copyfile(PRIOR / name, out / ('prior16_' + name))
        use_local_dependencies(prefer_environment=True)
        from manager.board import Connector
        from manager.service import converters
        writer = Writer(out / 'raw', maximum_expanded_bytes=512*1024**2, minimum_free_bytes=1024**3)
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
            clock = FunctionalClock()
            runner = Collector(connector, converters, writer, source, lambda s: print(s, flush=True),
                               schedule=schedule, clock=clock.now if smoke else time.monotonic,
                               sleep=clock.sleep if smoke else time.sleep)
            runner.run()
            manifest['status'] = 'completed'
    except (Exception, KeyboardInterrupt) as exc:
        manifest.update(status='failed', error=type(exc).__name__ + ': ' + str(exc))
    finally:
        if connector is not None:
            try:
                with (out / 'lifecycle_output.txt').open('a', encoding='utf-8') as stream:
                    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                        if not connector.close_port():
                            raise ValueError('Connector close_port failed')
            except Exception as exc:
                manifest.update(status='failed', close_error=str(exc))
        if writer is not None:
            try:
                writer.finish()
            except Exception as exc:
                manifest.update(status='failed', raw_finish_error=str(exc))
        finished = now()
        # Даже partial получает все запланированные адреса с explicit missing.
        rows = assess_cells(runner.points if runner else [], runner.updates if runner else [], schedule.cells)
        result: dict[str, object] = runner.result() if runner else dict(checkpoints=[], updates=[], pairs=[])
        result.update(cells=rows, status=manifest['status'], error=manifest['error'])
        atomic_json(out / 'observations.json', result)
        combined: dict[str, object] | None = None
        try:
            if source == 'hardware' and not smoke:
                combined = inventory(rows, prior, prior_manifest, runner.points if runner else [], sid, finished)
                atomic_json(out / 'inventory256.json', combined)
        except Exception as exc:
            manifest.update(status='failed', inventory_error=str(exc))
        manifest.update(impacts=runner.calls if runner else 0, pulses=runner.pulses if runner else 0,
                        phase_impacts=runner.phase_calls if runner else {},
                        finished_utc=finished, elapsed_seconds=time.monotonic() - started)
        summary: dict[str, object] = dict(
            session_id=sid, source=source, status=manifest['status'], error=manifest['error'],
            impacts=manifest['impacts'], pulses=manifest['pulses'], elapsed_seconds=manifest['elapsed_seconds'],
            programmed_targets=len(schedule.cells), hard_eligible=sum(r['hard_eligible'] is True for r in rows),
            quality_eligible=sum(r['eligible'] is True for r in rows),
            incomplete_cells=sum('missing_required_state_checkpoint' in str(r['hard_reasons']) for r in rows),
            final_old16_rechecks=len([p for p in runner.points if p.role == 'final_recheck']) if runner else 0,
            simulator_smoke=smoke, virtual_waits=smoke,
            retention_qualification_performed=not smoke and manifest['status'] == 'completed',
            simultaneous_256_calibration=False, hardware_mnist_200_qualified=False,
            prior16_programmability_finished_utc=prior_manifest['finished_utc'], current_finished_utc=finished,
            hard_diagnostic_candidates=combined['hard_diagnostic_candidates'] if combined else None,
            strict_quality_candidates=combined['strict_quality_candidates'] if combined else None)
        atomic_json(out / 'summary.json', summary)
        map_rows = list(row_map(combined['cells']).values()) if combined else rows
        (out / 'map.html').write_text(render_map(map_rows, summary), encoding='utf-8')
        manifest['result_sha256'] = {p.relative_to(out).as_posix(): sha(p) for p in out.rglob('*') if p.is_file()}
        atomic_json(out / 'manifest.json', manifest)
        if locked:
            if lock.read_text(encoding='utf-8').strip() != sid:
                raise ValueError('Hardware lock ownership changed')
            lock.unlink()
    # Publish only after terminal manifest, raw finish and port close. A failed
    # acquisition still publishes its exact partial evidence and returns code2.
    exit_code = 2 if manifest['status'] == 'failed' else 0
    publication: dict[str, object] | None = None
    if publish:
        try:
            publication = terminal_publication.publish(ROOT, out, log=log)
        except Exception as exc:
            publication = dict(status='pending', error_type=type(exc).__name__, local_log=log.relative_to(ROOT).as_posix())
            atomic_json(ROOT / '.tmp' / ('publication-pending-' + sid + '.json'), publication)
            exit_code = exit_code or 3
    return dict(session=out.relative_to(ROOT).as_posix(), summary=summary, publication=publication), exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--dry-run', action='store_true')
    modes.add_argument('--run-hardware', action='store_true')
    modes.add_argument('--run-simulator', action='store_true')
    modes.add_argument('--publish-session', metavar='UUID')
    parser.add_argument('--smoke', action='store_true', help='Only SIM:4targets,2old-cellreads,virtualwaits;no retention claim')
    parser.add_argument('--publish', action='store_true', help='Archive+scan+exactGitcommit/push after terminal acquisition')
    args = parser.parse_args(argv)
    if args.smoke and not args.run_simulator:
        parser.error('--smoke is allowed only with --run-simulator')
    if args.publish_session:
        sid = str(uuid.UUID(args.publish_session))
        result = terminal_publication.publish(ROOT, BASE / sid, log=ROOT / '.tmp' / ('publish-remaining-' + sid + '.log'))
        print(json.dumps(result))
        return 2 if read(BASE / sid / 'manifest.json')['status'] == 'failed' else 0
    if args.dry_run:
        _, hashes, _, _ = preflight()
        print(json.dumps(dict(protocol=protocol(False)[1], source_sha256=hashes, connector_opened=False)))
        return 0
    outcome, exit_code = run('hardware' if args.run_hardware else 'simulator', args.smoke, args.publish)
    print(json.dumps(outcome, ensure_ascii=False, allow_nan=False))
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
