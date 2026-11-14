"""Bounded official writable-cell waveforms via public MemriBoard only.

The same 44 historical cells are examined first, followed by the other 212.
This produces cell-level evidence, not an automatically accepted MNIST map.
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

from experiment_common.adaptive_programming import AcquisitionBudget, Collector
from experiment_common.cell_survey import addresses
from experiment_common.official_writable import characterize, render_map, load_tickets
from experiment_common.durable import Writer, atomic_json, sha
from experiment_common.runtime import INI, SIM_INI, use_local_dependencies
from experiment_common.selective_metrology import Source
from experiment_common import terminal_publication
from read_all import EXPECTED_BOARD

if TYPE_CHECKING:
    from manager.board import Connector

BASE = ROOT / 'results/official_writable_cells_20261114'
PRIORITY = Path(__file__).parent / 'priority44.json'


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def board_config() -> ConfigParser:
    config = ConfigParser()
    config.read_string((ROOT / INI).read_text(encoding='utf-8-sig'))
    if config.get('board', 'board_type') != 'rp5_fpga_python':
        raise ValueError('Unexpected INI board type')
    for key, expected in EXPECTED_BOARD.items():
        if config.getfloat('board', key) != expected:
            raise ValueError('Unexpected electrical setting: ' + key)
    return config


def run(source: Source, publish: bool, priority_only: bool = False, smoke: bool = False, pilot: bool = False) -> tuple[dict[str, object], int]:
    if smoke and source != 'simulator':
        raise ValueError('Smoke mode is simulator-only')
    selected = addresses(PRIORITY)
    if priority_only:
        selected = selected[:44]
    if smoke or pilot:
        selected = selected[:2]
    config = board_config()
    n_cells = len(selected)
    tickets_path = Path(__file__).parent / 'writable.json'
    tickets = load_tickets(tickets_path)
    budget = AcquisitionBudget(maximum_calls=1300*n_cells,
        maximum_decisions=620*n_cells, maximum_seconds=7200. if n_cells == 256 else 1800.)
    protocol: dict[str, object] = dict(schema='official-writable-docx-stops-json-waveforms-v1',
        cells=selected, expected_cells=n_cells, priority_cells=44, smoke=smoke, pilot=pilot,
        official_commit='e29ce4f06262a2bf96b15535422bc1ecc06679a2',
        official_json_sha256=sha(tickets_path), acquisition_budget=asdict(budget),
        sequence=['SET', 'RESET', 'IV'], stop_resistance_ohm=dict(SET_lte=296, RESET_gte=2998),
        legacy_importer_inverted_inequalities_not_used=True,
        json_adc_thresholds_diagnostic_only=dict(SET_gt=9801, RESET_lt=5400),
        requested_width_s=dict(SET=.0001, RESET=.001, IV=.00013), physical_pulse_width_s=None,
        IV_screenshot_width_s=.0001, width_discrepancy='JSON130us selected, screenshot100us retained',
        current_limits_a=dict(SET=.002, RESET=.04), maximum_tasks_per_cell=620,
        docx_classification='Rmin>100 ohm and Rmax/Rmin>2',
        gui_classification='Rmin>100 ohm and Rmax/Rmin>=2',
        signed_group_mvm_qualified=False, no_automatic_mnist=True,
        raw_expanded_bytes_max=2*1024**3, disk_reserve_bytes=2*1024**3)
    if shutil.disk_usage(ROOT).free < 4*1024**3:
        raise ValueError('Need at least 4 GiB free for bounded raw plus reserve')
    sid = str(uuid.uuid4())
    out = BASE / sid
    out.mkdir(parents=True)
    log = ROOT / '.tmp' / f'publish-survey256-{sid}.log'
    sources = [Path(__file__), PRIORITY, tickets_path, ROOT / INI, ROOT / SIM_INI,
               *(ROOT / 'vendor/MemriBoard/manager').rglob('*.py'),
               *(ROOT / 'vendor/MemriBoard/simulator').rglob('*.py'),
               *(ROOT / 'experiment_common').glob('*.py')]
    manifest: dict[str, object] = dict(session_id=sid, source=source, status='failed', error=None,
        started_utc=now(), protocol=protocol,
        source_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in sources},
        private_runtime_status='private_runtime_not_independently_reverified')
    collector: Collector | None = None
    connector: Connector | None = None
    writer: Writer | None = None
    lock = ROOT / 'results/read_all/.hardware.lock'
    locked = False
    rows: list[dict[str, object]] = []
    started = time.monotonic()
    current: tuple[int, int] | None = None
    print(f'Session {sid}; {source}; {n_cells} cells; priority44 first', flush=True)
    try:
        if publish:
            terminal_publication.preflight(ROOT, log)
        if source == 'hardware':
            lock.parent.mkdir(parents=True, exist_ok=True)
            with lock.open('x', encoding='utf-8') as stream:
                stream.write(sid + '\n')
            locked = True
        atomic_json(out / 'start_manifest.json', manifest)
        atomic_json(out / 'protocol.json', protocol)
        shutil.copyfile(ROOT / INI, out / 'board.ini')
        use_local_dependencies(prefer_environment=True)
        from manager.board import Connector
        from manager.service import converters
        writer = Writer(out / 'raw', maximum_expanded_bytes=2*1024**3, minimum_free_bytes=2*1024**3)
        with (out / 'lifecycle_output.txt').open('w', encoding='utf-8') as stream:
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                serial = ''
                if source == 'simulator':
                    import numpy as np
                    from simulator.src import create_crossbar_array
                    random.seed(20261114)
                    np.random.seed(20261114)
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
        collector = Collector(connector, converters, writer, source, selected, True, budget=budget)
        for index, (wl, bl) in enumerate(selected, 1):
            current = wl, bl
            try:
                row = characterize(collector, wl, bl, tickets)
            except ValueError as exc:
                if str(exc) != 'Invalid native resistance/ADC':
                    raise
                row = dict(wl=wl, bl=bl, complete=False, error=str(exc),
                           invalid_native_measurement=True, qualified_for_mnist=False)
                collector.emit('cell_invalid_measurement', **row)
            rows.append(row)
            atomic_json(out / f'cell_{wl}_{bl}.json', row)
            current = None
            writer.checkpoint(completed_cells=index, impacts=collector.calls, pulses=collector.pulses)
            print(f'Cell {index}/{n_cells} WL{wl}/BL{bl}; switching={row.get("observed_docx_switching")}; '
                  f'calls={collector.calls}; pulses={collector.pulses}', flush=True)
        manifest['status'] = 'completed'
    except (Exception, KeyboardInterrupt) as exc:
        manifest.update(status='failed', error=type(exc).__name__ + ': ' + str(exc))
        if current is not None:
            rows.append(dict(wl=current[0], bl=current[1], complete=False,
                             error=manifest['error'], qualified_for_mnist=False))
    finally:
        if connector is not None:
            try:
                with (out / 'lifecycle_output.txt').open('a', encoding='utf-8') as stream:
                    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                        if not connector.close_port():
                            raise ValueError('Connector close_port failed')
                manifest['port_closed'] = True
            except Exception as exc:
                manifest.update(status='failed', close_error=str(exc))
        if writer is not None:
            try:
                writer.finish()
            except Exception as exc:
                manifest.update(status='failed', raw_finish_error=str(exc))
        if locked:
            if lock.read_text(encoding='utf-8').strip() != sid:
                manifest.update(status='failed', lock_error='Hardware lock owner changed; lock preserved')
            else:
                lock.unlink()
    summary: dict[str, object] = dict(session_id=sid, source=source, status=manifest['status'],
        error=manifest['error'], expected_cells=n_cells,
        completed_cells=sum(r.get('complete') is True for r in rows),
        observed_docx_switching=sum(r.get('observed_docx_switching') is True for r in rows),
        observed_gui_switching=sum(r.get('observed_gui_switching') is True for r in rows),
        measurement_errors=sum(r.get('invalid_native_measurement') is True for r in rows),
        guard_modified_cells=sum(r.get('guard_modified_protocol') is True for r in rows),
        impacts=collector.calls if collector else 0, pulses=collector.pulses if collector else 0,
        pulse_decisions=collector.decisions if collector else 0,
        elapsed_seconds=time.monotonic()-started, accuracy=None,
        signed_group_mvm_qualified=False, mnist_selected_cells=0,
        note='Resistive switching only, not analog-weight precision or MNIST/group-MVM qualification.')
    for key in ('close_error', 'raw_finish_error', 'lock_error'):
        if key in manifest:
            summary[key] = manifest[key]
    atomic_json(out / 'observations.json', dict(cells=rows))
    atomic_json(out / 'summary.json', summary)
    (out / 'map.html').write_text(render_map(rows, summary), encoding='utf-8')
    manifest.update(finished_utc=now(), elapsed_seconds=summary['elapsed_seconds'],
                    impacts=summary['impacts'], pulses=summary['pulses'])
    manifest['result_sha256'] = {p.relative_to(out).as_posix(): sha(p) for p in out.rglob('*') if p.is_file()}
    atomic_json(out / 'manifest.json', manifest)
    code = 0 if manifest['status'] == 'completed' else 2
    if publish:
        try:
            summary['publication'] = terminal_publication.publish(ROOT, out, log=log)
        except Exception as exc:
            summary['publication'] = dict(status='pending', error_type=type(exc).__name__)
            code = code or 3
    return summary, code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--run-simulator', action='store_true')
    mode.add_argument('--run-hardware', action='store_true')
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--publish-session')
    parser.add_argument('--priority-only', action='store_true', help='Exactly the first 44 cells; not a full survey')
    parser.add_argument('--pilot2', action='store_true', help='Two-cell finite pilot, same protocol')
    parser.add_argument('--smoke', action='store_true', help='Simulator-only two-cell integration check')
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    if args.publish_session:
        sid = str(uuid.UUID(args.publish_session))
        print(json.dumps(terminal_publication.publish(ROOT, BASE / sid,
            log=ROOT / '.tmp' / f'publish-survey256-{sid}.log')))
        return 0
    if args.dry_run:
        board_config()
        cells = addresses(PRIORITY)
        print(json.dumps(dict(full_cells=len(cells), priority_cells=list(cells[:44]), connector_opened=False)))
        return 0
    result, code = run('hardware' if args.run_hardware else 'simulator', args.publish,
                       args.priority_only, args.smoke, args.pilot2)
    print(json.dumps(result), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
