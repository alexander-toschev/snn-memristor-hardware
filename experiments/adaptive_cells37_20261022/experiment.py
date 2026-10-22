"""Native MemriBoard37-cell amplitude feedback; acquisition and automatic Git publication.

CPU selects amplitudes/targets. MemriBoard performs ALL writes, G reads and
singleton MVM measurements. No MNIST accuracy is inferred from cell qualification.
Run SIM fixed and adaptive first; hardware uses adaptive only, exact37 addresses.
"""
from __future__ import annotations

import argparse
from configparser import ConfigParser
import contextlib
from datetime import datetime, timezone
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiment_common.runtime import INI, SIM_INI, use_local_dependencies
use_local_dependencies(prefer_environment=True)
import numpy as np
from experiment_common.adaptive_programming import Collector, LEVELS, TOLERANCE
from experiment_common.durable import Writer, atomic_json, sha
from experiment_common.selective_metrology import Source, object_dict, number
from experiment_common import terminal_publication
if TYPE_CHECKING:
    from manager.board import Connector

BASE = ROOT/'results/adaptive_cells37_20261022'
MAP = Path(__file__).parent/'candidates.json'


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def candidates() -> tuple[tuple[int, int], ...]:
    raw: object = json.loads(MAP.read_text(encoding='utf-8'))
    if not isinstance(raw, list):
        raise ValueError('Candidate list required')
    result: list[tuple[int, int]] = []
    for value in raw:
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError('Coordinate pair required')
        w, b = int(number(value[0])), int(number(value[1]))
        if not 0 <= w < 8 or not 0 <= b < 32:
            raise ValueError('Invalid address')
        result.append((w, b))
    if len(result) != 37 or len(set(result)) != 37:
        raise ValueError('Exact37 distinct candidates required')
    return tuple(result)


def run(source: Source, adaptive: bool, publish: bool) -> tuple[dict[str, object], int]:
    if source == 'hardware' and not adaptive:
        raise ValueError('Hardware authorization is adaptive37 only')
    cells = candidates()
    sid = str(uuid.uuid4())
    out = BASE/sid
    out.mkdir(parents=True)
    config = ConfigParser()
    config.read_string((ROOT/INI).read_text(encoding='utf-8-sig'))
    connector: Connector | None = None
    writer: Writer | None = None
    collector: Collector | None = None
    lock = ROOT/'results/read_all/.hardware.lock'
    locked = False
    started = time.monotonic()
    records: list[dict[str, object]] = []
    protocol = dict(cells=cells, policy='adaptive' if adaptive else 'fixed',
        discovery='RESET staircase then SET staircase; original0.3..3.0V envelope',
        targets='20/50/80 percent of independently measured endpoint range; same targets in reverse cycle',
        cycles=2, attempts_per_target=64, pulse_decisions_max=16000, calls_max=180000,
        wall_seconds=1800, tolerance_g_s=TOLERANCE, minimum_discovery_span_g_s=8*TOLERANCE,
        full_acceptance='6/6 confirmed targets and final whole-array readback within same tolerance',
        source_specific_dynamics=True, physical_pulse_width_s=None, submitted_pulse_width_s=.00001,
        current_limits_a=dict(SET=.002, RESET=.04, MVM=.005),
        no_automatic_mnist=True, no_test_labels=True, singleton_readout_not_signed_block_qualification=True)
    files = [Path(__file__), MAP, ROOT/INI, ROOT/SIM_INI, *(ROOT/'experiment_common').glob('*.py')]
    manifest: dict[str, object] = dict(session_id=sid, source=source, status='failed', error=None,
        started_utc=now(), protocol=protocol, source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in files})
    print(f'Session {sid}; {source}; policy={protocol["policy"]};37cells', flush=True)
    try:
        if source == 'hardware':
            lock.parent.mkdir(parents=True, exist_ok=True)
            with lock.open('x', encoding='utf-8') as stream:
                stream.write(sid+'\n')
            locked = True
        atomic_json(out/'start_manifest.json', manifest)
        atomic_json(out/'protocol.json', protocol)
        shutil.copyfile(ROOT/INI, out/'board.ini')
        writer = Writer(out/'raw', maximum_expanded_bytes=512*1024**2, minimum_free_bytes=1024**3)
        from manager.board import Connector
        from manager.service import converters
        with (out/'lifecycle_output.txt').open('w', encoding='utf-8') as stream:
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                serial = ''
                if source == 'simulator':
                    from simulator.src import create_crossbar_array
                    random.seed(20261022)
                    np.random.seed(20261022)
                    serial = str(out/'array')
                    create_crossbar_array(serial, 32, 8)
                    shutil.copyfile(Path(serial+'.cb'), out/'array_initial.cb')
                    shutil.copyfile(ROOT/SIM_INI, out/'simulator_settings.ini')
                connector = Connector(silent=True, logger=logging.getLogger(__name__),
                    cb_type='real' if source == 'hardware' else 'simulator', task_format='SI',
                    board_type=config.get('board', 'board_type'), config=config, crossbar_serial=serial)
                if not connector.open_port():
                    raise ValueError('Connector open_port failed')
            collector = Collector(connector, converters, writer, source, cells, adaptive)
            for index, (wl, bl) in enumerate(cells):
                # Explore only authorized native amplitudes. No target is
                # obtained by reading an internal simulator state.
                discovery: list[dict[str, object]] = []
                endpoints: list[float] = []
                for sign in (0, 1):
                    for voltage in LEVELS:
                        g, admitted = collector.pulse(wl, bl, sign, voltage, phase='discovery')
                        discovery.append(dict(sign=sign, voltage_v=voltage, g_s=g, admitted=admitted))
                    time.sleep(.1)
                    endpoints.append(collector.read_g(wl, bl, role='discovery_endpoint', sign=sign))
                low, high = endpoints
                span = high-low
                targets: list[dict[str, object]] = []
                if span > 8*TOLERANCE:
                    goals = [low+fraction*span for fraction in (.2, .5, .8)]
                    for cycle in range(2):
                        for level in ((0, 1, 2) if cycle == 0 else (2, 1, 0)):
                            result = collector.program(wl, bl, goals[level], cycle, level)
                            result['mvm'] = [collector.readout(wl, bl, bias) for bias in (0., .025)]
                            targets.append(result)
                row: dict[str, object] = dict(wl=wl, bl=bl, discovery=discovery, endpoints_g_s=endpoints,
                    span_g_s=span, targets=targets, tracking_passed=len(targets)==6 and all(r['passed'] is True for r in targets),
                    final_target_g_s=number(targets[-1]['target_g_s']) if targets else None,
                    range_failure=None if targets else 'insufficient_or_wrong_direction_endpoint_span')
                records.append(row)
                atomic_json(out/f'cell_{wl}_{bl}.json', row)
                writer.checkpoint(completed_cells=index+1, impacts=collector.calls, pulses=collector.pulses)
                print(f'Cell {index+1}/37 WL{wl}/BL{bl}: targets={sum(r["passed"] is True for r in targets)}/6 span={span*1e6:.2f}uS pulses={collector.pulses}', flush=True)
            for row in records:
                g = collector.read_g(int(number(row['wl'])), int(number(row['bl'])), role='whole_array_readback')
                row['whole_array_final_g_s'] = g
                target = row['final_target_g_s']
                row['qualified_programming'] = row['tracking_passed'] is True and target is not None and abs(g-number(target)) <= TOLERANCE
            manifest['status'] = 'completed'
    except (Exception, KeyboardInterrupt) as exc:
        manifest.update(status='failed', error=type(exc).__name__+': '+str(exc))
    finally:
        if connector is not None:
            try:
                with (out/'lifecycle_output.txt').open('a', encoding='utf-8') as stream:
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
                raise ValueError('Hardware lock owner changed')
            lock.unlink()
    summary: dict[str, object] = dict(session_id=sid, source=source, status=manifest['status'], error=manifest['error'],
        policy=protocol['policy'], completed_cells=len(records),
        qualified_programming=sum(r.get('qualified_programming') is True for r in records),
        impacts=collector.calls if collector else 0, pulses=collector.pulses if collector else 0,
        elapsed_seconds=time.monotonic()-started, accuracy=None, block_qualification_pending=True)
    atomic_json(out/'observations.json', dict(cells=records))
    atomic_json(out/'summary.json', summary)
    (out/'map.html').write_text('<!doctype html><meta charset="utf-8"><h1>Adaptive37 MemriBoard</h1><pre>'+html.escape(json.dumps(summary, indent=2))+'</pre>', encoding='utf-8')
    manifest.update(finished_utc=now(), impacts=summary['impacts'], pulses=summary['pulses'])
    manifest['result_sha256'] = {p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()}
    atomic_json(out/'manifest.json', manifest)
    code = 0 if manifest['status']=='completed' else 2
    if publish:
        try:
            summary['publication'] = terminal_publication.publish(ROOT, out, log=ROOT/'.tmp'/f'publish-adaptive37-{sid}.log')
        except Exception as exc:
            summary['publication'] = dict(status='pending', error=type(exc).__name__)
            code = code or 3
    return summary, code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--run-simulator', action='store_true')
    mode.add_argument('--run-hardware', action='store_true')
    mode.add_argument('--publish-session')
    parser.add_argument('--fixed', action='store_true', help='Matched SIM baseline only')
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    if args.publish_session:
        sid = str(uuid.UUID(args.publish_session))
        print(json.dumps(terminal_publication.publish(ROOT, BASE/sid, log=ROOT/'.tmp'/f'publish-adaptive37-{sid}.log')))
        return 0
    result, code = run('hardware' if args.run_hardware else 'simulator', not args.fixed, args.publish)
    print(json.dumps(result), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
