"""MNIST64: transfer frozen STDP conductances through twelve MemriBoard blocks.

1. CPU schedules 1,600 signed coefficients into 89 loads of 18 physical pairs.
2. MemriBoard writes both conductances with bounded feedback and fresh guards.
3. MemriBoard reads back all loaded weights, measures every binary block state,
   freezes a mean-of-three operator, and checks five independent repetitions.
4. CPU assembles the measured partial operators before each logical LIF tick.
5. Frozen archive LIF/head/scaler classify the same 10,000 MNIST test images.

This is hybrid inference with measured frozen operators, not live hardware
requests for every image/tick and not analog LIF or new on-board STDP training.
"""
from __future__ import annotations

import argparse
from configparser import ConfigParser
import contextlib
from dataclasses import asdict
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiment_common.runtime import INI, SIM_INI, use_local_dependencies
use_local_dependencies(prefer_environment=True)
import numpy as np
from experiment_common.durable import Writer, atomic_json, sha
from experiment_common.block_readout_validation import Collector as ReadCollector
from experiment_common.selective_metrology import Source, object_dict, number
from experiment_common.tiled_transfer import Programmer, acquire_load, compose, profiles, schedule
from experiment_common import terminal_publication
if TYPE_CHECKING:
    from manager.board import Connector

ASSETS = Path(__file__).parent / 'assets'
BASE = ROOT / 'results/mnist_tiled_transfer_20261022'
MAP = ROOT / 'reports/2026-10-22/block-readout-validation/STABLE_BLOCKS.json'


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def classify(out: Path, keys: np.ndarray[tuple[int, ...], np.dtype[np.uint32]],
             lut: np.ndarray[tuple[int, ...], np.dtype[np.float32]], source: Source) -> dict[str, object]:
    """CPU only. Frozen source scaler/head; no fitting or test-label selection."""
    import torch
    from experiment_common.lif import extract
    from experiment_common.mnist_inputs import images
    from experiment_common.readout import evaluate
    torch.set_num_threads(4)
    model = torch.nn.Linear(64 * 24 * 24, 10, bias=True)
    checkpoint = torch.load(ASSETS / 'readout.pt', weights_only=True, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    for name in ('readout.pt', 'scaler.pt'):
        shutil.copyfile(ASSETS / name, out / name)
    counts = np.zeros((10000, 64 * 24 * 24), dtype=np.uint8)
    extract(images('test', 10000), keys, lut, .38, counts, 8, lambda s: print(s, flush=True), 'test')
    np.save(out / 'test_counts.npy', counts)
    accuracy: dict[str, object] = dict(evaluate(model, counts, out))
    accuracy.update(physical_hardware=source == 'hardware', new_readout=False,
        execution='frozen measured partial MVM operators; CPU LIF and frozen CPU head',
        live_hardware_per_image=False, test_labels_used_for_selection=False)
    atomic_json(out / 'accuracy.json', accuracy)
    return accuracy


def run(source: Source, pilot: bool, publish: bool, acquire_only: bool = False) -> tuple[dict[str, object], int]:
    selected = profiles(MAP)
    assignments = schedule(selected)
    payload = object_dict(json.loads((ASSETS / 'learned_conductances.json').read_text(encoding='utf-8')))
    goals = np.asarray(payload['conductances_s'], dtype=np.float64)
    span = number(payload['span_g_s'])
    if goals.shape != (64, 25, 2) or not np.isfinite(goals).all() or np.any(goals <= 0) or span <= 0:
        raise ValueError('Invalid learned conductance checkpoint')
    keys = np.load(ASSETS / 'input_masks.npy', allow_pickle=False).astype(np.uint32)
    if keys.ndim != 1 or keys[0] != 0 or np.any(keys[1:] <= keys[:-1]) or keys[-1] >= 1 << 25:
        raise ValueError('Invalid source mask inventory')
    config = ConfigParser()
    config.read_string((ROOT / INI).read_text(encoding='utf-8-sig'))
    if config.get('board', 'board_type') != 'rp5_fpga_python':
        raise ValueError('Unexpected driver selection')
    maximum_loads = 1 if pilot else 89
    protocol: dict[str, object] = dict(schema='mnist64-frozen-twelve-block-transfer-v1', source_session=payload['source_session'],
        loads=maximum_loads, selected=[asdict(p) for p in selected], assignments=[asdict(a) for a in assignments],
        programming='both original measured conductances, 96 feedback attempts maximum, tolerance .02*source_span',
        source_span_g_s=span, requested_pulse_width_s=.00001, hardware_pulse_duration='unknown; installed driver native pulse',
        operator='all binary patterns; three repeats frozen, five independent heldout repeats',
        repeatability='leg/signed drift <=5ADC; signed SD<=5ADC; heldout residual <=max(5ADC, .1*(abs(plus)+abs(minus)))',
        low_signal='reported, not rejected solely for a zero/small programmed weight',
        current_scale='native ADC voltage / (330 ohm * native DAC increment * source conductance span)',
        cache_scope='one programmed load; immutable response snapshot retained after physical reuse',
        physical_lif=False, live_hardware_per_image=False, head='frozen source SGD25 head and train-only scaler',
        threshold=.38, test_images=0 if pilot else 10000, new_stdp_training=False,
        raw='full native requests/responses/stdout, no periodic deletion', wall_seconds=8*3600)
    BASE.mkdir(parents=True, exist_ok=True)
    sid = str(uuid.uuid4())
    out = BASE / sid
    log = ROOT / '.tmp' / ('publish-mnist-transfer-' + sid + '.log')
    if publish:
        terminal_publication.preflight(ROOT, log)
    out.mkdir()
    lock = ROOT / 'results/read_all/.hardware.lock'
    locked = False
    connector: Connector | None = None
    writer: Writer | None = None
    programmer: Programmer | None = None
    reader: ReadCollector | None = None
    lut = np.zeros((len(keys), 64), dtype=np.float32)
    completed = 0
    read_calls = 0
    written_cells = 0
    missed_cells = 0
    started = time.monotonic()
    files = [Path(__file__), MAP, ROOT / INI, ROOT / SIM_INI, *ASSETS.iterdir(), *(ROOT / 'experiment_common').glob('*.py')]
    manifest: dict[str, object] = dict(session_id=sid, source=source, status='failed', error=None,
        started_utc=now(), protocol=protocol, source_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in files if p.is_file()})
    accuracy: dict[str, object] | None = None
    print(f'Session {out.relative_to(ROOT).as_posix()}; {maximum_loads} loads; source={source}', flush=True)
    try:
        if source == 'hardware':
            lock.parent.mkdir(parents=True, exist_ok=True)
            with lock.open('x', encoding='utf-8') as stream:
                stream.write(sid + '\n')
            locked = True
        atomic_json(out / 'protocol.json', protocol)
        atomic_json(out / 'start_manifest.json', manifest)
        shutil.copyfile(ROOT / INI, out / 'board.ini')
        writer = Writer(out / 'raw', maximum_expanded_bytes=4*1024**3, minimum_free_bytes=1024**3)
        from manager.board import Connector
        from manager.service import converters
        serial = ''
        with (out / 'lifecycle_output.txt').open('w', encoding='utf-8') as lifecycle:
            with contextlib.redirect_stdout(lifecycle), contextlib.redirect_stderr(lifecycle):
                if source == 'simulator':
                    from simulator.src import create_crossbar_array
                    random.seed(20261022)
                    np.random.seed(20261022)
                    serial = str(out / 'array')
                    create_crossbar_array(serial, 32, 8)
                    shutil.copyfile(Path(serial + '.cb'), out / 'array_initial.cb')
                    shutil.copyfile(ROOT / SIM_INI, out / 'simulator_settings.ini')
                connector = Connector(silent=True, logger=logging.getLogger(__name__),
                    cb_type='real' if source == 'hardware' else 'simulator',
                    board_type=config.get('board', 'board_type'), task_format='SI', config=config, crossbar_serial=serial)
                if not connector.open_port():
                    raise ValueError('Connector open_port failed')
            programmer = Programmer(connector, converters, writer, source, span)
            reader = ReadCollector(connector, converters, writer, source)
            for load in range(maximum_loads):
                if time.monotonic() - started > 8*3600:
                    raise TimeoutError('Eight-hour budget')
                assigned = tuple(a for a in assignments if a.load == load)
                records: list[dict[str, object]] = []
                for p in selected:
                    for a in (a for a in assigned if a.block == p.block.name):
                        for leg, wl in enumerate((p.block.plus_wl, p.block.minus_wl)):
                            records.append(programmer.program(wl, p.block.bls[a.slot], float(goals[a.filter, a.tap, leg]),
                                load=load, block=a.block, filter=a.filter, tap=a.tap, leg=leg))
                    print(f'Load {load+1}/{maximum_loads} programmed {p.block.name}; pulses={programmer.pulses}', flush=True)
                # Read all weights again after ALL writes, so later programming
                # cannot silently invalidate the earlier per-cell readback.
                for r in records:
                    actual = programmer.read_g(int(number(r['wl'])), int(number(r['bl'])), role='whole_load_readback', load=load)
                    r.update(final_g_s=actual, error_g_s=actual-number(r['target_g_s']), within_tolerance=abs(actual-number(r['target_g_s'])) <= .02*span)
                written_cells += len(records)
                missed_cells += sum(r['within_tolerance'] is not True for r in records)
                atomic_json(out / f'load_{load:03d}_programming.json', records)
                reader.emit('load_started', load=load)
                try:
                    operators, passed = acquire_load(reader, selected, out, load)
                finally:
                    read_calls += reader.calls
                if not passed:
                    raise ValueError(f'Post-program repeatability failed in load {load}; see operators.json; no accuracy claimed')
                compose(keys, assigned, operators, span, lut)
                completed += 1
                np.save(out / 'partial_lut.npy', lut)
                atomic_json(out / 'progress.json', dict(completed_loads=completed, planned_loads=maximum_loads,
                    elapsed_s=time.monotonic()-started, impacts=programmer.calls+read_calls, pulses=programmer.pulses))
                print(f'Load {completed}/{maximum_loads} measured and validated; elapsed={time.monotonic()-started:.1f}s', flush=True)
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
        if locked:
            if lock.read_text(encoding='utf-8').strip() != sid:
                raise ValueError('Lock owner changed')
            lock.unlink()
    # No hardware is held open during the CPU classifier.
    if manifest['status'] == 'completed' and not pilot and not acquire_only:
        try:
            accuracy = classify(out, keys, lut, source)
        except (Exception, KeyboardInterrupt) as exc:
            manifest.update(status='failed', error=type(exc).__name__ + ': ' + str(exc))
    summary: dict[str, object] = dict(session_id=sid, source=source, status=manifest['status'], error=manifest['error'],
        completed_loads=completed, planned_loads=maximum_loads, pilot=pilot, accuracy=accuracy,
        scope='pilot' if pilot else ('acquisition_only_accuracy_pending' if acquire_only else 'acquisition_and_accuracy'),
        impacts=(programmer.calls if programmer else 0)+read_calls, pulses=programmer.pulses if programmer else 0,
        elapsed_seconds=time.monotonic()-started, physical_cells=36, logical_coefficients=1600,
        programmed_cell_loads=written_cells, target_misses=missed_cells,
        target_tolerance_s=.02*span, programming_errors_replaced_by_ideal_weights=False,
        inference='measured frozen partial MVM LUT + digital sum + CPU LIF + frozen CPU head')
    atomic_json(out / 'summary.json', summary)
    (out / 'map.html').write_text('<!doctype html><meta charset="utf-8"><h1>MNIST64 tiled transfer</h1><pre>'+html.escape(json.dumps(summary, indent=2, ensure_ascii=False))+'</pre>', encoding='utf-8')
    manifest.update(finished_utc=now(), **{k: summary[k] for k in ('impacts', 'pulses', 'completed_loads', 'elapsed_seconds')})
    manifest['result_sha256'] = {p.relative_to(out).as_posix(): sha(p) for p in out.rglob('*') if p.is_file()}
    atomic_json(out / 'manifest.json', manifest)
    code = 0 if manifest['status'] == 'completed' else 2
    if publish:
        try:
            summary['publication'] = terminal_publication.publish(ROOT, out, log=log)
        except Exception as exc:
            summary['publication'] = dict(status='pending', error=type(exc).__name__, local_log=log.relative_to(ROOT).as_posix())
            code = code or 3
    return summary, code


def evaluate_session(sid: str, publish: bool) -> tuple[dict[str, object], int]:
    """Read preserved acquisition on a CPU host; this path never opens a Connector."""
    source_path = BASE / sid
    meta = object_dict(json.loads((source_path / 'manifest.json').read_text(encoding='utf-8')))
    summary = object_dict(json.loads((source_path / 'summary.json').read_text(encoding='utf-8')))
    if meta['status'] != 'completed' or summary['completed_loads'] != 89 or summary['pilot'] is not False:
        raise ValueError('Full completed acquisition required; no partial accuracy')
    hashes = object_dict(meta['result_sha256'])
    if hashes['partial_lut.npy'] != sha(source_path / 'partial_lut.npy'):
        raise ValueError('Acquired LUT hash mismatch')
    code_hashes = object_dict(meta['source_sha256'])
    for asset in ASSETS.iterdir():
        if asset.is_file() and code_hashes[asset.relative_to(ROOT).as_posix()] != sha(asset):
            raise ValueError('Frozen inference asset differs from acquisition')
    source: Source = 'hardware' if meta['source'] == 'hardware' else 'simulator'
    result_id = str(uuid.uuid4())
    out = BASE / result_id
    out.mkdir()
    started = time.monotonic()
    manifest: dict[str, object] = dict(session_id=result_id, source=source, status='failed', error=None,
        started_utc=now(), source_session=sid, source_manifest_sha256=sha(source_path/'manifest.json'),
        source_lut_sha256=sha(source_path/'partial_lut.npy'), hardware_calls_this_process=0,
        protocol=meta['protocol'], source_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in (Path(__file__), *(ROOT/'experiment_common').glob('*.py'))})
    accuracy: dict[str, object] | None = None
    try:
        keys = np.load(ASSETS/'input_masks.npy', allow_pickle=False).astype(np.uint32)
        lut = np.load(source_path/'partial_lut.npy', allow_pickle=False).astype(np.float32)
        if lut.shape != (len(keys), 64) or not np.isfinite(lut).all():
            raise ValueError('Invalid acquired LUT')
        accuracy = classify(out, keys, lut, source)
        manifest['status'] = 'completed'
    except (Exception, KeyboardInterrupt) as exc:
        manifest.update(status='failed', error=type(exc).__name__ + ': ' + str(exc))
    result: dict[str, object] = dict(session_id=result_id, source=source, status=manifest['status'], error=manifest['error'],
        acquisition_session=sid, accuracy=accuracy, scope='CPU inference from frozen acquired operators',
        elapsed_seconds=time.monotonic()-started)
    atomic_json(out/'summary.json', result)
    (out/'map.html').write_text('<!doctype html><meta charset="utf-8"><h1>MNIST accuracy</h1><pre>'+html.escape(json.dumps(result, indent=2, ensure_ascii=False))+'</pre>', encoding='utf-8')
    manifest.update(finished_utc=now(), result_sha256={p.relative_to(out).as_posix(): sha(p) for p in out.rglob('*') if p.is_file()})
    atomic_json(out/'manifest.json', manifest)
    if publish:
        result['publication'] = terminal_publication.publish(ROOT, out, log=ROOT/'.tmp'/('publish-mnist-transfer-'+result_id+'.log'))
    return result, 0 if manifest['status'] == 'completed' else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--run-simulator', action='store_true')
    modes.add_argument('--run-hardware', action='store_true')
    modes.add_argument('--publish-session', metavar='UUID')
    modes.add_argument('--evaluate-session', metavar='UUID', help='CPU only from preserved full acquisition')
    parser.add_argument('--pilot', action='store_true', help='One load only, no accuracy claim')
    parser.add_argument('--acquire-only', action='store_true', help='Save operators for local CPU inference')
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args(argv)
    if args.publish_session:
        sid = str(uuid.UUID(args.publish_session))
        print(json.dumps(terminal_publication.publish(ROOT, BASE/sid, log=ROOT/'.tmp'/('publish-mnist-transfer-'+sid+'.log'))))
        return 0
    if args.evaluate_session:
        result, code = evaluate_session(str(uuid.UUID(args.evaluate_session)), args.publish)
    else:
        result, code = run('hardware' if args.run_hardware else 'simulator', args.pilot, args.publish, args.acquire_only)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
