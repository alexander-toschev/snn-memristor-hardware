"""Один SIM-only опыт: block-STDP → frozen native LUT → CPU LIF/readout.

--benchmark --block-size32: два физических блока, не accuracy64filters.
Full: fresh64filters, train10000/test10000; deadline8h означает partial,
а не выдачу точности незавершённой сети. Hardware CLI намеренно отсутствует.
"""
from __future__ import annotations
import argparse
import contextlib
import json
import logging
from pathlib import Path
import random
import shutil
import sys
import time
from typing import cast
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from experiments.mnist_block_stdp64_20261021.core import Collector, EvidenceWriter, WallBudget, train_block
from experiments.mnist_online_stdp64_20261021.logic import initialization
from experiments.mnist_online_stdp64_20261021.collector import INIT_CALLS, INIT_PULSES
from experiments.mnist_online_stdp64_20261021.experiment import conductance_mapping, query_cost
from experiments.mnist_archive64_20261020 import inputs, lif, readout
from experiments.mnist_archive64_20261020.schema import MemmapFactory
from experiments.stable_cell_calibration_20261020 import experiment as calibration
from experiments.cifar_resumable_20261019.durable import atomic_json, sha
from experiments.unn_rev3_20261014.paths import use_local_dependencies
from experiment_hardware import launch_provenance
from read_all import now

BASE = ROOT/'results/mnist_block_stdp64_20261021'
SPEC: dict[str, object] = dict(schema='mnist-block-delayed-native-stdp-v1',
    physical_hardware=False, pretrained_weights=False, train_images=10000, test_images=10000,
    filters=64, filters_per_stage=4, stages=16, kernel=5, spatial_positions=576,
    ticks=100, poisson_rate=.0015, poisson_seed=123, initialization_seed=42,
    lif_tau=150., lif_threshold=.38, lif_refrac=2, trace_tau=20., nu_pre=.001, nu_post=.0001,
    forward='native full25bit mask mode10; frozen measured LUT within block; no ideal W@x',
    plasticity='sequential CPU WDP shadow target, clipped each tick, carried across images',
    target_reset='at every block from clipped fresh measured nominal W',
    physical_update='at block end maximum1 plus-leg pulse per coefficient',
    deadband_span_fraction=.002, set_v=1.7, reset_v=1.4, pulse_width_s=.00001,
    mvm_v=.05, physical_gain='1/(330*decodedDAC(.05)*span)', gain_refit=False,
    native_readback_repeats=3, native_lut_initial_passes=3, native_lut_endcheck_passes=1,
    raw_policy='first training block each stage full; other low-level events summarized/digested',
    old_raw_deleted=False, maximum_raw_expanded_bytes=64*1024**3, minimum_disk_reserve_bytes=4*1024**3,
    wall_budget_hours=8., runtime_selection_only=True, test_accuracy_tuning=False,
    original_timestep_stdp_equivalence=False, private_runtime_status='private_runtime_not_independently_reverified')


def run(block_size: int, benchmark: bool, start_image: int, wall_hours: float, argv: list[str]) -> dict[str, object]:
    started = time.monotonic(); deadline = started+wall_hours*3600
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    config, hashes = calibration.preflight('simulator'); low, span = conductance_mapping()
    n = 2*block_size if benchmark else 10000
    stages = 1 if benchmark else 16
    if start_image < 0 or start_image+n > 10000 or (not benchmark and start_image):
        raise ValueError('Benchmark slice must stay within first10000train; full starts0')
    # Метки здесь не открываются. Полная union для frozen evaluation необходима
    # только full; benchmark измеряет реальные два блока и небольшой CPU хвост.
    train_pixels = inputs.images('train', max(128, start_image+n))
    data = dict(train=train_pixels[:128] if benchmark else train_pixels,
                test=inputs.images('test', 128 if benchmark else 10000))
    seen = {0}
    for split, pixels in data.items():
        for offset, masks in inputs.batches(pixels, 8):
            seen.update(map(int, np.unique(masks)))
            if time.monotonic() >= deadline:raise WallBudget('Inventory wall budget')
            if offset%2048 == 0:print('Input inventory %s %d/%d' % (split, offset+len(masks[0]), len(pixels)), flush=True)
    # Для train slice benchmark не совпадающего с CPU хвостом добавляем его
    # маски только для конечного верхнего бюджета, не для тюнинга весов.
    training = train_pixels[start_image:start_image+n]
    for _, masks in inputs.batches(training, 8):seen.update(map(int, np.unique(masks)))
    keys = np.asarray(sorted(seen), dtype=np.uint32)
    blocks = (n+block_size-1)//block_size
    max_queries = query_cost(list(map(int, keys)))
    cap = stages*(INIT_CALLS+blocks*(2300+max_queries*4//3)+max_queries*4//3)
    pulse_cap = stages*(INIT_PULSES+blocks*100)
    sid = str(uuid.uuid4()); out = BASE/sid; out.mkdir(parents=True)
    for path in Path(__file__).parent.glob('*.py'):hashes[path.relative_to(ROOT).as_posix()] = sha(path)
    for directory, names in (
        ('mnist_online_stdp64_20261021', ('logic.py','collector.py','experiment.py')),
        ('mnist_archive64_20261020', ('native.py','cache.py','inputs.py','lif.py','readout.py'))):
        for name in names:
            path = ROOT/'experiments'/directory/name; hashes[path.relative_to(ROOT).as_posix()] = sha(path)
    protocol = dict(SPEC, block_size=block_size, wall_budget_hours=wall_hours)
    manifest: dict[str, object] = dict(session_id=sid, status='failed', source='simulator',
        scope='two-block-cost-benchmark' if benchmark else 'block64-train10000-test10000',
        protocol=protocol, source_sha256=hashes, invocation=launch_provenance(argv), started_utc=now(),
        images_per_stage=n, stages=stages, first_training_image=start_image,
        maximum_impacts=cap, maximum_pulses=pulse_cap, low_g_s=low, span_g_s=span)
    atomic_json(out/'start_manifest.json', manifest); atomic_json(out/'protocol.json', protocol)
    print('Block STDP SIM session '+sid, flush=True)
    writer: EvidenceWriter|None = None; runner: Collector|None = None
    connector = None; opened = False; status = 'failed'; error: str|None = None
    phases: list[dict[str, object]] = []; block_records: list[dict[str, object]] = []
    def progress(message: str) -> None:
        if time.monotonic() >= deadline:raise WallBudget('Wall budget reached')
        visible = sys.__stdout__
        if visible is None:raise ValueError('Missing foreground stdout')
        visible.write(message+'\n'); visible.flush()
    def phase(name: str, stage: int, before: tuple[float, int, int]) -> None:
        if runner is None:raise ValueError('Missing collector')
        phases.append(dict(name=name, stage=stage, seconds=time.monotonic()-before[0],
            calls=runner.calls-before[1], pulses=runner.pulses-before[2]))
        atomic_json(out/'phase_progress.json', phases)
    def stamp() -> tuple[float, int, int]:
        if runner is None:raise ValueError('Missing collector')
        return time.monotonic(), runner.calls, runner.pulses
    try:
        if shutil.disk_usage(ROOT).free < 8*1024**3:raise ValueError('Need8GiB free disk')
        initial, clipped = initialization()
        np.save(out/'fresh_initial_weights.npy', initial.numpy(), allow_pickle=False)
        np.save(out/'initial_postclip_weights.npy', clipped.numpy(), allow_pickle=False)
        np.save(out/'input_masks.npy', keys, allow_pickle=False)
        shutil.copyfile(ROOT/calibration.INI, out/'board.ini')
        shutil.copyfile(ROOT/calibration.SIM_INI, out/'simulator_settings.ini')
        with (out/'lifecycle_output.txt').open('w', encoding='utf-8', newline='\n') as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                use_local_dependencies()
                from manager.board import Connector
                from manager.service import converters
                from simulator.src import create_crossbar_array
                random.seed(20261019); np.random.seed(20261019)
                create_crossbar_array(str(out/'array'), 32, 8)
                shutil.copyfile(out/'array.cb', out/'array_initial.cb')
                writer = EvidenceWriter(out/'raw')
                connector = Connector(silent=True, logger=logging.getLogger(__name__), cb_type='simulator',
                    board_type=config.get('board','board_type'), task_format='SI', config=config, crossbar_serial=str(out/'array'))
                runner = Collector(connector, converters, writer, cap, pulse_cap, span, progress, deadline)
                manifest.update(gain=runner.gain, decoded_input_v=runner.decoded_input_v)
                opened = True
                if not connector.open_port():raise ValueError('Public Connector.open_port failed')
                runner.emit('port_open')
                for stage in range(stages):
                    begin = stamp()
                    runner.initialize(clipped[stage*4:stage*4+4].reshape(4,25).numpy().astype(np.float64), low, stage, out)
                    phase('initialization', stage, begin)
                    for block, first in enumerate(range(0, n, block_size)):
                        record = train_block(runner, writer, training[first:first+block_size], stage, block, start_image+first, out)
                        block_records.append(record)
                        progress('Block STDP stage%d/%d images%d/%d calls=%d pulses=%d elapsed=%.1fmin' %
                            (stage+1, stages, min(first+block_size,n), n, runner.calls, runner.pulses, (time.monotonic()-started)/60))
                    begin = stamp()
                    frozen = runner.measure(list(map(int,keys)), stage, 'frozen_evaluation')
                    runner.verify_end(stage)
                    np.save(out/('frozen_lut_%02d.npy'%stage), frozen, allow_pickle=False)
                    runner.invalidate(); phase('frozen_lut', stage, begin)
                    shutil.copyfile(out/'array.cb', out/('array_stage_%02d.cb'%stage))
                if not connector.close_port():raise ValueError('Public Connector.close_port failed')
                opened = False; runner.emit('port_close')
        writer.finish()
        manifest['evidence'] = writer.summary(); writer = None
        # После закрытия SIM собираем измеренные четыре-канальные LUT всех
        # последовательных состояний в64канала. Это не64physicalfilters сразу.
        frozen_all = np.concatenate([np.load(out/('frozen_lut_%02d.npy'%s), allow_pickle=False) for s in range(stages)], axis=1)
        open_counts = cast(MemmapFactory, np.lib.format.open_memmap)
        train = open_counts(out/'train_counts.npy', mode='w+', dtype=np.uint8, shape=(len(data['train']), stages*2304))
        test = open_counts(out/'test_counts.npy', mode='w+', dtype=np.uint8, shape=(len(data['test']), stages*2304))
        begin = stamp()
        for split, counts in (('train',train), ('test',test)):
            lif.extract(data[split], keys, frozen_all, .38, counts, 8, progress, split)
            counts.flush()
        phase('cpu_features', -1, begin)
        begin = stamp(); head = readout.fit(train, out, progress); phase('cpu_readout_fit', -1, begin)
        if not benchmark:
            runner.check_time()
            manifest['accuracy'] = readout.evaluate(head, test, out)
            atomic_json(out/'accuracy.json', manifest['accuracy'])
        else:
            # Benchmark head лишь измеряет время. TEST LABELS НЕ ОТКРЫВАЕМ.
            manifest['benchmark_accuracy_not_evaluated'] = True
        status = 'completed'
    except WallBudget as exc:
        status = 'failed'; error = str(exc); manifest['error_kind'] = 'wall_budget_exhausted'
    except (Exception, KeyboardInterrupt) as exc:error = type(exc).__name__+': '+str(exc)
    finally:
        if opened and connector is not None:
            try:
                if not connector.close_port():raise ValueError('Public close failed')
            except Exception as exc:status = 'failed'; manifest['cleanup_error'] = repr(exc)
        if writer is not None:
            manifest['evidence'] = writer.summary()
            try:writer.finish()
            except Exception as exc:manifest['raw_finish_error'] = repr(exc)
        manifest.update(status=status, error=error, finished_utc=now(), elapsed_s=time.monotonic()-started, phases=phases)
        if runner is not None:manifest.update(impacts=runner.calls, pulses=runner.pulses, decisions=runner.decisions, learning_stats=runner.learning_stats)
        atomic_json(out/'blocks.json', block_records)
        manifest['result_sha256'] = {p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()}
        atomic_json(out/'manifest.json', manifest)
    return dict(session=out.relative_to(ROOT).as_posix(), status=status, error=error, accuracy=manifest.get('accuracy'))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-simulator', action='store_true')
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--block-size', type=int, choices=(32,64,128,256), default=32)
    parser.add_argument('--benchmark-start', type=int, default=0)
    parser.add_argument('--wall-hours', type=float, default=8.)
    args = parser.parse_args()
    if not 0 < args.wall_hours <= 8:parser.error('wall-hours must be in(0,8]')
    if not args.run_simulator:
        calibration.preflight('simulator'); low, span = conductance_mapping()
        print(json.dumps(dict(status='preflight_only', protocol=SPEC, block_size=args.block_size, low_g_s=low, span_g_s=span)))
        return 0
    result = run(args.block_size, args.benchmark, args.benchmark_start, args.wall_hours, sys.argv)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result['status']=='completed' else 3


if __name__ == '__main__':raise SystemExit(main())
