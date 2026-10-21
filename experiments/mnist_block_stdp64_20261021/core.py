"""Block STDP: SIM-проводимости frozen, цифровой target не участвует в forward."""
from __future__ import annotations
from collections.abc import Callable, Mapping
import hashlib
import json
from pathlib import Path
import time
from typing import TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray
import torch
from experiments.mnist_archive64_20261020 import inputs, native
from experiments.mnist_online_stdp64_20261021.collector import Collector as PreviousCollector
from experiments.mnist_online_stdp64_20261021.logic import DigitalStep
from experiments.cifar_resumable_20261019.durable import Writer, atomic_json
if TYPE_CHECKING:
    from manager.board import Connector


class WallBudget(Exception):
    """Конечный wall-budget: сохраняем partial, закрываем публичный порт."""


class EvidenceWriter(Writer):
    """Явно sampled raw, НЕ lossless полный журнал всех операций.

    Первый учебный блок каждой группы записывается полностью. Остальные
    сохраняют агрегаты и SHA потока пропущенных событий; SHA не заменяет raw
    replay. Старые журналы не удаляются. Все операции реально выполняются.
    """
    OMIT = frozenset(('request','result','driver_output','conductance_read'))

    def __init__(self, directory: Path) -> None:
        self.retain_raw = False
        self.omitted: dict[str, int] = {}
        self.omitted_hash = hashlib.sha256()
        super().__init__(directory, limit=8*1024**2,
                         maximum_expanded_bytes=64*1024**3, minimum_free_bytes=4*1024**3)

    def write(self, value: Mapping[str, object]) -> None:
        event = str(value.get('event', ''))
        if not self.retain_raw and event in self.OMIT:
            self.omitted[event] = self.omitted.get(event, 0)+1
            self.omitted_hash.update((json.dumps(dict(value), sort_keys=True,
                separators=(',', ':'), allow_nan=False)+'\n').encode())
            return
        super().write(value)

    def summary(self) -> dict[str, object]:
        return dict(raw_policy='first_training_block_each_stage_full_other_native_events_sampled',
                    omitted_event_counts=self.omitted, omitted_stream_sha256=self.omitted_hash.hexdigest(),
                    full_native_raw_replay_available=False, historical_raw_deleted=False)


class Collector(PreviousCollector):
    def __init__(self, connector: Connector, converters: native.Converters,
                 writer: EvidenceWriter, cap: int, pulse_cap: int, span: float,
                 progress: Callable[[str], None], deadline: float) -> None:
        super().__init__(connector, converters, writer, cap, pulse_cap, span, progress)
        self.deadline = deadline

    def check_time(self) -> None:
        if time.monotonic() >= self.deadline:
            raise WallBudget('8h-or-requested wall budget reached; partial is not accuracy')

    def call(self, command: dict[str, int|float|list[float]], **metadata: object) -> tuple[int, float, int]:
        self.check_time()
        return super().call(command, **metadata)

    def verify_end(self, stage: int) -> None:
        # CPU никогда не трогал состояние устройств. Всё же повторно читаем
        # каждый mode10 после reuse: несовпадение ADC/V запрещает результат.
        cache = self.active_cache
        if cache is None or not cache.sealed:
            raise ValueError('No sealed native cache for end check')
        for (wl, mask), old in sorted(cache.values.items(), reverse=True):
            new = self.query(wl, mask, stage, 3)
            if (old.adc, old.voltage) != (new.adc, new.voltage):
                raise ValueError('Frozen native MVM changed at block end')
        self.emit('frozen_endcheck_pass', stage=stage, generation=self.generation,
                  native_keys=len(cache.values))


def digital_block(pixels: NDArray[np.uint8], keys: NDArray[np.uint32],
                  lut: NDArray[np.float32], measured_raw: NDArray[np.float64],
                  check_time: Callable[[], None]) -> tuple[NDArray[np.float32], dict[str, object]]:
    """Последовательный shadow WDP, без цифрового W@x.

    target0=clip(measuredW,0,1). Каждый tick вычисляет обычный WDP с текущим
    shadow target, затем clip. Target переносится между изображениями,
    LIF/traces сбрасываются. Проводимости и forward LUT весь блок неизменны.
    Это новый block-delayed протокол, НЕ timestep physical STDP.
    """
    digital = DigitalStep()
    target = torch.from_numpy(np.clip(measured_raw, 0, 1).astype(np.float32))
    initial = target.clone()
    posts: list[int] = []
    changed_ticks = 0
    absolute_updates = 0.
    digest = hashlib.sha256()
    for image in range(len(pixels)):
        check_time()
        spikes = inputs.encode(pixels[image:image+1])
        masks = inputs.patch_masks(spikes)
        digest.update(masks.tobytes())
        digital.reset()
        total = 0
        for tick in range(100):
            if tick == 0:
                current = torch.zeros((1, 4, 24, 24))
            else:
                used = masks[tick-1, 0]
                index = np.searchsorted(keys, used)
                if np.any(index >= len(keys)) or not np.array_equal(keys[index], used):
                    raise ValueError('Native LUT miss; no ideal/zero substitution')
                current = torch.from_numpy(lut[index].transpose(2, 0, 1).copy()).unsqueeze(0)
            previous = target
            post, target = digital.step(current, spikes[tick], target)
            difference = target-previous
            changed_ticks += int(bool(torch.any(difference != 0)))
            absolute_updates += float(difference.abs().sum())
            total += int(post.sum())
        posts.append(total)
    result = target.numpy().copy()
    plasticity = result-initial.numpy()
    # Отдельно baseline clip correction: это НЕ корреляционное обучение.
    correction = np.clip(measured_raw, 0, 1)-measured_raw
    return result, dict(images=len(pixels), spikes_per_image=posts,
        correlated_update_ticks=changed_ticks, sum_absolute_tick_updates=absolute_updates,
        accumulated_plasticity=plasticity.tolist(), baseline_boundary_correction=correction.tolist(),
        plasticity_nonzero=int(np.count_nonzero(plasticity)),
        correction_nonzero=int(np.count_nonzero(correction)), encoding_sha256=digest.hexdigest())


def train_block(runner: Collector, writer: EvidenceWriter, pixels: NDArray[np.uint8],
                stage: int, block: int, start_image: int, out: Path) -> dict[str, object]:
    begin = time.monotonic(); calls = runner.calls; pulses = runner.pulses
    writer.retain_raw = block == 0
    before = runner.read_array(stage=stage, block=block, phase='block_baseline')
    raw = (before[:, :, 0]-before[:, :, 1])/runner.span
    seen = {0}
    for _, masks in inputs.batches(pixels, 8):
        seen.update(map(int, np.unique(masks)))
    keys = np.asarray(sorted(seen), dtype=np.uint32)
    lut = runner.measure(list(map(int, keys)), stage, 'block_frozen', block=block)
    native_ready = time.monotonic()
    target, digital = digital_block(pixels, keys, lut, raw, runner.check_time)
    digital_ready = time.monotonic()
    runner.verify_end(stage)
    # Mode7 служебный readback может измениться после MVM без изменения state.
    # Перед импульсным решением вновь измеряем G; не используем stale guard.
    pre_write = runner.read_array(stage=stage, block=block, phase='block_pre_write')
    raw_pre_write = (pre_write[:, :, 0]-pre_write[:, :, 1])/runner.span
    plasticity = target-np.clip(raw, 0, 1).astype(np.float32)
    correction = np.clip(raw, 0, 1)-raw
    metrology_shift = raw-raw_pre_write
    required = np.abs(target-raw_pre_write) > .002
    runner.update(pre_write, target, stage, start_image+len(pixels)-1, 100)
    after = runner.read_array(stage=stage, block=block, phase='block_post_write')
    runner.invalidate()  # Даже блок без admitted write не переносит кеш дальше.
    record: dict[str, object] = dict(stage=stage, block=block, first_image=start_image,
        images=len(pixels), seconds=time.monotonic()-begin, native_before_seconds=native_ready-begin,
        digital_seconds=digital_ready-native_ready, apply_seconds=time.monotonic()-digital_ready,
        calls=runner.calls-calls, pulses=runner.pulses-pulses, unique_masks=len(keys),
        measured_before_g_s=before.tolist(), measured_pre_write_g_s=pre_write.tolist(),
        measured_after_g_s=after.tolist(), target_nominal=target.tolist(), digital=digital,
        prewrite_metrology_shift_nominal=metrology_shift.tolist(),
        requested_with_plasticity=int(np.count_nonzero(required & (plasticity != 0))),
        requested_without_plasticity=int(np.count_nonzero(required & (plasticity == 0))),
        boundary_correction_present=int(np.count_nonzero(required & (correction != 0))),
        below_deadband=int(np.count_nonzero(~required)),
        cumulative_learning_stats=dict(runner.learning_stats))
    runner.emit('block_complete', **record)
    atomic_json(out/('block_%02d_%04d.json' % (stage, block)), record)
    writer.checkpoint(stage=stage, block=block, images=start_image+len(pixels), calls=runner.calls, pulses=runner.pulses)
    writer.retain_raw = False
    return record
