"""Transfer learned conductances through public MemriBoard, then freeze measured operators.

The CPU schedules physical tiles and adds measured partial responses. It never
substitutes target weights for an unmeasured response. LIF runs only after the
complete logical 25-input current has been assembled.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import statistics
import time
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from experiment_common.block_readout_validation import Block, Collector as ReadCollector, Pair, patterns, Phase, Mode
from experiment_common.durable import Writer, atomic_json
from experiment_common.native_memriboard import Converters
from experiment_common.selective_metrology import Source, object_dict, number
from experiment_common.stdp_collector import Collector as WriteCollector

if TYPE_CHECKING:
    from manager.board import Connector


@dataclass(frozen=True)
class Profile:
    block: Block
    biased: bool
    delta: float


@dataclass(frozen=True)
class Assignment:
    load: int
    block: str
    slot: int
    filter: int
    tap: int


def profiles(path: Path) -> tuple[Profile, ...]:
    data = object_dict(json.loads(path.read_text(encoding='utf-8')))
    selected = data['selected_blocks']
    if not isinstance(selected, list):
        raise ValueError('Expected selected blocks')
    result: list[Profile] = []
    used: set[tuple[int, int]] = set()
    for value in selected:
        row = object_dict(value)
        pair, bls = row['wl_pair'], row['bls']
        if not isinstance(pair, list) or len(pair) != 2 or not isinstance(bls, list):
            raise ValueError('Invalid block coordinates')
        block = Block(str(row['block']), int(number(pair[0])), int(number(pair[1])), tuple(int(number(b)) for b in bls))
        if not 1 <= len(block.bls) <= 3 or block.plus_wl == block.minus_wl:
            raise ValueError('Expected distinct differential legs and at most three inputs')
        for wl in (block.plus_wl, block.minus_wl):
            for bl in block.bls:
                if not 0 <= wl < 8 or not 0 <= bl < 25 or (wl, bl) in used:
                    raise ValueError('Duplicate or unsupported physical cell')
                used.add((wl, bl))
        if row['mode'] not in ('ordinary', 'fixed_positive') or number(row['delta_v']) not in (.025, .05):
            raise ValueError('Unexpected readout profile')
        result.append(Profile(block, row['mode'] == 'fixed_positive', number(row['delta_v'])))
    if len(result) != 12 or len(used) != 36:
        raise ValueError('Expected the qualified twelve blocks / 36 cells')
    return tuple(result)


def schedule(selected: tuple[Profile, ...]) -> tuple[Assignment, ...]:
    """Pack all 1,600 coefficients once; blocks may straddle logical filters.

    When a block straddles filters, each filter selects only its own slots from
    the complete measured binary LUT; other slots remain at their baseline.
    """
    slots = [(p.block.name, s) for p in selected for s in range(len(p.block.bls))]
    return tuple(Assignment(i // len(slots), *slots[i % len(slots)], i // 25, i % 25) for i in range(64 * 25))


class Programmer(WriteCollector):
    def __init__(self, connector: Connector, cv: Converters, writer: Writer, source: Source, span: float) -> None:
        super().__init__(connector, cv, writer, 2_000_000, 350_000, span, lambda s: print(s, flush=True))
        self.source = source

    def emit(self, event: str, **fields: object) -> None:
        self.writer.write(dict(event=event, source=self.source, monotonic_s=time.monotonic(), **fields))

    def program(self, wl: int, bl: int, goal: float, **metadata: object) -> dict[str, object]:
        if not math.isfinite(goal) or goal <= 0:
            raise ValueError('Invalid conductance target')
        g = self.read_g(wl, bl, role='initial', **metadata)
        initial = g
        # Same bounded feedback rule as the trained experiment. No compulsory
        # reset sweep: a cell already near its target receives no write pulse.
        attempts = 0
        for attempt in range(96):
            if abs(g - goal) <= .02 * self.span:
                break
            sign = int(g < goal)
            g = self.pulse(wl, bl, sign, 1.7 if sign else 1.4, .00001,
                           target_g_s=goal, attempt=attempt, **metadata)
            attempts += 1
        return dict(wl=wl, bl=bl, initial_g_s=initial, target_g_s=goal,
                    last_g_s=g, attempts=attempts)


def acquire_load(reader: ReadCollector, selected: tuple[Profile, ...], out: Path, load: int) -> tuple[list[dict[str, object]], bool]:
    """Measure ALL states (3 repeats), freeze, then validate (5 independent repeats).

    A zero weight/low baseline is allowed. Earlier current-state qualification
    required resolved individual legs; programmed zeros need not meet that
    criterion. This new protocol checks repeatability of the entire measured
    operator, not agreement with an ideal sum. No profile or mask is selected
    using labels or accuracy. Every state and every failure is retained.
    """
    reader.pairs.clear()
    reader.started = time.monotonic()
    reader.calls = 0
    for phase, repeats in (('basis', 3), ('validation', 5)):
        for repeat in range(repeats):
            order = selected if repeat % 2 == 0 else tuple(reversed(selected))
            for profile in order:
                block = profile.block
                masks = patterns(block)
                for targets in (masks if repeat % 2 == 0 else tuple(reversed(masks))):
                    leg_order = (block.plus_wl, block.minus_wl) if repeat % 2 == 0 else (block.minus_wl, block.plus_wl)
                    kind: Phase = 'basis' if phase == 'basis' else 'validation'
                    mode: Mode = 'fixed_positive' if profile.biased else 'ordinary'
                    legs = {wl: reader.measure_leg(block, wl, kind, repeat, mode, profile.delta, targets) for wl in leg_order}
                    pair = Pair(block.name, kind, repeat, mode, profile.delta, targets, leg_order, legs[block.plus_wl], legs[block.minus_wl])
                    reader.pairs.append(pair)
                    reader.emit('transfer_bracket', load=load, **asdict(pair))
        if phase == 'basis':
            atomic_json(out / f'load_{load:03d}_frozen.json', [asdict(r) for r in reader.pairs])
    operators: list[dict[str, object]] = []
    all_pass = True
    adc_step = reader.cv.convert_adc_to_volt(1, 14, 5, 1) - reader.cv.convert_adc_to_volt(1, 14, 5, 0)
    for profile in selected:
        b = profile.block
        lut = [0.0] * (1 << len(b.bls))
        checks: list[dict[str, object]] = []
        for mask, targets in enumerate(patterns(b), 1):
            basis = [p for p in reader.pairs if p.block == b.name and p.targets == targets and p.phase == 'basis']
            validation = [p for p in reader.pairs if p.block == b.name and p.targets == targets and p.phase == 'validation']
            pmean = statistics.mean(p.plus.centered('mean4') for p in basis)
            mmean = statistics.mean(p.minus.centered('mean4') for p in basis)
            signed = pmean - mmean
            bl = targets[0]
            leg = basis[0].plus
            delta = leg.signal_decoded_v[bl] - leg.baseline_decoded_v[bl]
            # CPU unit conversion only. Span is applied later. This is not a
            # regression against target weights and does not hide write errors.
            lut[mask] = signed * adc_step / (330. * delta)
            values = [p.plus.centered('mean4') - p.minus.centered('mean4') for p in validation]
            drift = max(abs(l.drift('mean4')) for p in basis + validation for l in (p.plus, p.minus))
            signed_drift = max(abs(p.plus.drift('mean4') - p.minus.drift('mean4')) for p in basis + validation)
            sd = statistics.stdev(values)
            residual = abs(statistics.mean(values) - signed)
            limit = max(5., .1 * (abs(pmean) + abs(mmean)))
            passed = drift <= 5 and signed_drift <= 5 and sd <= 5 and residual <= limit
            all_pass &= passed
            checks.append(dict(mask=mask, passed=passed, maximum_leg_drift_adc=drift,
                signed_drift_adc=signed_drift, signed_sd_adc=sd, heldout_error_adc=residual,
                heldout_limit_adc=limit, frozen_signed_adc=signed))
        operators.append(dict(block=b.name, lut_difference_g_s=lut, checks=checks))
    atomic_json(out / f'load_{load:03d}_operators.json', dict(operators=operators, passed=all_pass))
    return operators, all_pass


def compose(keys: NDArray[np.uint32], assignments: tuple[Assignment, ...], operators: list[dict[str, object]],
            span: float, target: NDArray[np.float32]) -> None:
    """CPU adds measured partial MVMs BEFORE the logical LIF tick."""
    for operator in operators:
        assigned = [a for a in assignments if a.block == operator['block']]
        lut = np.asarray(operator['lut_difference_g_s'], dtype=np.float64) / span
        for filt in sorted({a.filter for a in assigned}):
            local = np.zeros(len(keys), dtype=np.uint32)
            for a in assigned:
                if a.filter == filt:
                    local |= ((keys >> np.uint32(a.tap)) & np.uint32(1)) << np.uint32(a.slot)
            target[:, filt] += lut[local].astype(np.float32)
