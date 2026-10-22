"""Проверка 15 выбранных блоков: реальный MVM, четыре чтения на уровень.

Все токи и отклики измеряет публичный Connector. Цифровые операции здесь —
усреднение, вычитание baseline/ног и проверка предсказания из измеренного basis.
Ни весовое программирование, ни нейросеть, ни идеальный W@x не выполняются.
"""
from __future__ import annotations

from collections.abc import Callable
import contextlib
from dataclasses import asdict, dataclass
import hashlib
import html
import json
import math
import statistics
import time
from typing import TYPE_CHECKING, Literal

from experiment_common.durable import Writer
from experiment_common.native_output import DriverOutput, safe
from experiment_common.selective_metrology import Converters, Reading, Source, number

if TYPE_CHECKING:
    from manager.board import Connector

Mode = Literal['ordinary', 'fixed_positive']
Phase = Literal['basis', 'validation']
Policy = Literal['mean4', 'first_sample']


@dataclass(frozen=True)
class Block:
    name: str
    plus_wl: int
    minus_wl: int
    bls: tuple[int, ...]


# Точная прежняя карта: каждая (WL,BL) принадлежит одному блоку. Порядок BL
# фиксирован до измерения; плохой input не удаляется из блока при анализе.
BLOCKS = (
    Block('B01', 0, 1, (2, 4, 5, 6, 9, 12)),
    Block('B02', 0, 3, (8, 13)), Block('B03', 0, 4, (11,)),
    Block('B04', 1, 2, (15,)), Block('B05', 1, 3, (10,)),
    Block('B06', 1, 4, (14,)), Block('B07', 1, 5, (7,)),
    Block('B08', 2, 3, (5, 6, 9, 12, 31)), Block('B09', 2, 5, (0,)),
    Block('B10', 3, 4, (2, 3)), Block('B11', 4, 5, (1, 10, 15)),
    Block('B12', 4, 6, (13,)), Block('B13', 5, 6, (3, 5, 14)),
    Block('B14', 5, 7, (6, 8, 9, 31)), Block('B15', 6, 7, (10,)),
)
LEVELS: tuple[tuple[Mode, float], ...] = (
    ('ordinary', .05), ('fixed_positive', .025), ('fixed_positive', .05),
)
MAX_CALLS = 74518
SPEC: dict[str, object] = dict(
    schema='selected-block-readout-mean4-v1', blocks=[asdict(b) for b in BLOCKS],
    maximum_impacts=MAX_CALLS, maximum_pulses=0, maximum_wall_seconds=600,
    basis_repeats=3, validation_repeats=5, samples_per_level=4,
    readouts=[dict(mode=m, delta_v=d) for m, d in LEVELS],
    fixed_bias_v=.025, validation_patterns='all 137 nonempty binary masks across 15 blocks',
    pattern_order='ascending/reversed binary mask on alternating repeats',
    block_order='forward/reverse on alternating repeats',
    leg_order='plus/minus then minus/plus on alternating repeats',
    bracket='fresh R all active -> baseline x4 -> signal x4 -> baseline x4',
    primary_policy='mean4', first_sample_policy='descriptive only, never chosen after outcomes',
    binary_profile_priority=['ordinary_delta05', 'biased_delta025', 'biased_delta05'],
    cross_amplitude_linearity='diagnostic only; not a binary-profile qualification gate',
    current_limit_a=.005, predicted_headroom_v=3.3, measured_headroom_v=4.,
    screening=dict(leg_mean='> max(5 ADC, 5*SD across brackets)',
        maximum_abs_baseline_drift_adc=5,
        leg_prediction_error='<= max(5 ADC, .1*abs(predicted))',
        signed_prediction_error='<= max(5 ADC, .1*(abs(predPlus)+abs(predMinus)))',
        signed_maximum_sd_adc=5, signed_maximum_abs_drift_adc=5,
        cancellation='zero_or_unresolved annotation; not a failure by itself',
        gain_linearity='fixed-positive only, actual DAC increment ratio; no ordinary multi-amplitude claim'),
    no_subselection=True, qualified_scope='stable_current_state_readout_only',
    programming_qualified=False, mnist_qualified=False,
    private_runtime_status='private_runtime_not_independently_reverified',
)


def patterns(block: Block) -> tuple[tuple[int, ...], ...]:
    """Complete binary SNN input coverage without changing the BL mapping."""
    return tuple(tuple(bl for index, bl in enumerate(block.bls) if mask & (1 << index))
                 for mask in range(1, 1 << len(block.bls)))


@dataclass(frozen=True)
class Leg:
    wl: int
    guards: tuple[Reading, ...]
    before: tuple[Reading, ...]
    signal: tuple[Reading, ...]
    after: tuple[Reading, ...]
    baseline_codes: tuple[int, ...]
    signal_codes: tuple[int, ...]
    baseline_decoded_v: tuple[float, ...]
    signal_decoded_v: tuple[float, ...]
    current_a: float

    def centered(self, policy: Policy) -> float:
        def value(samples: tuple[Reading, ...]) -> float:
            return statistics.mean(r['adc'] for r in samples) if policy == 'mean4' else float(samples[0]['adc'])
        return value(self.signal) - (value(self.before) + value(self.after)) / 2

    def drift(self, policy: Policy) -> float:
        if policy == 'first_sample':
            return float(self.after[0]['adc'] - self.before[0]['adc'])
        return statistics.mean(r['adc'] for r in self.after) - statistics.mean(r['adc'] for r in self.before)


@dataclass(frozen=True)
class Pair:
    block: str
    phase: Phase
    repeat: int
    mode: Mode
    delta_v: float
    targets: tuple[int, ...]
    leg_order: tuple[int, int]
    plus: Leg
    minus: Leg


class Collector:
    def __init__(self, connector: Connector, converters: Converters, writer: Writer, source: Source) -> None:
        self.connector: Connector = connector
        self.cv, self.writer, self.source = converters, writer, source
        self.calls = 0
        self.started = time.monotonic()
        self.pairs: list[Pair] = []
        self.basis_sha256: str | None = None

    def emit(self, event: str, **fields: object) -> None:
        self.writer.write(dict(event=event, source=self.source, monotonic_s=time.monotonic(), **fields))

    def call(self, command: dict[str, int | float | list[float]], **metadata: object) -> Reading:
        if self.calls >= MAX_CALLS or time.monotonic() - self.started >= 600:
            raise TimeoutError('Finite block validation budget')
        mode, wl = command.get('mode_flag'), command.get('wl')
        if type(wl) is not int or not isinstance(wl, int) or not 0 <= wl < 8:
            raise ValueError('Invalid WL')
        if mode == 7:
            bl = command.get('bl')
            if type(bl) is not int or not isinstance(bl, int) or not 0 <= bl < 32 or command.get('vol') != 0. or command.get('sign') != 0 or command.get('pulse_width') != 0.:
                raise ValueError('Only zero-write mode7 reads permitted')
        elif mode == 10:
            vol = command.get('vol')
            if not isinstance(vol, list) or len(vol) != 32 or any(v not in (0., .025, .05, .075) for v in vol):
                raise ValueError('Invalid planned MVM voltage')
        else:
            raise ValueError('Read-only mode7/mode10')
        self.calls += 1
        ticket = {**command, 'id': self.calls if mode == 7 else 0}
        self.emit('request', task_id=self.calls, command_si=ticket, metadata=metadata)
        values = ticket['vol']
        copied = {**ticket, 'vol': list(values) if isinstance(values, list) else values}
        with contextlib.redirect_stdout(DriverOutput(self, 'stdout')), contextlib.redirect_stderr(DriverOutput(self, 'stderr')):
            result = self.connector.impact(copied)
        self.emit('result', task_id=self.calls, command_si=ticket, native_result=safe(result))
        if not isinstance(result, (tuple, list)) or len(result) != 3:
            raise ValueError('Invalid native response')
        resistance, echo, adc = result
        if type(echo) is not int or echo != ticket['id'] or type(adc) is not int or not 0 <= adc <= 16382:
            raise ValueError('Invalid ADC/ID')
        r = number(resistance)
        if mode == 7 and (r <= 1e-8 or adc < 20):
            raise ValueError('Invalid resistance measurement')
        if mode == 10:
            voltage = self.cv.convert_adc_to_volt(1, 14, 5, adc)
            if not math.isfinite(voltage) or voltage > 4.:
                raise ValueError('Measured ADC headroom')
        return Reading(task_id=self.calls, resistance_ohm=r, adc=adc)

    def measure_leg(self, block: Block, wl: int, phase: Phase, repeat: int,
                    mode: Mode, delta: float, targets: tuple[int, ...], *, global_bias: bool = False) -> Leg:
        if wl not in (block.plus_wl, block.minus_wl) or not targets or any(b not in block.bls for b in targets):
            raise ValueError('Unplanned block coordinates')
        if global_bias and mode != 'fixed_positive':
            raise ValueError('Global bias requires fixed-positive mode')
        active = tuple(range(32)) if global_bias else (targets if mode == 'ordinary' else block.bls)
        baseline = [(.025 if b in active else 0.) if mode == 'fixed_positive' else 0. for b in range(32)]
        signal = [round(baseline[b] + delta, 3) if b in targets else baseline[b] for b in range(32)]
        base_codes = tuple(self.cv.convert_volt_to_dac(12, 5, v) for v in baseline)
        codes = tuple(self.cv.convert_volt_to_dac(12, 5, v) for v in signal)
        base_decoded = tuple(self.cv.convert_dac_to_volt(12, 5, c) for c in base_codes)
        decoded = tuple(self.cv.convert_dac_to_volt(12, 5, c) for c in codes)
        meta: dict[str, object] = dict(block=block.name, phase=phase, repeat=repeat, mode=mode, delta_v=delta, targets=list(targets), global_bias=global_bias)
        guards = tuple(self.call(dict(mode_flag=7, wl=wl, bl=b, vol=0., sign=0, pulse_width=0.), role='guard', **meta) for b in active)
        current = sum(max(signal[b], decoded[b]) / r['resistance_ohm'] for b, r in zip(active, guards))
        if current > .005 or 330 * current >= 3.3:
            raise ValueError('Fresh summed current/headroom guard')
        # Усреднение четырёх аппаратных выборок — часть readout, а не выбор
        # удобных повторов. Все четыре ответа сохраняются независимо.
        before = tuple(self.call(dict(mode_flag=10, wl=wl, vol=baseline), role='before', sample=i, **meta) for i in range(4))
        response = tuple(self.call(dict(mode_flag=10, wl=wl, vol=signal), role='signal', sample=i, **meta) for i in range(4))
        after = tuple(self.call(dict(mode_flag=10, wl=wl, vol=baseline), role='after', sample=i, **meta) for i in range(4))
        return Leg(wl, guards, before, response, after, base_codes, codes, base_decoded, decoded, current)

    def diagnose(self, progress: Callable[[str], None]) -> None:
        for phase, repeats in (('basis', 3), ('validation', 5)):
            phase_name: Phase = 'basis' if phase == 'basis' else 'validation'
            for repeat in range(repeats):
                ordered = BLOCKS if repeat % 2 == 0 else tuple(reversed(BLOCKS))
                for block in ordered:
                    targets = tuple((b,) for b in block.bls) if phase_name == 'basis' else patterns(block)
                    if repeat % 2:
                        targets = tuple(reversed(targets))
                    for mask in targets:
                        for mode, delta in LEVELS:
                            order = (block.plus_wl, block.minus_wl) if repeat % 2 == 0 else (block.minus_wl, block.plus_wl)
                            legs = {wl: self.measure_leg(block, wl, phase_name, repeat, mode, delta, mask) for wl in order}
                            row = Pair(block.name, phase_name, repeat, mode, delta, mask, order, legs[block.plus_wl], legs[block.minus_wl])
                            self.pairs.append(row)
                            self.emit('paired_bracket', **asdict(row))
                    self.writer.checkpoint(phase=phase, repeat=repeat, block=block.name, impacts=self.calls)
                progress(f'{phase} repeat {repeat + 1}/{repeats}: {self.calls}/{MAX_CALLS}; 0 writes')
            if phase_name == 'basis':
                payload = json.dumps([asdict(r) for r in self.pairs], sort_keys=True, separators=(',', ':')).encode()
                self.basis_sha256 = hashlib.sha256(payload).hexdigest()
                self.emit('basis_frozen', basis_sha256=self.basis_sha256, pairs=len(self.pairs), primary_policy='mean4')


def statistics_row(rows: list[Pair], leg: Literal['plus', 'minus', 'signed'], policy: Policy) -> dict[str, object]:
    values: list[float] = []
    drifts: list[float] = []
    for row in rows:
        if leg == 'signed':
            values.append(row.plus.centered(policy) - row.minus.centered(policy))
            drifts.append(row.plus.drift(policy) - row.minus.drift(policy))
        else:
            measured = row.plus if leg == 'plus' else row.minus
            values.append(measured.centered(policy))
            drifts.append(measured.drift(policy))
    mean = statistics.mean(values) if values else None
    sd = statistics.stdev(values) if len(values) > 1 else None
    drift = max(map(abs, drifts), default=None)
    return dict(n=len(values), values_adc=values, mean_adc=mean, sd_adc=sd,
        drift_adc=drifts, maximum_abs_drift_adc=drift,
        drift_pass=drift is not None and drift <= 5,
        resolvable=mean is not None and sd is not None and mean > max(5., 5 * sd),
        zero_or_unresolved=mean is None or sd is None or abs(mean) <= max(5., 5 * sd))


def assess(rows: list[Pair], policy: Policy) -> list[dict[str, object]]:
    reports: list[dict[str, object]] = []
    for block in BLOCKS:
        for mode, profile_delta in LEVELS:
            deltas = (profile_delta,)
            profile = 'ordinary_delta05' if mode == 'ordinary' else ('biased_delta025' if profile_delta == .025 else 'biased_delta05')
            subset = [r for r in rows if r.block == block.name and r.mode == mode]
            reasons: list[str] = []
            basis: list[dict[str, object]] = []
            means: dict[tuple[int, float, str], float] = {}
            codes: dict[tuple[int, float], int] = {}
            complete = True
            nonpositive = False
            for bl in block.bls:
                for delta in deltas:
                    found = [r for r in subset if r.phase == 'basis' and r.targets == (bl,) and r.delta_v == delta]
                    if len(found) != 3:
                        complete = False
                    for leg in ('plus', 'minus'):
                        leg_name: Literal['plus', 'minus'] = 'plus' if leg == 'plus' else 'minus'
                        stat = statistics_row(found, leg_name, policy)
                        basis.append(dict(bl=bl, delta_v=delta, leg=leg, **stat))
                        if not stat['resolvable']:
                            reasons.append(f'basis_{leg}_BL{bl}_delta{delta}:unresolved')
                        if not stat['drift_pass']:
                            reasons.append(f'basis_{leg}_BL{bl}_delta{delta}:drift')
                        if stat['mean_adc'] is not None:
                            mean = number(stat['mean_adc'])
                            means[bl, delta, leg] = mean
                            nonpositive |= mean <= 0
                    if found:
                        codes[bl, delta] = found[0].plus.signal_codes[bl] - found[0].plus.baseline_codes[bl]
                    signed_stat = statistics_row(found, 'signed', policy)
                    basis.append(dict(bl=bl, delta_v=delta, leg='signed', **signed_stat))
                    if not signed_stat['drift_pass']:
                        reasons.append(f'basis_signed_BL{bl}_delta{delta}:drift')
                    if signed_stat['sd_adc'] is None or number(signed_stat['sd_adc']) > 5:
                        reasons.append(f'basis_signed_BL{bl}_delta{delta}:noise')
            linearity: list[dict[str, object]] = []
            if mode == 'fixed_positive':
                for bl in block.bls:
                    low = [r for r in subset if r.phase == 'basis' and r.targets == (bl,) and r.delta_v == .025]
                    high = [r for r in subset if r.phase == 'basis' and r.targets == (bl,) and r.delta_v == .05]
                    if len(low) != 3 or len(high) != 3:
                        continue
                    low_code = low[0].plus.signal_codes[bl] - low[0].plus.baseline_codes[bl]
                    high_code = high[0].plus.signal_codes[bl] - high[0].plus.baseline_codes[bl]
                    if low_code <= 0:
                        continue
                    ratio = high_code / low_code
                    lp = statistics.mean(r.plus.centered(policy) for r in low)
                    lm = statistics.mean(r.minus.centered(policy) for r in low)
                    hp = statistics.mean(r.plus.centered(policy) for r in high)
                    hm = statistics.mean(r.minus.centered(policy) for r in high)
                    for leg in ('plus', 'minus', 'signed'):
                        if leg == 'signed':
                            prediction = (lp-lm)*ratio
                            measured = hp-hm
                            scale = (abs(lp)+abs(lm))*ratio
                        else:
                            prediction = (lp if leg == 'plus' else lm)*ratio
                            measured = hp if leg == 'plus' else hm
                            scale = abs(prediction)
                        error = measured - prediction
                        passed = abs(error) <= max(5., .1 * scale)
                        linearity.append(dict(bl=bl, leg=leg, dac_increment_ratio=ratio, predicted_adc=prediction, measured_adc=measured, error_adc=error, limit_adc=max(5., .1*scale), passed=passed))
            validation: list[dict[str, object]] = []
            for mask in patterns(block):
                for delta in deltas:
                    found = [r for r in subset if r.phase == 'validation' and r.targets == mask and r.delta_v == delta]
                    if len(found) != 5:
                        complete = False
                    predictions = {leg: sum(means[b, delta, leg] for b in mask) for leg in ('plus', 'minus')} if all((b, delta, leg) in means for b in mask for leg in ('plus', 'minus')) else {}
                    stats = {leg: statistics_row(found, leg, policy) for leg in ('plus', 'minus', 'signed')}
                    for leg, stat in stats.items():
                        if not stat['drift_pass']:
                            reasons.append(f'validation_{leg}_{mask}_delta{delta}:drift')
                        if leg != 'signed' and not stat['resolvable']:
                            reasons.append(f'validation_{leg}_{mask}_delta{delta}:unresolved')
                        if leg == 'signed' and (stat['sd_adc'] is None or number(stat['sd_adc']) > 5):
                            reasons.append(f'validation_signed_{mask}_delta{delta}:noise')
                        if predictions and stat['mean_adc'] is not None:
                            predicted = predictions[leg] if leg != 'signed' else predictions['plus'] - predictions['minus']
                            scale = abs(predicted) if leg != 'signed' else abs(predictions['plus']) + abs(predictions['minus'])
                            error = number(stat['mean_adc']) - predicted
                            stat.update(predicted_adc=predicted, error_adc=error, absolute_error_adc=abs(error),
                                relative_error=error/max(scale, 1.), relative_denominator_adc=max(scale, 1.),
                                prediction_limit_adc=max(5., .1*scale), prediction_pass=abs(error) <= max(5., .1*scale))
                            if not stat['prediction_pass']:
                                reasons.append(f'validation_{leg}_{mask}_delta{delta}:prediction')
                    mean_plus, mean_minus = stats['plus']['mean_adc'], stats['minus']['mean_adc']
                    condition = None
                    if mean_plus is not None and mean_minus is not None:
                        p, m = number(mean_plus), number(mean_minus)
                        condition = (abs(p)+abs(m))/max(abs(p-m), 1.)
                    validation.append(dict(targets=list(mask), delta_v=delta, plus=stats['plus'], minus=stats['minus'], signed=stats['signed'], kappa_adc_floor_1=condition,
                        combination_unseen_in_basis=len(mask)>1))
            if not complete:
                reasons.append('incomplete_schedule')
            # Полный, но шумный/дрейфующий блок остаётся candidate с причинами.
            # Нулевой signed weight сам по себе не исключает блок.
            status = 'fail' if not complete or nonpositive else ('candidate' if reasons else 'stable_current_state_readout')
            reports.append(dict(block=block.name, wl_pair=[block.plus_wl, block.minus_wl], bls=list(block.bls), mode=mode,
                profile=profile, delta_v=profile_delta,
                policy=policy, status=status, reasons=sorted(set(reasons)), complete=complete,
                nonpositive_leg_basis=nonpositive, basis=basis, multi_amplitude_linearity_diagnostic=linearity, validation=validation,
                programming_qualified=False, mnist_qualified=False))
    return reports


def summarize(rows: list[Pair]) -> dict[str, object]:
    primary = assess(rows, 'mean4')
    diagnostic = assess(rows, 'first_sample')
    selected: list[dict[str, object]] = []
    for block in BLOCKS:
        valid = [r for r in primary if r['block'] == block.name and r['status'] == 'stable_current_state_readout']
        if valid:
            # При равном pass заранее предпочитаем простой ordinary. Не сравниваем
            # test accuracy и не выбираем first-sample вместо mean4.
            chosen = valid[0]  # LEVELS задаёт ordinary05 → biased025 → biased05.
            selected.append(dict(block=block.name, wl_pair=[block.plus_wl, block.minus_wl], bls=list(block.bls),
                mode=chosen['mode'], profile=chosen['profile'], delta_v=chosen['delta_v'],
                samples_per_level=4, coefficient_slots=len(block.bls),
                qualification='stable_current_state_readout_only'))
    return dict(primary_policy='mean4', primary=primary, first_sample_diagnostic=diagnostic,
        selected_blocks=selected, selected_coefficients=sum(int(number(r['coefficient_slots'])) for r in selected),
        selected_cells=2*sum(int(number(r['coefficient_slots'])) for r in selected),
        inputs_removed=False, programming_qualified=False, mnist_qualified=False)


def render(summary: dict[str, object]) -> str:
    return ('<!doctype html><html lang="ru"><meta charset="utf-8"><title>15 блоков: readout</title>'
        '<style>body{font:16px system-ui;max-width:1100px;margin:auto}pre{white-space:pre-wrap}</style>'
        '<h1>Проверка 15 блоков для последовательного readout</h1>'
        '<p>Четыре реальные выборки на уровень. Stable означает только нынешний readout; '
        'программируемость и MNIST не квалифицированы. Входы внутри блока не исключались. '
        'Zero/unresolved signed response не является автоматическим отказом.</p><pre>'
        + html.escape(json.dumps(summary, ensure_ascii=False, indent=2)) + '</pre></html>')
