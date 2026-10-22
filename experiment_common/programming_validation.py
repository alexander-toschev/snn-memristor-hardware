"""Ограниченная проверка программируемости выбранных16cells MemriBoard.

Исходные R/M/H recipes и критерии20Oct. Здесь RESET меняет сопротивление,
не обнуляет потенциал LIF. Ни нейросети, ни идеального W@x здесь нет.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import contextlib
import math
import statistics
import time
from typing import TYPE_CHECKING, Callable

from experiment_common.durable import Writer
from experiment_common.native_output import DriverOutput, safe
from experiment_common.selective_metrology import Bracket, Cell, Converters, Reading, Source, number

if TYPE_CHECKING:
    from manager.board import Connector

LEVELS = (.3, .6, .9, 1.2, 1.5, 2., 2.5, 3.)
BLS = (2, 3, 6, 10, 11, 12, 14, 15)
CELLS = tuple((wl, bl) for wl in (0, 7) for bl in BLS)


def recipe(state: str, incremental: bool = False) -> list[tuple[int, float]]:
    if state == 'R':
        return [(0, level) for level in LEVELS]
    if state not in ('M', 'H'):
        raise ValueError('Unknown state recipe')
    if incremental:
        return [(1, level) for level in (LEVELS[:4] if state == 'M' else LEVELS[4:])]
    return recipe('R') + [(1, level) for level in LEVELS[:4 if state == 'M' else 8]]


def pair_states(configuration: str, index: int) -> tuple[str, str]:
    if configuration == 'zero':
        return 'M', 'M'
    if configuration == 'plus' or configuration == 'checkerboard' and index % 2 == 0:
        return 'H', 'R'
    if configuration in ('minus', 'checkerboard'):
        return 'R', 'H'
    raise ValueError('Unknown pair configuration')


@dataclass(frozen=True)
class Point:
    wl: int
    bl: int
    cycle: int
    state: str
    role: str
    mean: float
    sd: float
    zero_mean: float
    zero_limit: float
    mean_g_s: float
    time: float


@dataclass(frozen=True)
class Pulse:
    phase: str
    wl: int
    bl: int
    cycle: int
    sign: int
    level: float
    admitted: bool


@dataclass(frozen=True)
class PairPoint:
    bl: int
    configuration: str
    role: str
    mean: float
    sd: float
    time: float


def assess_cells(points: list[Point], updates: list[Pulse]) -> list[dict[str, object]]:
    """Точные исходные lower-span/SNR/noise/drift/cycle формулы, без rerank."""
    reports: list[dict[str, object]] = []
    needed = [(cycle, state, role) for cycle in range(2) for state in ('R', 'M', 'H')
              for role in ('immediate', 'delay_0', 'delay_10', 'delay_60')] + [(-1, 'H', 'final')]
    for wl, bl in CELLS:
        by = {(p.cycle, p.state, p.role): p for p in points if (p.wl, p.bl) == (wl, bl)}
        hard: list[str] = []
        reasons: list[str] = []
        row: dict[str, object] = dict(wl=wl, bl=bl, hard_eligible=False, eligible=False,
                                      hard_reasons=hard, reasons=reasons)
        if any(key not in by for key in needed):
            hard.append('missing_required_state_checkpoint')
            reasons.extend(hard)
            reports.append(row)
            continue
        spans: list[dict[str, float | int]] = []
        for cycle in range(2):
            low, high = by[cycle, 'R', 'immediate'], by[cycle, 'H', 'immediate']
            noise = math.hypot(low.sd, high.sd)
            span = high.mean - low.mean - 3 * noise / math.sqrt(10)
            spans.append(dict(cycle=cycle, lower_span_adc=span, noise_adc=noise))
            admitted = [u for u in updates if u.phase == 'A' and u.cycle == cycle
                        and (u.wl, u.bl) == (wl, bl) and u.admitted]
            if not all(any(u.sign == sign for u in admitted) for sign in (0, 1)):
                hard.append('no_admitted_reset_or_set_cycle_' + str(cycle))
        span = min(s['lower_span_adc'] for s in spans)
        if any(span < max(10., 5 * s['noise_adc']) for s in spans):
            hard.append('insufficient_reproducible_span')
        reasons.extend(hard)
        sd_limit, drift_limit, cycle_limit = max(2., .05 * span), max(3., .05 * span), max(4., .1 * span)
        scores: list[float] = []
        for key in needed:
            cycle, state, role = key
            point = by[key]
            reference = by[cycle if cycle >= 0 else 1, state, 'immediate']
            drift = abs(point.mean - reference.mean)
            scores.extend([point.sd / sd_limit, drift / drift_limit, abs(point.zero_mean) / point.zero_limit])
            if point.sd > sd_limit:
                reasons.append('repeat_noise:' + str(key))
            if drift > drift_limit:
                reasons.append('state_drift:' + str(key))
            if abs(point.zero_mean) > point.zero_limit:
                reasons.append('zero_shift:' + str(key))
        for state in ('R', 'M', 'H'):
            delta = abs(by[0, state, 'immediate'].mean - by[1, state, 'immediate'].mean)
            scores.append(delta / cycle_limit)
            if delta > cycle_limit:
                reasons.append('cross_cycle:' + state)
        for cycle in range(2):
            a, m, b = [by[cycle, state, 'immediate'].mean for state in ('R', 'M', 'H')]
            scores.append(max(0., a - m, m - b) / cycle_limit)
            if not a - cycle_limit <= m <= b + cycle_limit:
                reasons.append('middle_outside_endpoints')
        row.update(span_adc=span, spans=spans, worst_normalized_error=max(scores),
                   hard_eligible=not hard, eligible=not reasons, initial_checkpoint_descriptive_only=True)
        reports.append(row)
    return reports


def assess_pairs(points: list[PairPoint], cells: list[dict[str, object]]) -> dict[str, object]:
    by = {(number(row['wl']), number(row['bl'])): row for row in cells}
    reports: list[dict[str, object]] = []
    mandatory: list[bool] = []
    qualities: list[bool] = []
    for bl in BLS:
        span = min(number(by[wl, bl]['span_adc']) for wl in (0, 7))
        hard: list[str] = []
        reasons: list[str] = []
        configurations: dict[str, object] = {}
        for configuration in ('plus', 'minus', 'zero', 'checkerboard'):
            keyed = {p.role: p for p in points if p.bl == bl and p.configuration == configuration}
            values: dict[str, object] = {}
            for role in ('immediate', 'delayed'):
                point = keyed.get(role)
                if point is None:
                    hard.append('missing:' + configuration + ':' + role)
                    reasons.append(hard[-1])
                    continue
                values[role] = dict(mean=point.mean, sd=point.sd, count=10)
                if point.sd > max(2., .05 * span):
                    reasons.append('noise:' + configuration + ':' + role)
                direction = 1 if configuration == 'plus' else -1 if configuration == 'minus' else 0
                if direction and direction * point.mean < max(5., 5 * point.sd):
                    hard.append('signed_response:' + configuration + ':' + role)
                    reasons.append(hard[-1])
            if 'immediate' in keyed and 'delayed' in keyed:
                if abs(keyed['delayed'].mean - keyed['immediate'].mean) > max(3., .05 * span):
                    reasons.append('drift:' + configuration)
            configurations[configuration] = values
        mandatory.append(not hard)
        qualities.append(not reasons)
        reports.append(dict(pair=[0, 7], bl=bl, pair_span_adc=span, reasons=reasons,
                            hard_reasons=hard, mandatory_pass=not hard, qualified=not reasons,
                            configurations=configurations))
    return dict(qualified=all(qualities), mandatory_pass=all(mandatory), cells=reports,
                rerank=False, zero_recipe_is_not_zero_weight=True)


class Collector:
    def __init__(self, connector: Connector, converters: Converters, writer: Writer,
                 source: Source, progress: Callable[[str], None], *,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.connector: Connector = connector
        self.cv, self.writer, self.source = converters, writer, source
        self.progress, self.clock, self.sleep = progress, clock, sleep
        self.calls = self.pulses = self.decisions = 0
        self.phase = 'A'
        self.phase_calls = dict(A=0, B=0)
        self.phase_pulses = dict(A=0, B=0)
        self.started = clock()
        self.last_decision = self.started
        self.waited = 0.
        self.points: list[Point] = []
        self.updates: list[Pulse] = []
        self.pairs: list[PairPoint] = []
        self.cells: list[dict[str, object]] = []
        self.pair_report: dict[str, object] = {}
        self.status = 'failed'

    def emit(self, event: str, **fields: object) -> None:
        self.writer.write(dict(event=event, source=self.source, phase=self.phase,
                               monotonic_s=self.clock(), **fields))

    def budget(self) -> None:
        if self.clock() - self.started >= 1200:
            raise TimeoutError('1200second finite wall budget')
        if self.calls >= 24320 or self.phase_calls[self.phase] >= (17664 if self.phase == 'A' else 6656):
            raise ValueError('Finite impact budget')

    def call(self, command: dict[str, int | float | list[float]], **metadata: object) -> Reading:
        self.budget()
        mode, wl = command['mode_flag'], command['wl']
        if type(wl) is not int or wl not in (0, 7) or type(mode) is not int:
            raise ValueError('Outside selected WL')
        if mode == 7:
            if type(command.get('bl')) is not int or type(command.get('sign')) is not int or command.get('bl') not in BLS or command.get('sign') not in (0, 1):
                raise ValueError('Outside selected cell/polarity')
            if command.get('vol') not in (0., *LEVELS):
                raise ValueError('Unreviewed pulse voltage')
            width = 0. if command['vol'] == 0. or self.source == 'hardware' else .000125
            if command.get('pulse_width') != width:
                raise ValueError('Invalid source-specific submitted width')
        elif mode == 10:
            voltage = command.get('vol')
            if not isinstance(voltage, list) or len(voltage) != 32 or sum(v > 0 for v in voltage) > 1:
                raise ValueError('Expected zero/singleton vector')
            if any(v not in (0., .05) or v > 0 and bl not in BLS for bl, v in enumerate(voltage)):
                raise ValueError('Unreviewed mode10 voltage/address')
        else:
            raise ValueError('Only public mode7/mode10')
        self.calls += 1
        self.phase_calls[self.phase] += 1
        task = {**command, 'id': self.calls if mode == 7 else 0}
        self.emit('request', task_id=self.calls, command_si=task, metadata=metadata)
        voltage = task['vol']
        with contextlib.redirect_stdout(DriverOutput(self, 'stdout')), contextlib.redirect_stderr(DriverOutput(self, 'stderr')):
            result = self.connector.impact({**task, 'vol': list(voltage) if isinstance(voltage, list) else voltage})
        self.emit('result', task_id=self.calls, command_si=task, native_result=safe(result))
        if not isinstance(result, (list, tuple)) or len(result) != 3:
            raise ValueError('Invalid native response')
        resistance, echo, adc = result
        if type(echo) is not int or echo != task['id'] or type(adc) is not int or not 0 <= adc <= 16382:
            raise ValueError('Invalid native ID/ADC')
        r = number(resistance)
        if mode == 7 and (r <= 1e-8 or adc < 20):
            raise ValueError('Invalid native resistance/readADC')
        return Reading(task_id=self.calls, resistance_ohm=r, adc=adc)

    def read(self, wl: int, bl: int, **metadata: object) -> Reading:
        return self.call(dict(mode_flag=7, wl=wl, bl=bl, vol=0., sign=0, pulse_width=0.), **metadata)

    def pulse(self, wl: int, bl: int, sign: int, level: float, cycle: int, **metadata: object) -> None:
        if sign not in (0, 1) or level not in LEVELS or self.decisions >= 1280:
            raise ValueError('Invalid finite pulse decision')
        guard = self.read(wl, bl, role='write_guard', **metadata)
        self.decisions += 1
        current = level / guard['resistance_ohm']
        admitted = current <= (.002 if sign else .04)
        record = Pulse(self.phase, wl, bl, cycle, sign, level, admitted)
        self.emit('pulse_decision', record=asdict(record), guard=guard,
                  predicted_current_a=current, metadata=metadata)
        self.last_decision = self.clock()
        response: Reading | None = None
        if admitted:
            if self.pulses >= 1280 or self.phase_pulses[self.phase] >= (512 if self.phase == 'A' else 768):
                raise ValueError('Finite pulse cap')
            self.pulses += 1
            self.phase_pulses[self.phase] += 1
            response = self.call(dict(mode_flag=7, wl=wl, bl=bl, vol=level, sign=sign,
                                      pulse_width=.000125 if self.source == 'simulator' else 0.),
                                 role='write', **metadata)
        self.updates.append(record)
        self.emit('pulse_complete', record=asdict(record), guard=guard, response=response, metadata=metadata)

    def bracket(self, wl: int, bl: int, **metadata: object) -> Bracket:
        before = self.call(dict(mode_flag=10, wl=wl, vol=[0.] * 32), bracket='before', **metadata)
        guard = self.read(wl, bl, bracket='guard', **metadata)
        decoded = self.cv.convert_dac_to_volt(12, 5, self.cv.convert_volt_to_dac(12, 5, .05))
        current = max(.05, decoded) / guard['resistance_ohm']
        if current > .005 or current * 330 >= 3.3:
            raise ValueError('Fresh current/headroom guard; missing not zero-filled')
        signal = self.call(dict(mode_flag=10, wl=wl, vol=[.05 if b == bl else 0. for b in range(32)]),
                           bracket='signal', **metadata)
        after = self.call(dict(mode_flag=10, wl=wl, vol=[0.] * 32), bracket='after', **metadata)
        vs = [self.cv.convert_adc_to_volt(1, 14, 5, r['adc']) for r in (before, signal, after)]
        if any(not math.isfinite(v) or v > 4. for v in vs):
            raise ValueError('Measured ADC headroom')
        row = Bracket(before=before, guard=guard, signal=signal, after=after,
                      centered_adc=signal['adc'] - (before['adc'] + after['adc']) / 2,
                      centered_v=vs[1] - (vs[0] + vs[2]) / 2)
        self.emit('bracket_complete', wl=wl, bl=bl, bracket=row, metadata=metadata)
        return row

    def checkpoint(self, wl: int, bl: int, cycle: int, state: str, role: str) -> None:
        rows = [self.bracket(wl, bl, cycle=cycle, state=state, role=role, repeat=r) for r in range(10)]
        values = [r['centered_adc'] for r in rows]
        zeros = [float(r['after']['adc'] - r['before']['adc']) for r in rows]
        point = Point(wl, bl, cycle, state, role, statistics.mean(values), statistics.stdev(values),
                      statistics.mean(zeros), max(2., 3 * statistics.stdev(zeros) / math.sqrt(10)),
                      statistics.mean(1 / r['guard']['resistance_ohm'] for r in rows), self.clock())
        self.points.append(point)
        self.emit('checkpoint_complete', checkpoint=asdict(point), brackets=rows)
        self.writer.checkpoint(phase=self.phase, impacts=self.calls, wl=wl, bl=bl)

    def wait(self, origin: float, delay: float) -> None:
        self.budget()
        remaining = max(0., origin + delay - self.clock())
        if self.waited + remaining > 600. + 1e-7 or self.clock() - self.started + remaining >= 1200:
            raise TimeoutError('Finite explicit wait/wall budget')
        self.emit('wait', requested_s=remaining, origin=origin, delay=delay)
        if remaining:
            self.sleep(remaining)
        self.waited += remaining

    def phase_a(self) -> None:
        for wl, bl in CELLS:
            self.checkpoint(wl, bl, -1, 'initial', 'baseline')
        for cycle in range(2):
            for state in ('R', 'M', 'H'):
                for wl, bl in CELLS:
                    for sign, level in recipe(state, incremental=True):
                        self.pulse(wl, bl, sign, level, cycle, state=state)
                    self.checkpoint(wl, bl, cycle, state, 'immediate')
                origin = self.last_decision
                for delay in (0, 10, 60):
                    self.wait(origin, delay)
                    for wl, bl in CELLS:
                        self.checkpoint(wl, bl, cycle, state, 'delay_' + str(delay))
                self.progress(f'A cycle{cycle+1}/2 state{state}; calls{self.calls} pulses{self.pulses}')
        for wl, bl in CELLS:
            self.checkpoint(wl, bl, -1, 'H', 'final')
        self.cells = assess_cells(self.points, self.updates)
        self.emit('phase_a_complete', cells=self.cells, rerank=False)

    def paired_checkpoint(self, bl: int, configuration: str, role: str) -> None:
        # Как в исходном B: interleaved WL0→WL7 для каждого repeat, не два
        # независимых пакета по10, чтобы не скрывать временной дрейф пары.
        rows = [[self.bracket(wl, bl, configuration=configuration, role=role, repeat=r)
                 for wl in (0, 7)] for r in range(10)]
        differences = [r[0]['centered_adc'] - r[1]['centered_adc'] for r in rows]
        point = PairPoint(bl, configuration, role, statistics.mean(differences),
                          statistics.stdev(differences), self.clock())
        self.pairs.append(point)
        self.emit('paired_checkpoint_complete', checkpoint=asdict(point), brackets=rows)
        self.writer.checkpoint(phase=self.phase, impacts=self.calls, bl=bl)

    def phase_b(self) -> None:
        self.phase = 'B'
        for configuration in ('plus', 'minus', 'zero', 'checkerboard'):
            for index, bl in enumerate(BLS):
                for wl, state in zip((0, 7), pair_states(configuration, index)):
                    for sign, level in recipe(state):
                        self.pulse(wl, bl, sign, level, -1, state=state, configuration=configuration)
                self.paired_checkpoint(bl, configuration, 'immediate')
            self.wait(self.last_decision, 60)
            for bl in BLS:
                self.paired_checkpoint(bl, configuration, 'delayed')
            self.progress(f'B {configuration}; calls{self.calls} pulses{self.pulses}')
        self.pair_report = assess_pairs(self.pairs, self.cells)
        self.emit('phase_b_complete', qualification=self.pair_report)

    def run(self) -> dict[str, object]:
        self.phase_a()
        # Не rerank и не расширение: все16 прежних legs должны снова иметь
        # разрешимый диапазон. Stablezero не является полезным synapticweight.
        if not all(row['hard_eligible'] is True for row in self.cells):
            self.status = 'no_stable_rectangle'
            return self.result()
        self.phase_b()
        self.status = 'completed' if self.pair_report['mandatory_pass'] is True else 'pair_qualification_failed'
        return self.result()

    def result(self) -> dict[str, object]:
        quality = self.status == 'completed' and all(row['eligible'] is True for row in self.cells) and self.pair_report.get('qualified') is True
        return dict(status=self.status, cells=self.cells, pair_qualification=self.pair_report,
                    checkpoints=[asdict(p) for p in self.points], updates=[asdict(u) for u in self.updates],
                    pair_observations=[asdict(p) for p in self.pairs], impacts=self.calls, pulses=self.pulses,
                    phase_impacts=self.phase_calls, phase_pulses=self.phase_pulses,
                    explicit_wait_requested_s=self.waited, qualified=quality,
                    diagnostic_best_available=self.status == 'completed' and not quality,
                    full_array_validated=False, calibration_matrix_reused=False, rerank=False)
