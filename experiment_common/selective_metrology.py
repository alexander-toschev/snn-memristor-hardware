"""Выборочная метрология через публичный MemriBoard; никаких SET/RESET.

Проверяется нынешняя повторяемость, а не сохранность программируемых уровней
старой карты: после неё проводимости уже менялись другими опытами.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import contextlib
import json
import math
from pathlib import Path
import statistics
import time
from typing import TYPE_CHECKING, Callable, Literal, Protocol, TypedDict

from experiment_common.durable import Writer
from experiment_common.native_output import DriverOutput, safe

if TYPE_CHECKING:
    from manager.board import Connector

Source = Literal['hardware', 'simulator']


class Converters(Protocol):
    def convert_volt_to_dac(self, bits: int, reference: float, voltage: float) -> int: ...
    def convert_dac_to_volt(self, bits: int, reference: float, code: int) -> float: ...
    def convert_adc_to_volt(self, gain: int, bits: int, reference: float, code: int) -> float: ...


def object_dict(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(k, str) for k in value):
        raise ValueError('Expected JSON object')
    return {str(k): v for k, v in value.items()}


def number(value: object) -> float:
    if type(value) not in (int, float) or not isinstance(value, (int, float)):
        raise ValueError('Expected number')
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('Nonfinite number')
    return result


@dataclass(frozen=True)
class Cell:
    wl: int
    bl: int
    span_adc: float
    sample_reason: str
    historical_hard_eligible: bool
    historical_quality_qualified: bool


def load_sample(path: Path) -> tuple[dict[str, object], list[Cell]]:
    decoded: object = json.loads(path.read_text(encoding='utf-8'))
    data = object_dict(decoded)
    if data.get('schema') != 'hardware-map-fixed-sample-v1' or data.get('parent_source') != 'hardware':
        raise ValueError('Expected fixed hardware sample')
    rows = data.get('sample')
    if not isinstance(rows, list):
        raise ValueError('Missing sample')
    cells: list[Cell] = []
    for item in rows:
        row = object_dict(item)
        wl, bl = row['wl'], row['bl']
        reason, hard, quality = row['sample_reason'], row['historical_hard_eligible'], row['historical_quality_qualified']
        if type(wl) is not int or type(bl) is not int or not isinstance(wl, int) or not isinstance(bl, int):
            raise ValueError('Integer addresses required')
        if not 0 <= wl < 8 or not 0 <= bl < 32 or not isinstance(reason, str):
            raise ValueError('Invalid sample address/reason')
        if type(hard) is not bool or type(quality) is not bool or not isinstance(hard, bool) or not isinstance(quality, bool):
            raise ValueError('Invalid historical grades')
        cells.append(Cell(wl, bl, number(row['span_adc']), reason, hard, quality))
    if len(cells) != 40 or len({(c.wl, c.bl) for c in cells}) != 40 or {c.wl for c in cells} != set(range(8)):
        raise ValueError('Expected40distinctcells covering8WL')
    return data, cells


class Reading(TypedDict):
    task_id: int
    resistance_ohm: float
    adc: int


class Bracket(TypedDict):
    before: Reading
    guard: Reading
    signal: Reading
    after: Reading
    centered_adc: float
    centered_v: float


class Summary(TypedDict):
    mean_adc: float
    sd_adc: float
    mean_g_s: float
    zero_difference_adc: float
    zero_shift_limit_adc: float
    noise_limit_adc: float
    drift_limit_adc: float
    drift_from_initial_adc: float
    noise_pass: bool
    zero_shift_pass: bool
    drift_pass: bool


def summarize(rows: list[Bracket], cell: Cell, initial: float | None) -> Summary:
    if len(rows) != 10:
        raise ValueError('Exactly10freshbrackets required')
    values = [row['centered_adc'] for row in rows]
    zero = [float(row['after']['adc'] - row['before']['adc']) for row in rows]
    mean, sd = statistics.mean(values), statistics.stdev(values)
    # Точно прежние формулы metrics.checkpoint/cell_summary. span — историческая
    # шкала допуска, не сравнение текущей G с устаревшим значением G.
    noise_limit = max(2., .05 * cell.span_adc)
    drift_limit = max(3., .05 * cell.span_adc)
    zero_limit = max(2., 3 * statistics.stdev(zero) / math.sqrt(10))
    drift = 0. if initial is None else abs(mean - initial)
    return Summary(mean_adc=mean, sd_adc=sd,
                   mean_g_s=statistics.mean(1 / row['guard']['resistance_ohm'] for row in rows),
                   zero_difference_adc=statistics.mean(zero), zero_shift_limit_adc=zero_limit,
                   noise_limit_adc=noise_limit, drift_limit_adc=drift_limit,
                   drift_from_initial_adc=drift, noise_pass=sd <= noise_limit,
                   zero_shift_pass=abs(statistics.mean(zero)) <= zero_limit,
                   drift_pass=drift <= drift_limit)


class Collector:
    """Независимый read-only collector для всех32BL; запись API здесь отсутствует."""

    def __init__(self, connector: Connector, converters: Converters, writer: Writer,
                 source: Source, *, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.connector: Connector = connector
        self.cv = converters
        self.writer = writer
        self.source = source
        self.clock, self.sleep = clock, sleep
        self.calls = 0
        self.started = clock()
        self.checkpoints: list[dict[str, object]] = []

    def emit(self, event: str, **fields: object) -> None:
        self.writer.write(dict(event=event, source=self.source, monotonic_s=self.clock(), **fields))

    def call(self, command: dict[str, int | float | list[float]], **metadata: object) -> Reading:
        if self.calls >= 6400 or self.clock() - self.started >= 600:
            raise TimeoutError('Read-only finite budget exhausted')
        # Запрет записи проверяется ДО impact, включая ошибочный вызывающий код.
        mode = command['mode_flag']
        if mode == 7:
            if command.get('vol') != 0. or command.get('pulse_width') != 0. or command.get('sign') != 0:
                raise ValueError('Writes forbidden in selective metrology')
        elif mode == 10:
            vol = command.get('vol')
            if not isinstance(vol, list) or len(vol) != 32 or any(v not in (0., .05) for v in vol) or sum(v > 0 for v in vol) > 1:
                raise ValueError('Only zero/singleton .05V mode10 permitted')
        else:
            raise ValueError('Only public mode7read/mode10 permitted')
        wl = command.get('wl')
        if type(wl) is not int or not isinstance(wl, int) or not 0 <= wl < 8:
            raise ValueError('Invalid WL')
        if mode == 7:
            bl = command.get('bl')
            if type(bl) is not int or not isinstance(bl, int) or not 0 <= bl < 32:
                raise ValueError('Invalid BL')
        self.calls += 1
        ticket = {**command, 'id': self.calls if mode == 7 else 0}
        self.emit('request', task_id=self.calls, command_si=ticket, metadata=metadata)
        voltage = ticket['vol']
        copied = {**ticket, 'vol': list(voltage) if isinstance(voltage, list) else voltage}
        with contextlib.redirect_stdout(DriverOutput(self, 'stdout')), contextlib.redirect_stderr(DriverOutput(self, 'stderr')):
            response = self.connector.impact(copied)
        self.emit('result', task_id=self.calls, command_si=ticket, native_result=safe(response))
        if not isinstance(response, (list, tuple)) or len(response) != 3:
            raise ValueError('Invalid native response')
        resistance, echo, adc = response
        if type(echo) is not int or echo != ticket['id'] or type(adc) is not int or not 0 <= adc <= 16382:
            raise ValueError('Invalid native ID/ADC')
        r = number(resistance)
        if mode == 7 and (r <= 1e-8 or adc < 20):
            raise ValueError('Invalid native R/readADC')
        return Reading(task_id=self.calls, resistance_ohm=r, adc=adc)

    def vector(self, wl: int, bl: int | None, **metadata: object) -> Reading:
        return self.call(dict(mode_flag=10, wl=wl, vol=[.05 if b == bl else 0. for b in range(32)]), **metadata)

    def checkpoint(self, cell: Cell, role: str, initial: float | None) -> Summary:
        rows: list[Bracket] = []
        for repeat in range(10):
            meta: dict[str, object] = dict(cell_wl=cell.wl, cell_bl=cell.bl, role=role, repeat=repeat)
            before = self.vector(cell.wl, None, bracket='before', **meta)
            guard = self.call(dict(mode_flag=7, wl=cell.wl, bl=cell.bl, vol=0., sign=0, pulse_width=0.), bracket='guard', **meta)
            code = self.cv.convert_volt_to_dac(12, 5, .05)
            voltage = max(.05, self.cv.convert_dac_to_volt(12, 5, code))
            current = voltage / guard['resistance_ohm']
            if current > .005 or 330 * current >= 3.3:
                raise ValueError('Fresh R current/headroom guard failed; no zero fill')
            signal = self.vector(cell.wl, cell.bl, bracket='signal', **meta)
            after = self.vector(cell.wl, None, bracket='after', **meta)
            vs = [self.cv.convert_adc_to_volt(1, 14, 5, r['adc']) for r in (before, signal, after)]
            if any(not math.isfinite(v) or v > 4. for v in vs):
                raise ValueError('Native MVM measured headroom failed')
            row = Bracket(before=before, guard=guard, signal=signal, after=after,
                          centered_adc=signal['adc'] - (before['adc'] + after['adc']) / 2,
                          centered_v=vs[1] - (vs[0] + vs[2]) / 2)
            rows.append(row)
            self.emit('bracket_complete', cell=asdict(cell), role=role, repeat=repeat, bracket=row)
        summary = summarize(rows, cell, initial)
        record: dict[str, object] = dict(cell=asdict(cell), role=role, monotonic_s=self.clock(), summary=summary, brackets=rows)
        self.checkpoints.append(record)
        self.emit('checkpoint_complete', checkpoint=record)
        self.writer.checkpoint(impacts=self.calls, cell=[cell.wl, cell.bl], role=role)
        return summary

    def run(self, cells: list[Cell], progress: Callable[[str], None]) -> list[Summary]:
        initial: dict[tuple[int, int], float] = {}
        summaries: list[Summary] = []
        origin = self.clock()
        for role, delay in [('initial', 0), ('delay_0', 0), ('delay_10', 10), ('delay_60', 60)]:
            wait = max(0., origin + delay - self.clock()) if role != 'initial' else 0.
            if self.clock() - self.started + wait >= 600:
                raise TimeoutError('Read-only wall budget')
            self.emit('wait', role=role, requested_s=wait)
            if wait:
                self.sleep(wait)
            for index, cell in enumerate(cells):
                summary = self.checkpoint(cell, role, initial.get((cell.wl, cell.bl)))
                summaries.append(summary)
                if role == 'initial':
                    initial[cell.wl, cell.bl] = summary['mean_adc']
                if (index + 1) % 8 == 0 or index + 1 == len(cells):
                    progress(f'{role}: {index + 1}/{len(cells)} cells; {self.calls}/6400 calls')
            if role == 'initial':
                origin = self.clock()
        return summaries
