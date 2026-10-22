"""Bounded amplitude feedback through public MemriBoard only.

The CPU chooses pulses. Conductance and readout are measured by the Connector;
no simulator state access and no synthetic substitution for an unsuccessful write.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import statistics
import time
from typing import TYPE_CHECKING

from experiment_common.durable import Writer
from experiment_common.native_memriboard import Collector as NativeCollector, Converters
from experiment_common.selective_metrology import Source

if TYPE_CHECKING:
    from manager.board import Connector

TOLERANCE = 2.5603554203687646e-6
LEVELS = (.3, .6, .9, 1.2, 1.5, 2., 2.5, 3.)


@dataclass
class Feedback:
    """Independent SET/RESET amplitudes; duration is not an adaptive parameter."""
    levels: list[float] = field(default_factory=lambda: [1., 1.])
    steps: list[float] = field(default_factory=lambda: [.2, .2])
    last_sign: int | None = None
    reversals: int = 0
    ceiling_stalls: int = 0

    def command(self, measured: float, target: float) -> tuple[int, float]:
        sign = int(measured < target)
        return sign, round(self.levels[sign], 3)

    def observe(self, before: float, after: float, goal: float, sign: int, tolerance: float) -> None:
        direction = 1 if sign else -1
        delta = (after-before)*direction
        crossed = (before-goal)*(after-goal) < 0
        if delta > tolerance*.2:
            self.ceiling_stalls = 0
        if self.last_sign is not None and self.last_sign != sign:
            self.reversals += 1
        self.last_sign = sign
        if crossed:
            self.steps[sign] = max(.025, self.steps[sign]/2)
            self.levels[sign] -= self.steps[sign]
        elif delta <= tolerance*.2:
            self.ceiling_stalls = self.ceiling_stalls+1 if self.levels[sign] >= 3 else 0
            self.levels[sign] += self.steps[sign]
        elif delta > abs(goal-after)*.6:
            self.levels[sign] -= self.steps[sign]/2
        elif delta < abs(goal-after)*.1:
            self.levels[sign] += self.steps[sign]/4
        self.levels[sign] = min(3., max(.3, self.levels[sign]))


class Collector(NativeCollector):
    def __init__(self, connector: Connector, cv: Converters, writer: Writer, source: Source,
                 cells: tuple[tuple[int, int], ...], adaptive: bool) -> None:
        super().__init__(connector, cv, writer, 180000, lambda msg: print(msg, flush=True))
        self.source, self.allowed, self.adaptive = source, frozenset(cells), adaptive

    def emit(self, event: str, **fields: object) -> None:
        self.writer.write(dict(event=event, source=self.source, monotonic_s=time.monotonic(), **fields))

    def call(self, command: dict[str, int | float | list[float]], **metadata: object) -> tuple[int, float, int]:
        if time.monotonic()-self.started > 1800:
            raise TimeoutError('Thirty-minute acquisition budget')
        mode, wl, bl = command['mode_flag'], command['wl'], command.get('bl')
        if type(wl) is not int or mode not in (7, 10):
            raise ValueError('Only public mode7/mode10')
        if mode == 7:
            voltage = command['vol']
            if type(bl) is not int or (wl, bl) not in self.allowed or not isinstance(voltage, (int, float)):
                raise ValueError('Outside fixed37 cell scope')
            if not 0 <= float(voltage) <= 3. or command['sign'] not in (0, 1):
                raise ValueError('Pulse voltage/polarity outside budget')
        else:
            volts = command['vol']
            if not isinstance(volts, list) or len(volts) != 32 or any(v not in (0., .025, .05, .075) for v in volts):
                raise ValueError('Invalid bounded MVM vector')
            active = [b for b, v in enumerate(volts) if v]
            if len(active) > 1 or any((wl, b) not in self.allowed for b in active):
                raise ValueError('Only selected singleton MVM')
        result = super().call(command, **metadata)
        if mode == 10 and self.cv.convert_adc_to_volt(1, 14, 5, result[2]) > 4.:
            raise ValueError('MVM measured output range')
        return result

    def read_r(self, wl: int, bl: int, **metadata: object) -> float:
        # All32 BL are supported by MemriBoard; this experiment includes BL31.
        if (wl, bl) not in self.allowed:
            raise ValueError('Outside selected37 addresses')
        return self.call(dict(mode_flag=7, wl=wl, bl=bl, vol=0., sign=0, pulse_width=0.), **metadata)[1]

    def pulse(self, wl: int, bl: int, sign: int, voltage: float, **metadata: object) -> tuple[float, bool]:
        if self.decisions >= 16000:
            raise ValueError('Finite16000 pulse-decision budget')
        r = self.read_r(wl, bl, role='pulse_guard', **metadata)
        self.decisions += 1
        admitted = voltage/r <= (.002 if sign else .04)
        self.emit('pulse_decision', wl=wl, bl=bl, sign=sign, voltage_v=voltage,
                  admitted=admitted, predicted_current_a=voltage/r, metadata=metadata,
                  submitted_width_s=.00001, physical_width_s=None)
        if admitted:
            self.pulses += 1
            self.call(dict(mode_flag=7, wl=wl, bl=bl, vol=voltage, sign=sign, pulse_width=.00001),
                      role='write', **metadata)
        return self.read_g(wl, bl, role='post_program', **metadata), admitted

    def program(self, wl: int, bl: int, target: float, cycle: int, level: int) -> dict[str, object]:
        if not math.isfinite(target) or target <= 0:
            raise ValueError('Invalid target')
        feedback = Feedback()
        g = self.read_g(wl, bl, role='initial', cycle=cycle, level=level)
        initial, best = g, abs(g-target)
        status, attempts, blocked = 'attempt_limit', 0, 0
        for attempt in range(64):
            # Inner half-band leaves margin; acceptance uses the unchanged full
            # tolerance and two fresh independent read triples after settling.
            if abs(g-target) <= .5*TOLERANCE:
                time.sleep(.1)
                confirm = [self.read_g(wl, bl, role='confirm', cycle=cycle, level=level) for _ in range(2)]
                g = confirm[-1]
                if all(abs(v-target) <= TOLERANCE for v in confirm):
                    status = 'confirmed'
                    break
            sign, voltage = feedback.command(g, target)
            if not self.adaptive:
                voltage = 1.7 if sign else 1.4
            previous = g
            g, admitted = self.pulse(wl, bl, sign, voltage, cycle=cycle, level=level,
                                     attempt=attempt, target_g_s=target)
            attempts += 1
            best = min(best, abs(g-target))
            blocked = 0 if admitted else blocked+1
            feedback.observe(previous, g, target, sign, TOLERANCE)
            if blocked >= 2:
                status = 'current_guard'
                break
            if self.adaptive and feedback.ceiling_stalls >= 4:
                status = 'no_response_at_maximum'
                break
        # Final acceptance is never based on an earlier/best state.
        time.sleep(.1)
        final = [self.read_g(wl, bl, role='final_confirm', cycle=cycle, level=level) for _ in range(2)]
        passed = all(abs(v-target) <= TOLERANCE for v in final)
        self.emit('target_complete', wl=wl, bl=bl, cycle=cycle, level=level,
                  target_g_s=target, final_g_s=final, passed=passed, stop=status)
        return dict(wl=wl, bl=bl, cycle=cycle, level=level, target_g_s=target, initial_g_s=initial,
                    final_g_s=final, passed=passed, stop=status, attempts=attempts,
                    best_observed_error_g_s=best, reversals=feedback.reversals)

    def readout(self, wl: int, bl: int, bias: float) -> dict[str, object]:
        values: list[float] = []
        for repeat in range(4):
            r = self.read_r(wl, bl, role='mvm_guard', repeat=repeat)
            signal = round(bias+.05, 3)
            decoded = self.cv.convert_dac_to_volt(12, 5, self.cv.convert_volt_to_dac(12, 5, signal))
            current = max(signal, decoded)/r
            if current > .005 or 330*current >= 3.3:
                raise ValueError('Fresh MVM current/headroom guard')
            samples = [self.call(dict(mode_flag=10, wl=wl, vol=[v if b == bl else 0. for b in range(32)]),
                                 role=role, repeat=repeat)[2]
                       for role, v in (('before', bias), ('signal', signal), ('after', bias))]
            values.append(samples[1]-(samples[0]+samples[2])/2)
        return dict(bias_v=bias, delta_v=.05, mean_adc=statistics.mean(values), sd_adc=statistics.stdev(values),
                    resolved=abs(statistics.mean(values)) >= max(5., 5*statistics.stdev(values)),
                    repeatable=statistics.stdev(values) <= 5.)
