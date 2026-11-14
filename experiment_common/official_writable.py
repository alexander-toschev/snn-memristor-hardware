"""Official JSON waveforms with explicit DOCX resistance stopping criteria.

No GUI legacy terminator conversion: it reverses the intended meaning when an
ADC threshold is converted to resistance but the inequality is kept unchanged.
All physical commands still go through the public MemriBoard Collector.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
import html
import json
import math
from numbers import Real
from pathlib import Path
from typing import cast

from experiment_common.adaptive_programming import Collector
from experiment_common.selective_metrology import number, object_dict

Generator = Callable[[str, dict[str, object], dict[str, object], str], Iterator[list[object]]]


def waveform(collector: Collector, legacy: dict[str, object], wl: int, bl: int) -> list[dict[str, int | float | list[float]]]:
    from manager.modes.std import get_std  # type: ignore[import-untyped]  # Public untyped vendor boundary, typed below.
    generate = cast(Generator, get_std)
    p = object_dict(legacy['params'])
    params: dict[str, object] = dict(count=int(number(p['count'])), reverse=int(number(p['reverse'])), id=0, wl=wl, bl=bl)
    for direction in ('dir', 'rev'):
        for output, source in (('start', 'strt'), ('stop', 'stop'), ('step', 'step')):
            params[f'{output}_{direction}'] = collector.cv.convert_dac_to_volt(12, 5, int(number(p[f'v_{direction}_{source}_inc'])))
        params[f'pulse_width_{direction}'] = number(p[f't_{direction}_msec_inc'])*1e-3+number(p[f't_{direction}_usec_inc'])*1e-6
        params[f'amount_{direction}'] = int(number(p[f'{direction}_inc_countr']))
        params[f'double_{direction}'] = bool(p[f'{direction}_dec_countr'])
    commands: list[dict[str, int | float | list[float]]] = []
    for task in generate('volt_sweep', params, {'type': 'pass', 'value': 0}, 'mode_7'):
        raw = object_dict(task[0])
        command: dict[str, int | float | list[float]] = {k: int(number(raw[k])) for k in ('mode_flag', 'sign', 'id', 'wl', 'bl')}
        native_voltage = raw['vol']
        if not isinstance(native_voltage, Real) or isinstance(native_voltage, bool):
            raise ValueError('Native generator returned nonnumeric voltage')
        command.update(vol=float(native_voltage), pulse_width=number(raw['pulse_width']))
        if (command['mode_flag'] != 7 or command['sign'] not in (0, 1)
                or not 0 <= number(command['vol']) <= 2.200001
                or not 0 <= number(command['pulse_width']) <= .001001):
            raise ValueError('Official waveform bounds violated')
        commands.append(command)
    return commands


def load_tickets(path: Path) -> dict[str, object]:
    import hashlib
    raw = path.read_bytes()
    # Exact public source, pinned in manifest; no arbitrary executable ticket data.
    if hashlib.sha256(raw).hexdigest() != OFFICIAL_SHA256:
        raise ValueError('Official JSON changed; review waveform and provenance first')
    return object_dict(json.loads(raw))


OFFICIAL_SHA256 = 'f3eaab2cff2e047bd4b13d0cbdcc6b4c0bb437fc14167c917549595296d16222'


def characterize(collector: Collector, wl: int, bl: int, tickets: dict[str, object]) -> dict[str, object]:
    readings: list[float] = []
    stages: list[dict[str, object]] = []
    for key, name, expected in (('0', 'SET', 140), ('1', 'RESET', 180), ('2', 'IV', 300)):
        commands = waveform(collector, object_dict(tickets[key]), wl, bl)
        if len(commands) != expected:
            raise ValueError('Official native waveform length changed')
        blocked = 0
        submitted = 0
        stopped = False
        last_r = collector.read_r(wl, bl, role='official_stage_initial', stage=name)
        for index, command in enumerate(commands):
            if (name == 'SET' and last_r <= 296.) or (name == 'RESET' and last_r >= 2998.):
                stopped = True
                break
            if collector.decisions >= collector.budget.maximum_decisions:
                raise ValueError('Official pulse-decision budget exhausted')
            guard_r = collector.read_r(wl, bl, role='official_current_guard', stage=name, step=index)
            voltage = number(command['vol'])
            sign = int(number(command['sign']))
            actual_v = collector.cv.convert_dac_to_volt(12, 5, collector.cv.convert_volt_to_dac(12, 5, voltage))
            current = max(voltage, actual_v)/guard_r
            admitted = current <= (.002 if sign else .04)
            collector.decisions += 1
            collector.emit('official_pulse_decision', wl=wl, bl=bl, stage=name, step=index,
                           admitted=admitted, predicted_current_a=current, command=command,
                           submitted_width_s=command['pulse_width'], physical_width_s=None)
            if not admitted:
                blocked += 1
                last_r = guard_r
                continue
            submitted += 1
            collector.pulses += int(voltage > 0)
            _, last_r, adc = collector.call(command, role='official_waveform', stage=name, step=index)
            readings.append(last_r)
            collector.emit('official_observation', wl=wl, bl=bl, stage=name, step=index,
                           resistance_ohm=last_r, adc=adc)
        stopped = stopped or (name == 'SET' and last_r <= 296.) or (name == 'RESET' and last_r >= 2998.)
        stage = dict(name=name, maximum_tasks=expected, submitted_tasks=submitted,
                     skipped_current_guard=blocked, stopped_by_docx_resistance=stopped,
                     last_r_ohm=last_r)
        stages.append(stage)
        collector.emit('official_stage_complete', wl=wl, bl=bl, **stage)
    if not readings or any(not math.isfinite(r) or r <= 0 for r in readings):
        raise ValueError('Invalid official resistance observations')
    low, high = min(readings), max(readings)
    skipped = sum(int(number(s['skipped_current_guard'])) for s in stages)
    return dict(wl=wl, bl=bl, complete=True, stages=stages, min_r_ohm=low, max_r_ohm=high,
                ratio=high/low, observed_docx_switching=low > 100 and high/low > 2,
                observed_gui_switching=low > 100 and high/low >= 2,
                skipped_current_guard=skipped, guard_modified_protocol=skipped > 0,
                qualified_for_mnist=False, signed_group_mvm='not_tested')


def render_map(rows: list[dict[str, object]], summary: dict[str, object]) -> str:
    by_address = {(int(number(r['wl'])), int(number(r['bl']))): r for r in rows}
    table = '<table><tr><th>WL / BL</th>'+''.join(f'<th>{b}</th>' for b in range(32))+'</tr>'
    for w in range(8):
        table += f'<tr><th>{w}</th>'
        for b in range(32):
            row = by_address.get((w, b), {'status': 'unmeasured'})
            colour = '#ddd' if (w, b) not in by_address else ('#acf0ad' if row.get('observed_docx_switching') else '#ffd590')
            if row.get('error'):
                colour = '#ff9999'
            details = html.escape(json.dumps(row, ensure_ascii=False, indent=2), quote=True)
            table += f'<td style="background:{colour}" title="{details}">{w},{b}</td>'
        table += '</tr>'
    return '<!doctype html><meta charset="utf-8"><title>Official switching survey</title><h1>Observed resistive switching</h1><p>Green: observed DOCX R criterion, orange: not met, red: measurement error, gray: unmeasured. Hover for details. Guard skips alter the protocol. This is NOT MNIST or grouped MVM qualification; physical pulse widths are unknown.</p><pre>'+html.escape(json.dumps(summary, indent=2))+'</pre>'+table+'</table>'
