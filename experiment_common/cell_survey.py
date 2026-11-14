"""Cell-level characterization, separate from signed/group MVM qualification.

CPU selects target conductances and reports measured errors. All measurements
and programming are delegated to the public MemriBoard Collector. A weak MVM
or missed conductance target is a recorded limitation, never an ideal fallback.
"""
from __future__ import annotations

import html
import json
from pathlib import Path
import time

from experiment_common.adaptive_programming import Collector, LEVELS, TOLERANCE
from experiment_common.selective_metrology import number, object_dict

Address = tuple[int, int]


def addresses(priority_path: Path) -> tuple[Address, ...]:
    """The historic 44 coordinates first, then every other WL/BL exactly once."""
    raw: object = json.loads(priority_path.read_text(encoding='utf-8'))
    if not isinstance(raw, list):
        raise ValueError('Priority list required')
    priority: list[Address] = []
    for item in raw:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError('Expected WL/BL coordinate pair')
        wl, bl = item
        if type(wl) is not int or type(bl) is not int or not 0 <= wl < 8 or not 0 <= bl < 32:
            raise ValueError('Invalid priority coordinate')
        priority.append((wl, bl))
    if len(priority) != 44 or len(set(priority)) != 44:
        raise ValueError('Exactly 44 distinct priority cells required')
    return tuple(priority + [(w, b) for w in range(8) for b in range(32) if (w, b) not in priority])


def characterize(collector: Collector, wl: int, bl: int) -> dict[str, object]:
    """Two bounded sweeps of three measured-range goals, unchanged feedback."""
    started = time.monotonic()
    initial = collector.read_g(wl, bl, role='survey_initial')
    discovery: list[dict[str, object]] = []
    endpoints: list[float] = []
    for sign in (0, 1):
        for voltage in LEVELS:
            g, admitted = collector.pulse(wl, bl, sign, voltage, phase='discovery')
            discovery.append(dict(sign=sign, voltage_v=voltage, g_s=g, admitted=admitted))
        time.sleep(.1)
        endpoints.append(collector.read_g(wl, bl, role='discovery_endpoint', sign=sign))
    low, high = endpoints
    span = high - low
    targets: list[dict[str, object]] = []
    if span > 8*TOLERANCE:
        goals = [low + fraction*span for fraction in (.2, .5, .8)]
        for cycle in range(2):
            for level in ((0, 1, 2) if cycle == 0 else (2, 1, 0)):
                result = collector.program(wl, bl, goals[level], cycle, level)
                # A singleton readout is not proof that a multi-cell group works.
                result['mvm'] = [collector.readout(wl, bl, bias) for bias in (0., .025)]
                targets.append(result)
    # Even a narrow/wrong-direction discovery remains visible in the cell map.
    # Measure its actual state, rather than fabricating a program target or weight.
    final_g = collector.read_g(wl, bl, role='cell_final_state')
    readout = [collector.readout(wl, bl, bias) for bias in (0., .025)]
    return dict(wl=wl, bl=bl, complete=True, initial_g_s=initial,
        discovery=discovery, endpoints_g_s=endpoints, span_g_s=span, targets=targets,
        tracking_passed=len(targets) == 6 and all(r['passed'] is True for r in targets),
        final_target_g_s=number(targets[-1]['target_g_s']) if targets else None,
        cell_final_g_s=final_g, final_mvm=readout,
        range_failure=None if targets else 'insufficient_or_wrong_direction_endpoint_span',
        characterized_monotonic_s=time.monotonic(), elapsed_seconds=time.monotonic()-started,
        qualified_for_mnist=False, signed_group_mvm='not_tested')


def final_readback(collector: Collector, row: dict[str, object]) -> None:
    """Retention/disturb after intervening cells; no second programming pass."""
    wl, bl = int(number(row['wl'])), int(number(row['bl']))
    measured = collector.read_g(wl, bl, role='whole_array_readback')
    row['whole_array_final_g_s'] = measured
    row['readback_delay_seconds'] = time.monotonic() - number(row['characterized_monotonic_s'])
    row['late_change_g_s'] = measured - number(row['cell_final_g_s'])
    target = row['final_target_g_s']
    row['final_target_error_g_s'] = measured-number(target) if target is not None else None
    row['qualified_programming'] = (row['tracking_passed'] is True and target is not None
                                   and abs(measured-number(target)) <= TOLERANCE)


def problems(row: dict[str, object]) -> list[str]:
    """Observed limitations only; no exclusion from MNIST is implied here."""
    result: list[str] = []
    if row.get('error'):
        result.append(str(row['error']))
    if row.get('range_failure'):
        result.append('Endpoint span too small or opposite to expected SET/RESET direction')
    targets = row.get('targets', [])
    if isinstance(targets, list):
        failures = [object_dict(r) for r in targets if object_dict(r).get('passed') is not True]
        if failures:
            result.append(f'{len(failures)}/{len(targets)} conductance goals missed')
            result.extend(sorted({str(r.get('stop')) for r in failures}))
    final = row.get('final_mvm', [])
    if isinstance(final, list):
        for item in final:
            r = object_dict(item)
            if r.get('resolved') is not True:
                result.append(f'Singleton MVM unresolved at bias {r["bias_v"]} V')
            if r.get('repeatable') is not True:
                result.append(f'Singleton MVM SD > 5 ADC at bias {r["bias_v"]} V')
    late_error = row.get('final_target_error_g_s')
    if late_error is not None and abs(number(late_error)) > TOLERANCE:
        result.append('Late readback outside final target tolerance')
    return result


def render_map(rows: list[dict[str, object]], summary: dict[str, object]) -> str:
    by_cell = {(int(number(r['wl'])), int(number(r['bl']))): r for r in rows}
    table = '<tr><th>WL / BL</th>' + ''.join(f'<th>{b}</th>' for b in range(32)) + '</tr>'
    details = ''
    for wl in range(8):
        table += f'<tr><th>{wl}</th>'
        for bl in range(32):
            row = by_cell.get((wl, bl))
            if row is None:
                color, label, message = '#e5e7eb', '—', 'Not measured in this session'
            else:
                issues = problems(row)
                color = '#fecaca' if row.get('error') else '#bbf7d0' if row.get('qualified_programming') else '#fed7aa'
                label = 'G' if row.get('qualified_programming') else '!'
                message = '; '.join(issues) or 'No cell-level limit observed'
                message += '; signed/group MVM not tested; not yet selected for MNIST'
                details += f'<details id="c{wl}_{bl}"><summary>WL{wl} / BL{bl}: {html.escape(message)}</summary><pre>'
                details += html.escape(json.dumps(row, ensure_ascii=False, indent=2)) + '</pre></details>'
            table += f'<td style="background:{color}" title="{html.escape(message, quote=True)}"><a href="#c{wl}_{bl}">{label}</a></td>'
        table += '</tr>'
    return ('<!doctype html><html lang="ru"><meta charset="utf-8"><title>256-cell characterization</title>'
        '<style>body{font:15px sans-serif;margin:24px}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:5px}pre{white-space:pre-wrap}details{margin:8px 0}</style>'
        '<h1>Обследование 256 ячеек через MemriBoard</h1>'
        '<p>Зеленый: строгая проверка программирования пройдена. Оранжевый: измерена с ограничениями. '
        'Красный: ошибка. Серый: нет измерений. Наведите на ячейку или нажмите для подробностей.</p>'
        '<p>Это карта отдельных ячеек. Зеленый не означает пригодность для MNIST: совместный и разностный MVM еще не проверены. '
        'Промах записи или слабый сигнал сами по себе не исключают ячейку из дальнейшего исследования.</p><table>'
        + table + '</table><pre>' + html.escape(json.dumps(summary, ensure_ascii=False, indent=2))
        + '</pre>' + details + '</html>')
