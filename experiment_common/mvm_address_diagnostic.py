"""Read-only проверка адресов mode10: штатный вектор32, без SET/RESET.

Mode7 измеряет R другим аналоговым маршрутом. Его R не подставляется вместо
mode10 и не считается доказательством ожидаемой амплитуды MVM.
"""
from __future__ import annotations

from collections.abc import Callable
import contextlib
import html
import json
import math
import statistics

from experiment_common.selective_metrology import Collector as ReadOnlyCollector, Reading
from experiment_common.selective_metrology import number
from experiment_common.native_output import DriverOutput, safe

ORDERS = (tuple(range(32)), tuple(reversed(range(32))),
          tuple(b for low in range(16) for b in (low, low + 16)))
MODE9_CELLS = tuple((w, b) for w in (0, 7) for b in (14, 15, 16, 17, 20, 30, 31))
MAX_CALLS = 3240
SPEC: dict[str, object] = dict(
    schema='mvm-address-readonly-v1', wls=list(range(8)), bl_orders=[list(x) for x in ORDERS],
    amplitude_v=.05, repeats=3, maximum_impacts=MAX_CALLS, maximum_pulses=0,
    maximum_wall_seconds=600, modes=[7, 10, 9], mode10_id=0, vector_length=32,
    mode9_cells=[list(c) for c in MODE9_CELLS], mode9_repeats=3, mode9_additional_calls=168,
    sequence=['mode10_zero', 'fresh_mode7_R', 'mode10_singleton', 'mode10_zero'],
    programmable_range_validated=False, new_calibration=False,
    interpretation='Compare independently measured mode7 and mode10; do not equate their analog transfer functions.',
    prior_observation='Survey reported changing mode7 conductance but zero mode10 for BL16..30; diagnosis, not a confirmed cause.',
    private_runtime_status='private_runtime_not_independently_reverified')


class Collector(ReadOnlyCollector):
    """Наследуется запрет записи и проверка public-native response/ADC."""

    def call(self, command: dict[str, int | float | list[float]], **metadata: object) -> Reading:
        if self.calls >= MAX_CALLS:
            raise ValueError('Address diagnostic call cap')
        if command['mode_flag'] == 9:
            # Штатный GUI IVC: скалярные 0/.05 В, id=0. Это независимый
            # read path, не внутренний driver вызов и не SET/RESET.
            if self.clock() - self.started >= 600 or command.get('vol') not in (0., .05):
                raise ValueError('Mode9 read-only voltage/wall limit')
            wl, bl = command.get('wl'), command.get('bl')
            if type(wl) is not int or type(bl) is not int or (wl, bl) not in MODE9_CELLS or set(command) != {'mode_flag', 'vol', 'wl', 'bl'}:
                raise ValueError('Mode9 fixed address/schema')
            self.calls += 1
            ticket = {**command, 'id': 0}
            self.emit('request', task_id=self.calls, command_si=ticket, metadata=metadata)
            with contextlib.redirect_stdout(DriverOutput(self, 'stdout')), contextlib.redirect_stderr(DriverOutput(self, 'stderr')):
                response = self.connector.impact(dict(ticket))
            self.emit('result', task_id=self.calls, command_si=ticket, native_result=safe(response))
            if not isinstance(response, (list, tuple)) or len(response) != 3:
                raise ValueError('Mode9 response shape')
            resistance, echo, adc = response
            if type(echo) is not int or echo != 0 or type(adc) is not int or not 0 <= adc <= 16382:
                raise ValueError('Mode9 native ID/ADC')
            return Reading(task_id=self.calls, resistance_ohm=number(resistance), adc=adc)
        return super().call(command, **metadata)

    def voltage(self, wl: int, bl: int, signal: bool, mode: int, **metadata: object) -> Reading:
        if mode == 9:
            return self.call(dict(mode_flag=9, wl=wl, bl=bl, vol=.05 if signal else 0.), **metadata)
        return self.vector(wl, bl if signal else None, **metadata)

    def diagnose(self, progress: Callable[[str], None]) -> list[dict[str, object]]:
        observations: list[dict[str, object]] = []
        phases = [(10, repeat, wl, order) for repeat, order in enumerate(ORDERS) for wl in range(8)]
        phases += [(9, repeat, wl, tuple(b for w, b in MODE9_CELLS if w == wl)) for repeat in range(3) for wl in (0, 7)]
        for mode, repeat, wl, order in phases:
                for bl in order:
                    metadata: dict[str, object] = dict(wl_target=wl, bl_target=bl, repeat=repeat, diagnostic_mode=mode)
                    before = self.voltage(wl, bl, False, mode, role='zero_before', **metadata)
                    guard = self.call(dict(mode_flag=7, wl=wl, bl=bl, vol=0., sign=0, pulse_width=0.), role='guard', **metadata)
                    dac = self.cv.convert_volt_to_dac(12, 5, .05)
                    voltage = max(.05, self.cv.convert_dac_to_volt(12, 5, dac))
                    current = voltage / guard['resistance_ohm']
                    if current > .005 or 330 * current >= 3.3:
                        raise ValueError('Fresh current/headroom guard failed')
                    signal = self.voltage(wl, bl, True, mode, role='signal', **metadata)
                    after = self.voltage(wl, bl, False, mode, role='zero_after', **metadata)
                    volts = [self.cv.convert_adc_to_volt(1, 14, 5, item['adc']) for item in (before, signal, after)]
                    if any(not math.isfinite(v) or v > 4. for v in volts):
                        raise ValueError('Measured MVM headroom')
                    row: dict[str, object] = dict(
                        wl=wl, bl=bl, repeat=repeat, mode=mode, before=before, guard=guard,
                        signal=signal, after=after, native_dac_code=dac,
                        predicted_current_a=current, centered_adc=signal['adc'] - (before['adc'] + after['adc'])/2,
                        centered_v=volts[1] - (volts[0] + volts[2])/2,
                        mode7_g_s=1/guard['resistance_ohm'])
                    observations.append(row)
                    self.checkpoints.append(row)
                    self.emit('diagnostic_bracket', **row)
                self.writer.checkpoint(repeat=repeat, wl=wl, impacts=self.calls)
                progress(f'Mode{mode} pass {repeat+1}/3 WL{wl}: {self.calls}/{MAX_CALLS} calls; 0 writes')
        return observations


def summarize(rows: list[dict[str, object]], mode: int = 10) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for wl in range(8):
        for bl in range(32):
            found = [r for r in rows if r['wl'] == wl and r['bl'] == bl and r['mode'] == mode]
            values = [number(r['centered_adc']) for r in found]
            conductance = [number(r['mode7_g_s']) for r in found]
            output.append(dict(wl=wl, bl=bl, complete=len(found)==3,
                               centered_adc_by_order=values, mode7_g_s_by_order=conductance,
                               mean_adc=statistics.mean(values) if values else None,
                               sd_adc=statistics.stdev(values) if len(values)>1 else None,
                               all_mvm_zero=all(v==0 for v in values) if len(found)==3 else None))
    return output


def render(rows: list[dict[str, object]], summary: dict[str, object]) -> str:
    body = '<tr><th>WL / BL</th>' + ''.join(f'<th>{b}</th>' for b in range(32)) + '</tr>'
    for wl in range(8):
        body += f'<tr><th>{wl}</th>'
        for row in rows[wl*32:(wl+1)*32]:
            color = '#ddd' if not row['complete'] else '#ffd58a' if row['all_mvm_zero'] else '#c2ddfa'
            title = html.escape(json.dumps(row, ensure_ascii=False), quote=True)
            body += f'<td style="background:{color}" title="{title}">{row["mean_adc"]}</td>'
        body += '</tr>'
    return ('<!doctype html><html lang="ru"><meta charset="utf-8"><title>Диагностика mode10</title>'
            '<style>body{font:14px sans-serif}td,th{border:1px solid #999;padding:4px}table{border-collapse:collapse}pre{white-space:pre-wrap}</style>'
            '<h1>Средний центрированный mode10, ADC</h1><p>Жёлтый: три нулевых отклика; голубой: есть ненулевой отклик; серый: неполные данные. Это не карта пригодности/программируемости.</p><table>'
            + body + '</table><pre>' + html.escape(json.dumps(summary, ensure_ascii=False, indent=2)) + '</pre></html>')
