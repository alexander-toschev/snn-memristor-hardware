"""Короткое сравнение read paths mode9/10 без SET/RESET и смены проводимостей.

Напряжения 0,025–0,1 В подаются только публичным Connector. R используется
для свежего защитного ограничения; измеренный MVM никогда не заменяется V/R.
"""
from __future__ import annotations

from collections.abc import Callable
import contextlib
import html
import json
import math
import statistics
import time
from typing import TYPE_CHECKING

from experiment_common.durable import Writer
from experiment_common.native_output import DriverOutput, safe
from experiment_common.selective_metrology import Converters, Reading, Source, number

if TYPE_CHECKING:
    from manager.board import Connector

CELLS = tuple((w, b) for w in (0, 7) for b in (14, 15, 16, 17, 20, 30, 31))
VOLTAGES = (.025, .05, .075, .1)
MAX_CALLS = 1344
SPEC: dict[str, object] = dict(
    schema='mvm-voltage-readonly-v1', cells=[list(c) for c in CELLS], voltages_v=list(VOLTAGES),
    repeats=3, modes=[9,10], maximum_impacts=MAX_CALLS, maximum_pulses=0,
    maximum_wall_seconds=600, current_guard='max(.1,nativeDAC_to_V(requested))/fresh_R <= .005 A',
    predicted_output_limit_v=3.3, measured_output_limit_v=4.,
    bracket=['zero','fresh_mode7_R','signal','zero'],
    programming_performed=False, analog_gains_assumed_equal=False,
    private_runtime_status='private_runtime_not_independently_reverified')


class Collector:
    def __init__(self, connector: Connector, converters: Converters, writer: Writer, source: Source) -> None:
        self.connector: Connector = connector
        self.cv, self.writer, self.source = converters, writer, source
        self.calls = 0
        self.started = time.monotonic()
        self.checkpoints: list[dict[str, object]] = []

    def emit(self, event: str, **fields: object) -> None:
        self.writer.write(dict(event=event, source=self.source, monotonic_s=time.monotonic(), **fields))

    def call(self, mode: int, wl: int, bl: int, voltage: float, **metadata: object) -> Reading:
        if self.calls >= MAX_CALLS or time.monotonic()-self.started >= 600:
            raise TimeoutError('Read-only finite budget')
        if type(wl) is not int or type(bl) is not int or (wl,bl) not in CELLS or voltage not in (0.,*VOLTAGES):
            raise ValueError('Unplanned address/voltage')
        task: dict[str, int | float | list[float]] = dict(mode_flag=mode, wl=wl)
        if mode == 7:
            if voltage != 0.:
                raise ValueError('SET/RESET forbidden')
            task.update(bl=bl, vol=0., sign=0, pulse_width=0., id=self.calls+1)
        elif mode == 9:
            task.update(bl=bl, vol=voltage, id=0)
        elif mode == 10:
            task.update(vol=[voltage if b==bl else 0. for b in range(32)], id=0)
        else:
            raise ValueError('Unplanned mode')
        self.calls += 1
        self.emit('request', task_id=self.calls, command_si=task, metadata=metadata)
        values = task['vol']
        copy = {**task, 'vol': list(values) if isinstance(values,list) else values}
        with contextlib.redirect_stdout(DriverOutput(self,'stdout')), contextlib.redirect_stderr(DriverOutput(self,'stderr')):
            response = self.connector.impact(copy)
        self.emit('result', task_id=self.calls, command_si=task, native_result=safe(response))
        if not isinstance(response,(list,tuple)) or len(response)!=3:
            raise ValueError('Invalid response shape')
        resistance, echo, adc = response
        if type(echo) is not int or echo!=task['id'] or type(adc) is not int or not 0<=adc<=16382:
            raise ValueError('Invalid native ID/ADC')
        r = number(resistance)
        if mode==7 and (r<=1e-8 or adc<20):
            raise ValueError('Invalid mode7 R/readADC')
        return Reading(task_id=self.calls,resistance_ohm=r,adc=adc)

    def diagnose(self, progress: Callable[[str],None]) -> None:
        for repeat in range(3):
            for wl,bl in CELLS:
                for voltage in VOLTAGES:
                    # Меняем порядок режимов между повторами, не напряжения/границы.
                    for mode in ((9,10) if repeat%2==0 else (10,9)):
                        meta: dict[str,object] = dict(cell_wl=wl,cell_bl=bl,repeat=repeat,voltage_v=voltage,diagnostic_mode=mode)
                        before=self.call(mode,wl,bl,0.,role='zero_before',**meta)
                        guard=self.call(7,wl,bl,0.,role='guard',**meta)
                        code=self.cv.convert_volt_to_dac(12,5,voltage)
                        decoded=self.cv.convert_dac_to_volt(12,5,code)
                        current=max(.1,decoded)/guard['resistance_ohm']
                        if current>.005 or 330*current>=3.3:
                            raise ValueError('Fresh .1V conservative current/headroom guard')
                        signal=self.call(mode,wl,bl,voltage,role='signal',**meta)
                        after=self.call(mode,wl,bl,0.,role='zero_after',**meta)
                        vs=[self.cv.convert_adc_to_volt(1,14,5,r['adc']) for r in (before,signal,after)]
                        if any(not math.isfinite(v) or v>4. for v in vs):
                            raise ValueError('Measured ADC headroom')
                        row: dict[str,object]=dict(**meta,wl=wl,bl=bl,before=before,guard=guard,signal=signal,after=after,
                            dac_code=code,native_decoded_v=decoded,guard_current_a=current,
                            centered_adc=signal['adc']-(before['adc']+after['adc'])/2,
                            centered_v=vs[1]-(vs[0]+vs[2])/2,mode7_g_s=1/guard['resistance_ohm'])
                        self.checkpoints.append(row)
                        self.emit('voltage_bracket',**row)
                self.writer.checkpoint(repeat=repeat,wl=wl,bl=bl,impacts=self.calls)
                progress(f'Pass{repeat+1}/3 WL{wl} BL{bl}: {self.calls}/{MAX_CALLS}; 0 writes')


def summarize(rows: list[dict[str,object]]) -> list[dict[str,object]]:
    result: list[dict[str,object]]=[]
    for wl,bl in CELLS:
        for voltage in VOLTAGES:
            modes: dict[str,object]={}
            for mode in (9,10):
                selected=[r for r in rows if r['wl']==wl and r['bl']==bl and r['voltage_v']==voltage and r['diagnostic_mode']==mode]
                values=[number(r['centered_adc']) for r in selected]
                gs=[number(r['mode7_g_s']) for r in selected]
                modes[str(mode)]=dict(count=len(values),centered_adc=values,mean_adc=statistics.mean(values) if values else None,
                    sd_adc=statistics.stdev(values) if len(values)>1 else None,mode7_g_s=gs,
                    mean_g_s=statistics.mean(gs) if gs else None)
            result.append(dict(wl=wl,bl=bl,voltage_v=voltage,modes=modes))
    return result


def render(rows: list[dict[str,object]], summary: dict[str,object]) -> str:
    from experiment_common.selective_metrology import object_dict
    body=''
    for row in rows:
        modes=object_dict(row['modes'])
        body+='<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in
            (row['wl'],row['bl'],row['voltage_v'],object_dict(modes['9'])['mean_adc'],object_dict(modes['10'])['mean_adc']))+'</tr>'
    return ('<!doctype html><html lang="ru"><meta charset="utf-8"><title>MVM voltage sweep</title>'
            '<style>body{font:16px sans-serif;max-width:1100px;margin:auto}td,th{padding:6px;border:1px solid #aaa}table{border-collapse:collapse}pre{white-space:pre-wrap}</style>'
            '<h1>Сравнение mode9/10 при 0,025–0,1 В</h1><p>Средние центрированные ADC. Разные аналоговые маршруты; не требуем равенства. Нулевые/неполные данные не заменены расчётом.</p>'
            '<table><tr><th>WL</th><th>BL</th><th>V</th><th>mode9 ADC</th><th>mode10 ADC</th></tr>'+body+'</table><pre>'
            +html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre></html>')
