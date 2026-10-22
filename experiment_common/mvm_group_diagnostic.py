"""Полные групповые векторы через публичный Connector, без записи.

Сумма V/R служит только guard. Реальная сумма токов и ADC получаются mode10;
CPU вычисляет описательные разности измерений, но не заменяет ими forward.
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
from experiment_common.selective_metrology import Converters, Reading, Source, number, object_dict

if TYPE_CHECKING:
    from manager.board import Connector

MASKS: tuple[tuple[str,tuple[int,...]],...] = (
    ('lower0_7',tuple(range(8))),('lower8_15',tuple(range(8,16))),
    ('upper16_23',tuple(range(16,24))),('upper24_31',tuple(range(24,32))),
    ('upper16_31',tuple(range(16,32))),('upper16_30',tuple(range(16,31))),
    ('BL31',(31,)),('BL14',(14,)),('BL14_plus_upper16_30',(14,*range(16,31))))
MAX_CALLS=648
SPEC: dict[str,object]=dict(schema='mvm-groups-readonly-v1',wls=[0,7],repeats=3,
    masks=[dict(name=name,bls=list(bls)) for name,bls in MASKS],amplitude_v=.05,
    maximum_impacts=MAX_CALLS,maximum_pulses=0,maximum_wall_seconds=600,
    guard='sum(max(.05,nativeDAC_to_V)/fresh_R_active) <= .005 A; 330I <3.3V',
    measured_output_limit_v=4.,sequence=['zero','freshR_each_active','full_vector_MVM','zero'],
    private_runtime_status='private_runtime_not_independently_reverified',
    no_ideal_sum=True,programming_performed=False)


class Collector:
    def __init__(self,connector: Connector,converters: Converters,writer: Writer,source: Source) -> None:
        self.connector: Connector=connector
        self.cv,self.writer,self.source=converters,writer,source
        self.calls=0
        self.started=time.monotonic()
        self.checkpoints: list[dict[str,object]]=[]

    def emit(self,event: str,**fields: object) -> None:
        self.writer.write(dict(event=event,source=self.source,monotonic_s=time.monotonic(),**fields))

    def call(self,mode: int,wl: int,bls: tuple[int,...],**metadata: object) -> Reading:
        if self.calls>=MAX_CALLS or time.monotonic()-self.started>=600:
            raise TimeoutError('Group diagnostic finite budget')
        if type(wl) is not int or wl not in (0,7) or any(type(b) is not int or not 0<=b<32 for b in bls) or len(set(bls))!=len(bls):
            raise ValueError('Invalid address')
        command: dict[str,int|float|list[float]]=dict(mode_flag=mode,wl=wl)
        if mode==7 and len(bls)==1:
            command.update(bl=bls[0],vol=0.,sign=0,pulse_width=0.,id=self.calls+1)
        elif mode==10 and (not bls or bls in [mask for _,mask in MASKS]):
            command.update(vol=[.05 if b in bls else 0. for b in range(32)],id=0)
        else:
            raise ValueError('Only fixed groups/read-only mode7 permitted')
        self.calls+=1
        self.emit('request',task_id=self.calls,command_si=command,metadata=metadata)
        volts=command['vol']
        with contextlib.redirect_stdout(DriverOutput(self,'stdout')),contextlib.redirect_stderr(DriverOutput(self,'stderr')):
            response=self.connector.impact({**command,'vol':list(volts) if isinstance(volts,list) else volts})
        self.emit('result',task_id=self.calls,command_si=command,native_result=safe(response))
        if not isinstance(response,(list,tuple)) or len(response)!=3:
            raise ValueError('Invalid native response shape')
        resistance,echo,adc=response
        if type(echo) is not int or echo!=command['id'] or type(adc) is not int or not 0<=adc<=16382:
            raise ValueError('Invalid ID/ADC')
        r=number(resistance)
        if mode==7 and (r<=1e-8 or adc<20):
            raise ValueError('Invalid native R')
        return Reading(task_id=self.calls,resistance_ohm=r,adc=adc)

    def diagnose(self,progress: Callable[[str],None]) -> None:
        for repeat in range(3):
            for wl in (0,7):
                for name,bls in MASKS:
                    meta: dict[str,object]=dict(repeat=repeat,mask=name,cell_wl=wl)
                    before=self.call(10,wl,(),role='zero_before',**meta)
                    guards=[self.call(7,wl,(bl,),role='guard',active_bl=bl,**meta) for bl in bls]
                    dac=self.cv.convert_volt_to_dac(12,5,.05)
                    v=max(.05,self.cv.convert_dac_to_volt(12,5,dac))
                    current=sum(v/g['resistance_ohm'] for g in guards)
                    if current>.005 or 330*current>=3.3:
                        raise ValueError('Fresh summed-current/headroom guard')
                    signal=self.call(10,wl,bls,role='group_signal',**meta)
                    after=self.call(10,wl,(),role='zero_after',**meta)
                    volts=[self.cv.convert_adc_to_volt(1,14,5,r['adc']) for r in (before,signal,after)]
                    if any(not math.isfinite(v) or v>4. for v in volts):
                        raise ValueError('Measured group headroom')
                    row: dict[str,object]=dict(wl=wl,repeat=repeat,mask=name,bls=list(bls),before=before,
                        guards=guards,signal=signal,after=after,predicted_sum_current_a=current,
                        centered_adc=signal['adc']-(before['adc']+after['adc'])/2,
                        centered_v=volts[1]-(volts[0]+volts[2])/2)
                    self.checkpoints.append(row)
                    self.emit('group_bracket',**row)
                    self.writer.checkpoint(wl=wl,repeat=repeat,mask=name,impacts=self.calls)
                progress(f'Pass{repeat+1}/3 WL{wl}: {self.calls}/{MAX_CALLS}; 0 writes')


def summarize(rows: list[dict[str,object]]) -> list[dict[str,object]]:
    result: list[dict[str,object]]=[]
    for wl in (0,7):
        by={(int(number(r['repeat'])),str(r['mask'])):number(r['centered_adc']) for r in rows if r['wl']==wl}
        for name,bls in MASKS:
            values=[by[k,name] for k in range(3) if (k,name) in by]
            result.append(dict(wl=wl,mask=name,bls=list(bls),count=len(values),centered_adc=values,
                mean_adc=statistics.mean(values) if values else None,
                sd_adc=statistics.stdev(values) if len(values)>1 else None))
        # Измеренная приращённая реакция. Не считаем аналоговую сумму идеальной.
        differences=[by[k,'BL14_plus_upper16_30']-by[k,'BL14'] for k in range(3)
                     if (k,'BL14_plus_upper16_30') in by and (k,'BL14') in by]
        result.append(dict(wl=wl,mask='measured_mixed_minus_BL14',count=len(differences),
            centered_adc=differences,mean_adc=statistics.mean(differences) if differences else None,
            interpretation='Adjacent measured responses; not proof of additivity or isolated channel contribution.'))
    return result


def render(rows: list[dict[str,object]],summary: dict[str,object]) -> str:
    body=''.join('<tr>'+''.join('<td>'+html.escape(str(r[k]))+'</td>' for k in ('wl','mask','count','mean_adc'))+'</tr>' for r in rows)
    return ('<!doctype html><html lang="ru"><meta charset="utf-8"><title>MVM groups</title>'
            '<style>body{font:16px sans-serif;max-width:1100px;margin:auto}td,th{padding:6px;border:1px solid #aaa}pre{white-space:pre-wrap}</style>'
            '<h1>Групповые mode10 при0,05 В</h1><p>Все группы измерены через Connector. Разность mixed−BL14 описательная, не идеальный расчёт суммы.</p>'
            '<table><tr><th>WL</th><th>Маска</th><th>Повторы</th><th>ADC mean</th></tr>'+body+'</table><pre>'
            +html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre></html>')
