"""Конечное программирование и реальные запросы штатного MemriBoard SIM.

Здесь нет физической платы: cb_type=simulator явно выбирает SIM; board_type и электрические параметры идут из INI. Однако каждая
запись/проверка/MVM ниже действительно проходит публичный Connector. NumPy здесь
только выбирает следующий импульс и обрабатывает измерения; не вычисляет W@input.
"""
from __future__ import annotations
from collections.abc import Callable
import contextlib
import math
import time
from typing import TYPE_CHECKING, Protocol
import numpy as np
from experiment_common.durable import Writer
from experiment_common.native_output import DriverOutput, safe
from experiment_common.exact_cache import Response
if TYPE_CHECKING:
    from manager.board import Connector

class Converters(Protocol):
    def convert_volt_to_dac(self, bits: int, reference: float, voltage: float) -> int: ...
    def convert_dac_to_volt(self, bits: int, reference: float, code: int) -> float: ...
    def convert_adc_to_volt(self, gain: int, bits: int, reference: float, code: int) -> float: ...

LEVELS=(.3,.6,.9,1.2,1.5,2.,2.5,3.)

class Collector:
    def __init__(self, connector: Connector, converters: Converters, writer: Writer,
                 cap: int, progress: Callable[[str],None]) -> None:
        self.connector: Connector=connector
        self.cv=converters;self.writer=writer;self.cap=cap;self.progress=progress
        self.calls=0;self.pulses=0;self.decisions=0;self.started=time.monotonic()

    def emit(self,event: str,**fields: object) -> None:
        self.writer.write(dict(event=event,source='simulator',monotonic_s=time.monotonic(),**fields))

    def call(self, command: dict[str,int|float|list[float]], **metadata: object) -> tuple[int,float,int]:
        # Бюджет проверяем до обращения к SIM. Тот же публичный SI-интерфейс,
        # что у GUI: mode7 для чтения/записи, mode10 для суммы токов.
        if self.calls>=self.cap:raise ValueError('Finite native call cap exhausted')
        self.calls+=1;task={**command,'id':self.calls if command['mode_flag']==7 else 0}
        self.emit('request',task_id=self.calls,command_si=task,metadata=metadata)
        with contextlib.redirect_stdout(DriverOutput(self,'stdout')),contextlib.redirect_stderr(DriverOutput(self,'stderr')):
            # Connector делает shallow-copy ticket и преобразует vol-list
            # in-place в DAC. Отдельная копия сохраняет оба raw command_si
            # в вольтах; численные значения, переданные Connector, прежние.
            native=self.connector.impact({**task,'vol':list(task['vol']) if isinstance(task['vol'],list) else task['vol']})
        self.emit('result',task_id=self.calls,command_si=task,native_result=safe(native))
        if not isinstance(native,(tuple,list)) or len(native)!=3:raise ValueError('Invalid native response')
        resistance,echo,adc=native
        if type(echo) is not int or echo!=task['id'] or type(adc) is not int or not 0<=adc<=16382:
            raise ValueError('Invalid native ID/ADC')
        if command['mode_flag']==7:
            if type(resistance) not in (int,float) or not math.isfinite(resistance) or resistance<=1e-8 or adc<20:
                raise ValueError('Invalid native resistance/ADC')
        return self.calls,float(resistance),adc

    def read_r(self,wl: int,bl: int,**metadata: object) -> float:
        if not 0<=wl<8 or not 0<=bl<25:raise ValueError('Invalid mapped cell')
        # vol=0 mode7 означает штатное чтение R, не нулевое сопротивление.
        # Реальное напряжение чтения задаёт драйвер/INI (0.3V nominal).
        return self.call(dict(mode_flag=7,wl=wl,bl=bl,vol=0.,sign=0,pulse_width=0.),**metadata)[1]

    def read_g(self,wl: int,bl: int,**metadata: object) -> float:
        # CPU: G=1/R [Siemens], среднее трёх реально измеренных сопротивлений.
        rs=[self.read_r(wl,bl,repeat=i,**metadata) for i in range(3)]
        g=float(np.mean([1/r for r in rs]))
        self.emit('conductance_read',wl=wl,bl=bl,resistance_ohm=rs,mean_g_s=g,metadata=metadata)
        return g


    def query(self,wl: int,mask: int,stage: int,pass_id: int) -> Response:
        bits=[b for b in range(25) if mask&(1<<b)]
        if not 0<=mask<1<<25:raise ValueError('Expected25-bit native mask')
        rs=tuple(self.read_r(wl,b,role='mvm_guard',stage=stage,mask=mask,pass_id=pass_id) for b in bits)
        # Реальное суммирование всех активных токов (до25) выполняет native mode10. Здесь CPU
        # лишь проверяет консервативный предел тока/ADC по свежим R.
        decoded=self.cv.convert_dac_to_volt(12,5,self.cv.convert_volt_to_dac(12,5,.05))
        current=sum(max(.05,decoded)/r for r in rs)
        if current>.005 or 330*current>=3.3:raise ValueError('Native MVM current/headroom guard; no ideal fill')
        task,_,adc=self.call(dict(mode_flag=10,wl=wl,vol=[.05 if b in bits else 0. for b in range(32)]),
                            role='mvm',stage=stage,mask=mask,pass_id=pass_id)
        v=float(self.cv.convert_adc_to_volt(1,14,5,adc))
        if not math.isfinite(v) or v>4.:raise ValueError('MVM output range')
        return Response(adc,v,rs,task)

