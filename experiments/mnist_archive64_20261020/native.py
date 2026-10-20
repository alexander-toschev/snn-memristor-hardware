"""Конечное программирование и реальные запросы штатного MemriBoard SIM.

Здесь нет физической платы: cb_type=simulator явно выбирает SIM; board_type и электрические параметры идут из INI. Однако каждая
запись/проверка/MVM ниже действительно проходит публичный Connector. NumPy здесь
только выбирает следующий импульс и обрабатывает измерения; не вычисляет W@input.
"""
from __future__ import annotations
from collections.abc import Callable
import contextlib
import math
from pathlib import Path
import shutil
import time
from typing import TYPE_CHECKING, Protocol
import numpy as np
from numpy.typing import NDArray
from experiments.cifar_resumable_20261019.durable import Writer, atomic_json
from experiments.stable_cell_calibration_20261020.core import DriverOutput, safe
from .cache import ExactCache, Response
if TYPE_CHECKING:
    from manager.board import Connector

class Converters(Protocol):
    def convert_volt_to_dac(self, bits: int, reference: float, voltage: float) -> int: ...
    def convert_dac_to_volt(self, bits: int, reference: float, code: int) -> float: ...
    def convert_adc_to_volt(self, gain: int, bits: int, reference: float, code: int) -> float: ...

LEVELS=(.3,.6,.9,1.2,1.5,2.,2.5,3.)
MAX_PULSES=332800
PROGRAM_CALL_CAP=1683200
# После программирования дополнительно три readback на каждую из 3200 ног.
POST_READ_CAP=9600

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

    def pulse(self,wl: int,bl: int,sign: int,level: float,duration: float,**metadata: object) -> float:
        if sign not in (0,1) or not .3<=level<=3. or duration not in (.000125,.00001):raise ValueError('Invalid pulse')
        if self.decisions>=MAX_PULSES:raise ValueError('Finite programming decision cap')
        # CPU guard из свежего nativeR: SET≤2mA, RESET≤40mA как reviewed GUI.
        r=self.read_r(wl,bl,role='pulse_guard',**metadata);self.decisions+=1
        current=level/r;admitted=current<=(.002 if sign else .04)
        self.emit('pulse_decision',wl=wl,bl=bl,sign=sign,voltage_v=level,duration_s=duration,
                  resistance_ohm=r,predicted_current_a=current,admitted=admitted,metadata=metadata)
        if admitted:
            self.pulses+=1
            self.call(dict(mode_flag=7,wl=wl,bl=bl,vol=level,sign=sign,pulse_width=duration),role='program',**metadata)
        return self.read_g(wl,bl,role='post_program',**metadata)

    def program(self,weights: NDArray[np.float64],low: float,span: float,stage: int,out: Path) -> NDArray[np.float64]:
        if weights.shape!=(4,25) or not np.isfinite(weights).all() or np.any(weights<0):raise ValueError('Positive archived weights required')
        target=np.empty((4,25,2));actual=np.empty_like(target)
        for f in range(4):
            for bl in range(25):
                for leg in range(2):
                    wl=2*f+leg
                    # Обе ноги имеют ненулевой baseline L. Положительный W
                    # задаётся разностью Gplus-Gminus; baseline не игнорируется.
                    goal=low+(float(weights[f,bl])*span/float(weights[f].max()) if leg==0 else 0.)
                    target[f,bl,leg]=goal
                    meta=dict(stage=stage,filter=stage*4+f,leg=leg,target_g_s=goal)
                    g=self.read_g(wl,bl,role='initial',**meta)
                    for level in LEVELS:g=self.pulse(wl,bl,0,level,.000125,recipe='reset',**meta)
                    # CPU feedback только по G, без изображений/меток/accuracy.
                    # 10us = 10 шагов native SIM; эквивалентность HW не заявляем.
                    for attempt in range(96):
                        if abs(g-goal)<=.02*span:break
                        sign=1 if g<goal else 0
                        g=self.pulse(wl,bl,sign,1.7 if sign else 1.4,.00001,attempt=attempt,**meta)
                    actual[f,bl,leg]=self.read_g(wl,bl,role='leg_final',**meta)
            self.progress('Native program filter %d/64; calls=%d pulses=%d'%(stage*4+f+1,self.calls,self.pulses))
            self.writer.checkpoint(stage=stage,filter=stage*4+f,impacts=self.calls,pulses=self.pulses)
        # Все остальные записи данного тайла уже завершились. Три чтения на
        # ногу фиксируют конечную проводимость и обусловливают initial clamp.
        for f in range(4):
            for bl in range(25):
                for leg in range(2):actual[f,bl,leg]=self.read_g(2*f+leg,bl,role='stage_final',stage=stage)
        atomic_json(out/('program_%02d.json'%stage),dict(target_g_s=target.tolist(),measured_g_s=actual.tolist(),
            tolerance_g_s=.02*span,target_miss_fraction=float(np.mean(np.abs(target-actual)>.02*span))))
        # Native .cb — последняя штатная сериализация записи; last_resistance
        # последующих read может отличаться в RAM. Не называем exact RAM snapshot.
        shutil.copyfile(out/'array.cb',out/('array_stage_%02d.cb'%stage))
        return actual

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

    def measure_lut(self,stage: int,masks: list[int],native_masks: list[int],weights: NDArray[np.float64],out: Path) -> NDArray[np.float32]:
        # Первый проход, точный повтор и заключительный проход в обратном
        # порядке. Никакой усредняющей/приближённой мемоизации: любая разница
        # ADC/V MVM запрещает LUT и останавливает опыт.
        cache=ExactCache(stage,native_masks)
        for pass_id in range(3):
            order=native_masks if pass_id<2 else list(reversed(native_masks))
            for wl in range(8):
                for mask in order:cache.observe(stage,wl,mask,self.query(wl,mask,stage,pass_id))
            self.writer.checkpoint(stage=stage,lut_pass=pass_id,impacts=self.calls)
            self.progress('Native LUT stage %d/16 pass %d/3; calls=%d'%(stage+1,pass_id+1,self.calls))
        cache.seal()
        # CPU, label-free калибровка единиц: G измерено при штатном read, но
        # MVM при .05V. Поэтому alpha получаем из native singleton-response,
        # а не предполагаем линейный перенос проводимости с .3V на .05V.
        zero=np.asarray([cache.get(stage,2*f,0).voltage-cache.get(stage,2*f+1,0).voltage for f in range(4)])
        singleton=np.asarray([[cache.get(stage,2*f,1<<b).voltage-cache.get(stage,2*f+1,1<<b).voltage
            for b in range(25)] for f in range(4)])-zero[:,None]
        denominator=(singleton*singleton).sum(axis=1)
        alpha=(weights*singleton).sum(axis=1)/denominator
        if not np.all(np.isfinite(alpha)) or np.any(alpha<=0):raise ValueError('Unresolvable singleton calibration')
        lut=np.empty((len(masks),4),dtype=np.float32)
        for i,mask in enumerate(masks):
            # Cache hit НЕ новый Connector call. Полная сумма всех активных
            # BL уже выполнена native mode10. CPU лишь вычитает две WL-ноги.
            for f in range(4):
                difference=cache.get(stage,2*f,mask).voltage-cache.get(stage,2*f+1,mask).voltage-zero[f]
                lut[i,f]=difference*alpha[f]
        atomic_json(out/('lut_%02d.json'%stage),dict(stage=stage,alpha=alpha.tolist(),zero_difference_v=zero.tolist(),
            singleton_voltage_v=singleton.tolist(),singleton_weight_rmse=float(np.sqrt(np.mean((singleton*alpha[:,None]-weights)**2))),
            responses=[dict(wl=wl,mask=m,adc=r.adc,voltage_v=r.voltage,guards_r=r.guards_r,first_task_id=r.task_id,verified_passes=3)
                       for (wl,m),r in sorted(cache.values.items())],exact_repeat_and_endcheck=True))
        np.save(out/('lut_%02d.npy'%stage),lut,allow_pickle=False)
        cache.invalidate() # Следующая запись создаёт новый stage/generation.
        return lut
