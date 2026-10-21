"""Единственный native массив: initialization, online pulses, точные MVM.

Все чтения/импульсы идут через публичный Connector родительского Collector.
Новых прямых обращений к драйверу, состояниям модели или GPIO нет.
"""
from __future__ import annotations
from collections.abc import Callable
from pathlib import Path
import math
from typing import TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray
from experiments.mnist_archive64_20261020 import native
from experiments.mnist_archive64_20261020.cache import ExactCache
from experiments.cifar_resumable_20261019.durable import Writer,atomic_json
if TYPE_CHECKING:
    from manager.board import Connector

DEADBAND=.002
INIT_CALLS=105800
INIT_PULSES=20800


class Collector(native.Collector):
    def __init__(self,connector: Connector,converters: native.Converters,writer: Writer,cap: int,
                 pulse_cap: int,span: float,progress: Callable[[str],None]) -> None:
        super().__init__(connector,converters,writer,cap,progress)
        self.pulse_cap=pulse_cap;self.span=span;self.generation=0
        self.active_cache: ExactCache|None=None
        self.learning_stats: dict[str,int]={name:0 for name in ('desired_set','desired_reset','deadband','guard_skipped','admitted_set','admitted_reset','delta_expected','delta_opposite','delta_zero','out_of_range')}
        actual=float(converters.convert_dac_to_volt(12,5,converters.convert_volt_to_dac(12,5,.05)))
        if not math.isfinite(actual) or actual<=0 or span<=0:raise ValueError('Invalid physical scale')
        # Vout≈330ohm × Vin[V] × G[S]. Деление на330*Vin*span переводит
        # измеренную разность выходов в nominal W units. Это фиксированная
        # физическая шкала, НЕ regression к desiredW и НЕ её refit после STDP.
        self.gain=1./(330.*actual*span);self.decoded_input_v=actual

    def invalidate(self) -> None:
        if self.active_cache is not None:self.active_cache.invalidate()
        self.active_cache=None;self.generation+=1

    def pulse(self,wl: int,bl: int,sign: int,level: float,duration: float,**metadata: object) -> float:
        # SET/RESET накопительно изменяют проводимость мемристора. RESET здесь
        # НЕ обнуление мембранного потенциала нейрона. Цифровой LIF выполняет
        # отдельный membrane reset; нейронная схема с C_out/RL/opamp и парой
        # мемристоров не реализована текущим crossbar mode10.
        # Собственные конечные limits: global MAX_PULSES старого опыта не
        # используется. Токовые/voltage/ADC guards остаются теми же.
        if sign not in (0,1) or not .3<=level<=3. or duration not in (.000125,.00001):
            raise ValueError('Invalid pulse')
        if self.decisions>=self.pulse_cap:raise ValueError('Finite online pulse decision cap')
        r=self.read_r(wl,bl,role='pulse_guard',**metadata);self.decisions+=1
        current=level/r;admitted=current<=(.002 if sign else .04)
        self.emit('pulse_decision',wl=wl,bl=bl,sign=sign,voltage_v=level,duration_s=duration,
            predicted_current_a=current,resistance_ohm=r,admitted=admitted,metadata=metadata)
        if admitted:
            self.invalidate();self.pulses+=1
            self.call(dict(mode_flag=7,wl=wl,bl=bl,vol=level,sign=sign,pulse_width=duration),role='program',**metadata)
        return self.read_g(wl,bl,role='post_program',**metadata)

    def read_array(self,**metadata: object) -> NDArray[np.float64]:
        # Все200 используемых ног: каждая3nativeR. Не предполагаем, что
        # baseline-нога неизменна после других импульсов/чтений.
        return np.asarray([[[self.read_g(2*f+leg,b,**metadata) for leg in range(2)]
            for b in range(25)] for f in range(4)],dtype=np.float64)

    def initialize(self,weights: NDArray[np.float64],low: float,stage: int,out: Path) -> None:
        if weights.shape!=(4,25) or not np.isfinite(weights).all() or np.any((weights<0)|(weights>1)):
            raise ValueError('Expected fresh clipped weights')
        target=np.empty((4,25,2),dtype=np.float64)
        for f in range(4):
            for bl in range(25):
                for leg in range(2):
                    wl=2*f+leg;goal=low+(float(weights[f,bl])*self.span if leg==0 else 0.)
                    target[f,bl,leg]=goal;meta=dict(stage=stage,phase='initialization',target_g_s=goal)
                    g=self.read_g(wl,bl,role='initial',**meta)
                    for level in native.LEVELS:g=self.pulse(wl,bl,0,level,.000125,**meta)
                    for attempt in range(96):
                        if abs(g-goal)<=.02*self.span:break
                        sign=1 if g<goal else 0
                        g=self.pulse(wl,bl,sign,1.7 if sign else 1.4,.00001,attempt=attempt,**meta)
                    self.read_g(wl,bl,role='leg_final',**meta)
            self.writer.checkpoint(stage=stage,filter=f,impacts=self.calls,pulses=self.pulses)
            self.progress('Initialize block%d filter%d/4 calls=%d pulses=%d'%(stage,f+1,self.calls,self.pulses))
        actual=self.read_array(stage=stage,phase='initialization_final')
        atomic_json(out/('initialization_%02d.json'%stage),dict(target_g_s=target.tolist(),measured_g_s=actual.tolist(),
            target_miss_fraction=float(np.mean(np.abs(actual-target)>.02*self.span))))

    def measure(self,masks: list[int],stage: int,phase: str,**metadata: object) -> NDArray[np.float32]:
        keys=sorted(set(masks)|{0});cache=ExactCache(self.generation,keys);self.active_cache=cache
        # Три exact прохода до reuse. Каждое окно измерено заново; cache не
        # переносится между timestep, даже если не было импульса.
        for repeat in range(3):
            for wl in range(8):
                for mask in (keys if repeat<2 else list(reversed(keys))):
                    cache.observe(self.generation,wl,mask,self.query(wl,mask,stage,repeat))
        cache.seal()
        zero=[cache.get(self.generation,2*f,0).voltage-cache.get(self.generation,2*f+1,0).voltage for f in range(4)]
        values=np.asarray([[(cache.get(self.generation,2*f,m).voltage-cache.get(self.generation,2*f+1,m).voltage-zero[f])*self.gain
            for f in range(4)] for m in masks],dtype=np.float32)
        self.emit('measured_currents',stage=stage,phase=phase,generation=self.generation,masks=masks,
            gain=self.gain,zero_difference_v=zero,currents=values.tolist(),metadata=metadata,
            queries=[dict(wl=wl,mask=m,task_id=r.task_id,adc=r.adc,voltage_v=r.voltage) for (wl,m),r in sorted(cache.values.items())])
        return values

    def update(self,conductance: NDArray[np.float64],target: NDArray[np.float32],stage: int,image: int,tick: int) -> None:
        raw=(conductance[:,:,0]-conductance[:,:,1])/self.span
        self.learning_stats['out_of_range']+=int(np.count_nonzero((raw<0)|(raw>1)))
        for f in range(4):
            for bl in range(25):
                # Desired G ограничен nominaltarget[0,1], но rawG/forward НЕ
                # подменяются этим target. Дискретный один импульс может
                # промахнуться; фактический postR сохраняется, retry запрещён.
                delta=(float(target[f,bl])-float(raw[f,bl]))*self.span
                if abs(delta)<=DEADBAND*self.span:
                    self.learning_stats['deadband']+=1
                    continue
                sign=1 if delta>0 else 0
                self.learning_stats['desired_set' if sign else 'desired_reset']+=1
                before_pulses=self.pulses
                post=self.pulse(2*f,bl,sign,1.7 if sign else 1.4,.00001,
                    phase='learning',stage=stage,image=image,tick=tick,filter=f,
                    raw_nominal=float(raw[f,bl]),target_nominal=float(target[f,bl]),delta_target_g_s=delta)
                admitted=self.pulses>before_pulses;change=post-float(conductance[f,bl,0])
                self.learning_stats[('admitted_set' if sign else 'admitted_reset') if admitted else 'guard_skipped']+=1
                if admitted:self.learning_stats['delta_zero' if change==0 else 'delta_expected' if change*(1 if sign else -1)>0 else 'delta_opposite']+=1
                self.emit('learning_readback',stage=stage,image=image,tick=tick,filter=f,bl=bl,sign=sign,admitted=admitted,
                    before_g_s=float(conductance[f,bl,0]),after_g_s=post,delta_g_s=change,
                    target_residual_g_s=float(conductance[f,bl,1])+self.span*float(target[f,bl])-post)
