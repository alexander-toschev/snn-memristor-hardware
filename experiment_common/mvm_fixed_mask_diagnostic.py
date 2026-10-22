"""Постоянно положительная маска: измеренные конечные разности mode10.

Каждое число отклика приходит через публичный Connector. Сумма V/R нужна
только для ограничения тока. Сумма singleton-разностей — проверяемая гипотеза
аддитивности на heldout-комбинациях, а не замена аппаратного MVM.
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

BIAS=.025
DELTAS=(.025,.05)
TARGETS=(14,16,20,30,31)
CONTEXTS=(('upper16_plus_BL14',(14,*range(16,32))),('all32',tuple(range(32))))
HELDOUT=((16,20),(20,30,31),(14,16,20,30,31))
MAX_CALLS=4290
SPEC: dict[str,object]=dict(schema='mvm-fixed-positive-mask-v1',bias_v=BIAS,deltas_v=list(DELTAS),
    contexts=[dict(name=name,bls=list(bls)) for name,bls in CONTEXTS],wls=[0,7],targets=list(TARGETS),
    heldout=[list(mask) for mask in HELDOUT],heldout_delta_v=.025,repeats=3,
    maximum_impacts=MAX_CALLS,maximum_pulses=0,maximum_wall_seconds=600,
    bracket=['freshR_all_active','positive_baseline','perturbation_same_mask','positive_baseline'],
    current_limit_a=.005,predicted_output_limit_v=3.3,measured_output_limit_v=4.,
    zero_vector_used=False,selection_or_tuning_on_heldout=False,calibrated_forward_qualified=False,
    exploratory_screening=dict(singleton_mean_gt_max_adc_5_or_sd_times_5=True,
        scaled_delta_error_limit='max(5 ADC, 0.1*abs(predicted))',
        heldout_mean_residual_limit='max(5 ADC, 0.1*abs(predicted))',
        maximum_baseline_absolute_drift_adc=5,
        engineering_screening_only=True,all_targets_required_for_further_exploration=False),
    private_runtime_status='private_runtime_not_independently_reverified')


class Collector:
    def __init__(self,connector: Connector,converters: Converters,writer: Writer,source: Source) -> None:
        self.connector: Connector=connector
        self.cv,self.writer,self.source=converters,writer,source
        self.calls=0
        self.started=time.monotonic()
        self.checkpoints: list[dict[str,object]]=[]

    def emit(self,event: str,**fields: object) -> None:
        self.writer.write(dict(event=event,source=self.source,monotonic_s=time.monotonic(),**fields))

    def call(self,command: dict[str,int|float|list[float]],**metadata: object) -> Reading:
        if self.calls>=MAX_CALLS or time.monotonic()-self.started>=600:
            raise TimeoutError('Fixed-mask finite budget')
        wl=command.get('wl')
        mode=command.get('mode_flag')
        if type(wl) is not int or wl not in (0,7):
            raise ValueError('Unplanned WL')
        if mode==7:
            bl=command.get('bl')
            if type(bl) is not int or not isinstance(bl,int) or not 0<=bl<32 or command.get('vol')!=0. or command.get('sign')!=0 or command.get('pulse_width')!=0.:
                raise ValueError('Only mode7 read permitted')
        elif mode==10:
            values=command.get('vol')
            if not isinstance(values,list) or len(values)!=32 or any(v not in (0.,.025,.05,.075) for v in values):
                raise ValueError('Unplanned full voltage vector')
            active=tuple(b for b,v in enumerate(values) if v>0)
            if active not in [bls for _,bls in CONTEXTS]:
                raise ValueError('Mask must stay exactly positive within one declared context')
        else:
            raise ValueError('Only mode7 read/mode10')
        self.calls+=1
        ticket={**command,'id':self.calls if mode==7 else 0}
        self.emit('request',task_id=self.calls,command_si=ticket,metadata=metadata)
        values=ticket['vol']
        copied={**ticket,'vol':list(values) if isinstance(values,list) else values}
        with contextlib.redirect_stdout(DriverOutput(self,'stdout')),contextlib.redirect_stderr(DriverOutput(self,'stderr')):
            response=self.connector.impact(copied)
        self.emit('result',task_id=self.calls,command_si=ticket,native_result=safe(response))
        if not isinstance(response,(list,tuple)) or len(response)!=3:
            raise ValueError('Invalid native response shape')
        resistance,echo,adc=response
        if type(echo) is not int or echo!=ticket['id'] or type(adc) is not int or not 0<=adc<=16382:
            raise ValueError('Invalid ID/ADC')
        r=number(resistance)
        if mode==7 and (r<=1e-8 or adc<20):
            raise ValueError('Invalid mode7 resistance')
        return Reading(task_id=self.calls,resistance_ohm=r,adc=adc)

    def diagnose(self,progress: Callable[[str],None]) -> None:
        singleton: list[tuple[tuple[int,...],float]]=[((bl,),delta) for bl in TARGETS for delta in DELTAS]
        heldout: list[tuple[tuple[int,...],float]]=[(mask,.025) for mask in HELDOUT]
        for context,active in CONTEXTS:
            baseline=[BIAS if b in active else 0. for b in range(32)]
            baseline_codes=[self.cv.convert_volt_to_dac(12,5,v) for v in baseline]
            for wl in (0,7):
                for phase,conditions in [('singleton',singleton),('heldout',heldout)]:
                    for repeat in range(3):
                        for targets,delta in conditions:
                            vector=[round(BIAS+delta,3) if b in targets else baseline[b] for b in range(32)]
                            codes=[self.cv.convert_volt_to_dac(12,5,v) for v in vector]
                            decoded=[self.cv.convert_dac_to_volt(12,5,c) for c in codes]
                            meta: dict[str,object]=dict(context=context,cell_wl=wl,phase=phase,repeat=repeat,targets=list(targets),delta_requested_v=delta)
                            guards=[self.call(dict(mode_flag=7,wl=wl,bl=b,vol=0.,sign=0,pulse_width=0.),active_bl=b,role='guard',**meta) for b in active]
                            current=sum(max(vector[b],decoded[b])/r['resistance_ohm'] for b,r in zip(active,guards))
                            if current>.005 or 330*current>=3.3:
                                raise ValueError('Fresh summed-current/headroom guard')
                            # Все три MVM имеют ОДИН positive bitmask. Mode7 перед
                            # bracket может переключать тракт; baseline измерен после него.
                            before=self.call(dict(mode_flag=10,wl=wl,vol=baseline),role='baseline_before',**meta)
                            signal=self.call(dict(mode_flag=10,wl=wl,vol=vector),role='perturbation',**meta)
                            after=self.call(dict(mode_flag=10,wl=wl,vol=baseline),role='baseline_after',**meta)
                            vs=[self.cv.convert_adc_to_volt(1,14,5,r['adc']) for r in (before,signal,after)]
                            if any(not math.isfinite(v) or v>4. for v in vs):
                                raise ValueError('Measured ADC headroom')
                            row: dict[str,object]=dict(**meta,wl=wl,active_bls=list(active),guards=guards,
                                target_dac_increment=codes[targets[0]]-baseline_codes[targets[0]],
                                baseline_before=before,signal=signal,baseline_after=after,
                                baseline_requested_v=baseline,perturbation_requested_v=vector,
                                baseline_dac_codes=baseline_codes,perturbation_dac_codes=codes,
                                baseline_native_decoded_v=[self.cv.convert_dac_to_volt(12,5,c) for c in baseline_codes],
                                perturbation_native_decoded_v=decoded,guard_sum_current_a=current,
                                delta_adc=signal['adc']-(before['adc']+after['adc'])/2,
                                delta_v=vs[1]-(vs[0]+vs[2])/2,
                                baseline_drift_adc=after['adc']-before['adc'])
                            self.checkpoints.append(row)
                            self.emit('fixed_mask_bracket',**row)
                        self.writer.checkpoint(context=context,wl=wl,phase=phase,repeat=repeat,impacts=self.calls)
                    progress(f'{context} WL{wl} {phase}: {self.calls}/{MAX_CALLS}; 0 writes')


def summarize(rows: list[dict[str,object]]) -> list[dict[str,object]]:
    reports: list[dict[str,object]]=[]
    for context,_ in CONTEXTS:
        for wl in (0,7):
            subset=[r for r in rows if r['context']==context and r['wl']==wl]
            fitted: dict[tuple[int,float],float]={}
            increments: dict[tuple[int,float],float]={}
            singles: list[dict[str,object]]=[]
            for bl in TARGETS:
                for delta in DELTAS:
                    found=[r for r in subset if r['phase']=='singleton' and r['targets']==[bl] and r['delta_requested_v']==delta]
                    values=[number(r['delta_adc']) for r in found]
                    if len(values)==3:
                        fitted[bl,delta]=statistics.mean(values)
                        increments[bl,delta]=number(found[0]['target_dac_increment'])
                    mean=statistics.mean(values) if values else None
                    sd=statistics.stdev(values) if len(values)>1 else None
                    singles.append(dict(bl=bl,delta_requested_v=delta,count=len(values),measured_delta_adc=values,
                        mean_delta_adc=mean,sd_delta_adc=sd,
                        target_dac_increment=increments.get((bl,delta)),
                        resolvable_screen_pass=len(values)==3 and mean is not None and sd is not None and mean>max(5.,5*sd),
                        baseline_drift_adc=[r['baseline_drift_adc'] for r in found]))
            scaled: list[dict[str,object]]=[]
            for bl in TARGETS:
                if (bl,.025) in fitted and (bl,.05) in fitted and increments[bl,.025]>0:
                    ratio=increments[bl,.05]/increments[bl,.025]
                    predicted=fitted[bl,.025]*ratio
                    error=fitted[bl,.05]-predicted
                    scaled.append(dict(bl=bl,actual_dac_increment_ratio=ratio,predicted_delta_adc=predicted,
                        measured_delta_adc=fitted[bl,.05],error_adc=error,absolute_error_adc=abs(error),
                        screen_pass=abs(error)<=max(5.,.1*abs(predicted))))
            held: list[dict[str,object]]=[]
            for mask in HELDOUT:
                found=[r for r in subset if r['phase']=='heldout' and r['targets']==list(mask)]
                values=[number(r['delta_adc']) for r in found]
                prediction=sum(fitted[b,.025] for b in mask) if all((b,.025) in fitted for b in mask) else None
                residual=[v-prediction for v in values] if prediction is not None else []
                mean_residual=statistics.mean(residual) if residual else None
                denominator=max(abs(prediction),1.) if prediction is not None else None
                held.append(dict(targets=list(mask),count=len(values),measured_delta_adc=values,
                    predicted_sum_of_fit_singleton_deltas_adc=prediction,residual_adc=residual,
                    absolute_residual_adc=[abs(v) for v in residual],mean_residual_adc=mean_residual,
                    relative_residual_denominator_adc=denominator,
                    mean_relative_residual=mean_residual/denominator if mean_residual is not None and denominator is not None else None,
                    screen_pass=len(values)==3 and mean_residual is not None and prediction is not None and abs(mean_residual)<=max(5.,.1*abs(prediction)),
                    baseline_drift_adc=[r['baseline_drift_adc'] for r in found]))
            drift=max((abs(number(r['baseline_drift_adc'])) for r in subset),default=None)
            reports.append(dict(context=context,wl=wl,singleton=singles,scaled_delta_comparison=scaled,heldout=held,
                maximum_abs_baseline_drift_adc=drift,baseline_drift_screen_pass=len(subset)==39 and drift is not None and drift<=5,
                heldout_used_for_fit=False,qualified=False))
    return reports


def render(rows: list[dict[str,object]],summary: dict[str,object]) -> str:
    body=''
    for row in rows:
        body+='<h2>'+html.escape(str(row['context']))+' WL'+str(row['wl'])+'</h2><pre>'+html.escape(json.dumps(row,ensure_ascii=False,indent=2))+'</pre>'
    return ('<!doctype html><html lang="ru"><meta charset="utf-8"><title>Fixed-mask diagnostic</title>'
        '<style>body{font:16px sans-serif;max-width:1100px;margin:auto}pre{white-space:pre-wrap;background:#eee;padding:12px}</style>'
        '<h1>Постоянная положительная маска: измеренные приращения</h1><p>Baseline0,025 В; perturbation0,05/0,075 В. '
        'Heldout residual проверяет гипотезу аддитивности, не заменяет MVM расчётом. Это не готовая калибровка/исправление.</p>'
        +body+'<pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre></html>')
