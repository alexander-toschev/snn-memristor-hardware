"""Resident64x4: two physical32x4 planes, one LIF step after their measured sum."""
from __future__ import annotations

import contextlib
import math
import time
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Unpack, cast

if TYPE_CHECKING:
    from manager.board import Connector
from experiments.mnist_conductance_20261019.core import Collector as Physical,DriverLog,safe,GAINS
from experiments.mnist64_20261020 import inputs,logic
from experiments.mnist64_20261020.types import (
    Calibration, Checkpoint, Converters, Fixture, ImageMeasurement, ImageOutcome,
    InputImage, Limits, NativeCommand, NativeReading, NeuronOutcome, Parent,
    Progress, ProtocolSpec, PulseDecision, PulseMetadata, RawWriter, Scope,
    Signal, Source, Summary,
)

SPEC: ProtocolSpec = dict(schema='mnist-resident64x4-stdp-and-frozen-accuracy-v1',physical_shape=[32,8],logical_shape=[64,4],
    pooling='integer8x8,28px boundaries0,3,7,10,14,17,21,24,28',timesteps=32,amplitude_v=.05,maximum_mask=4,
    mapping='WL=neuron+4*(feature//32);BL=feature%32',tile_sum_before_lif=True,separate_control_bank=False,
    lif_leak=.5,lif_threshold=1.,eligibility_features=64,trace_decay=math.exp(-.25),depression=1.05,
    eligibility_threshold=.25,updates='maxabs64feature,lowerfeature tie;at most1/logicalneuron/image',
    gain_grid=list(GAINS),gain_fit='fit32 only;each neuron8..64posts,>=4activeimages,max8/image,no empty posts',
    gain_selection='min(abs(totalposts-32),gain) separately per4logicalneurons',
    preparation_signs=[0]*4+[1]*4,reset_v=1.4,set_v=1.7,reset_current_a=.04,set_current_a=.002,
    simulator_pulse_width_s=.0001,hardware_pulse_width_s=0.,hardware_physical_duration_s=None,
    train_upper_impacts=129024,train_max_pulses=2560,accuracy_upper_impacts=128064,accuracy_max_pulses=0,
    train_checkpoints=['initial','after_preparation','final'],accuracy_checkpoints=['start','end'],
    retention_seconds=60.,readout='same-entry4x32raster,ridge1,train-onlystandardization,unpenalizedbias',
    accuracy='officialtest0..99; silent/tiedabstaincountsincorrect,denominator100',labels_in_acquisition=False)


class Collector(Physical):
    def __init__(self, connector: Connector, converters: Converters, writer: RawWriter,
                 source: Source, fixture: Fixture, gain: list[float] | None = None,
                 parent: Parent | None = None, progress: Progress | None = None) -> None:
        if source not in ('hardware','simulator') or connector.cb_type!=('real' if source=='hardware' else 'simulator'):
            raise ValueError('Public Connector/source mismatch')
        self.connector: Connector = connector
        self.converters: Converters = converters
        self.writer: RawWriter = writer
        self.source: Source = source
        self.images: dict[int, InputImage] = inputs.verify(fixture)
        self.scope: Scope = fixture['scope']
        self.limits: Limits = inputs.budget(fixture)
        self.gain: list[float] | None = gain
        self.parent: Parent | None = parent
        self.reference: None = None
        self.calibration: Calibration | None = None
        if self.scope=='accuracy' and (not isinstance(gain,list) or len(gain)!=4 or any(g not in GAINS for g in gain)):
            raise ValueError('Frozen4gain parent required')
        self.calls: int = 0
        self.pulses: int = 0
        self.decisions: int = 0
        self.ordinal: int = 0
        self.outcomes: list[ImageOutcome] = []
        self.updates: list[PulseDecision] = []
        self.checkpoints: list[Checkpoint] = []
        self.image_cell_participation: list[list[int]] = [[0]*32 for _ in range(8)]
        self.input_event_counts: dict[str, list[int]] = {p:[0]*64 for p in inputs.groups(self.scope)}
        self.monotonic: Callable[[], float] = time.monotonic
        self.sleep: Callable[[float], None] = time.sleep
        self.progress: Progress = progress or (lambda text:None)

    def call(self, command: NativeCommand, phase: str, **meta: object) -> NativeReading:
        # Explicit new cap. Does not call a historical method with global SPEC.
        if self.calls>=self.limits['maximum_impacts']:raise ValueError('New64x4 impact cap')
        if self.scope=='accuracy' and command['mode_flag']==7 and command['vol']!=0:raise ValueError('Frozen accuracy write forbidden')
        self.calls+=1
        task: NativeCommand = {**command,'id':self.calls if command['mode_flag']==7 else 0}
        fields=dict(task_id=self.calls,command_si=task,phase=phase,**meta)
        self.emit('request',**fields);started=self.monotonic()
        try:
            with contextlib.redirect_stdout(DriverLog(self,'stdout')),contextlib.redirect_stderr(DriverLog(self,'stderr')):
                voltage = task['vol']
                submitted: NativeCommand = {**task,'vol':list(voltage) if isinstance(voltage,list) else voltage}
                result=self.connector.impact(submitted)
            self.emit('result',**fields,native_result=safe(result),latency_s=self.monotonic()-started)
            if not isinstance(result,(list,tuple)) or len(result)!=3:raise ValueError('Invalid native tuple')
            r,echo,adc=result
            if type(echo)is not int or echo!=task['id'] or type(adc)is not int:raise ValueError('Invalid native ID/ADC')
            if task['mode_flag']==7:
                if type(r)not in (float,int) or not math.isfinite(r) or r<=1e-8 or not 20<=adc<=16382:raise ValueError('Invalid native R')
            elif task['mode_flag']!=10 or not 0<=adc<=16382:raise ValueError('Invalid native MVM')
            return dict(r=r,adc=adc,task_id=self.calls)
        except (Exception,KeyboardInterrupt) as exc:
            self.emit('call_error',task_id=self.calls,error=repr(exc));raise

    def pulse(self, wl: int, bl: int, sign: int, phase: str, **meta: Unpack[PulseMetadata]) -> None:
        if self.scope!='train' or self.decisions>=2560 or phase not in ('preparation','training_update'):raise ValueError('Write scope/cap')
        level=1.7 if sign else 1.4
        guard=self.read_r(wl,bl,'pulse_guard',intended_phase=phase,**meta)
        self.decisions+=1;admitted=level/guard['r']<=(.002 if sign else .04)
        # Unpack[PulseMetadata] restricts the extra fields; this exact-shape cast
        # bridges mypy's open-TypedDict unpack limitation, without Any.
        payload: dict[str, object] = dict(wl=wl,bl=bl,sign=sign,voltage_v=level,phase=phase,
                                          decision=self.decisions,guard=guard,admitted=admitted,**meta)
        result = cast(PulseDecision, payload)
        self.emit('pulse_decision',**result)
        if admitted:
            self.pulses+=1
            result['pulse']=self.call(dict(mode_flag=7,wl=wl,bl=bl,vol=level,sign=sign,
                pulse_width=.0001 if self.source=='simulator' else 0.),phase,**meta)
            result['post']=self.read_r(wl,bl,'pulse_post_r',intended_phase=phase,**meta)
            result['delta_g_s']=1/result['post']['r']-1/guard['r']
        self.updates.append(result);self.emit('pulse_complete',**result)
        self.writer.checkpoint(reason='complete_update',impacts=self.calls,pulses=self.pulses)

    def image(self, phase: str, index: int) -> ImageMeasurement:
        if (phase=='fit')!=(self.gain is None):raise ValueError('Image/gain freeze order')
        row=self.images[index];schedule=row['schedule'];ordinal=self.ordinal;self.ordinal+=1
        meta=dict(image_phase=phase,image_ordinal=ordinal,train_index=index)
        self.emit('image_begin',phase=phase,ordinal=ordinal,train_index=index,image_sha256=row['image_sha256'],
                  schedule_sha256=row['schedule_sha256'],schedule=schedule)
        order=list(range(8)) if ordinal%2==0 else list(reversed(range(8)))
        before={wl:self.vector(wl,[],'image_zero_before',**meta) for wl in order}
        signals: list[Signal] = []
        for tick,mask in enumerate(schedule):
            for f in mask:self.input_event_counts[phase][f]+=1
            for tile in (0,1):
                active=[f%32 for f in mask if f//32==tile]
                for micro,start in enumerate(range(0,len(active),4)):
                    chunk=active[start:start+4]
                    neuron_order: Iterable[int] = range(4) if (ordinal+tick+tile+micro)%2==0 else reversed(range(4))
                    for neuron in neuron_order:
                        wl=4*tile+neuron
                        signal=self.vector(wl,chunk,'image_signal',t=tick,tile=tile,micro=micro,neuron=neuron,**meta)
                        signals.append(dict(wl=wl,neuron=neuron,tile=tile,t=tick,micro=micro,mask=chunk,**signal))
                        for bl in chunk:self.image_cell_participation[wl][bl]+=1
        after={wl:self.vector(wl,[],'image_zero_after',**meta) for wl in order[::-1]}
        volts=[[0.]*32 for _ in range(4)]
        for s in signals:
            # Sum both tile planes and all microvectors before ONE leak/threshold.
            volts[s['neuron']][s['t']]+=s['voltage_v']-(before[s['wl']]['voltage_v']+after[s['wl']]['voltage_v'])/2
        result: ImageMeasurement = dict(phase=phase,ordinal=ordinal,train_index=index,schedule=schedule,voltages=volts,
            zero_before=before,zero_after=after,signals=signals,zero_drift_v=[after[w]['voltage_v']-before[w]['voltage_v'] for w in range(8)])
        self.emit('image_measurements_complete',**result)
        if phase!='fit':
            assert self.gain is not None  # Already enforced by the image/gain guard above.
            neurons: list[NeuronOutcome] = []
            for n in range(4):
                steps,choice=logic.trajectory(schedule,volts[n],self.gain[n])
                for step in steps:self.emit('lif_step',image_ordinal=ordinal,image_phase=phase,train_index=index,neuron=n,**step)
                neurons.append(dict(neuron=n,posts=[s['post'] for s in steps],eligibility=steps[-1]['eligibility'],choice=choice))
            outcome: ImageOutcome = dict(phase=phase,ordinal=ordinal,train_index=index,neurons=neurons,zero_drift_v=result['zero_drift_v'])
            self.outcomes.append(outcome);self.emit('image_feedback_complete',**outcome)
            if phase=='train':
                for n in (range(4) if index%2==0 else reversed(range(4))):
                    choice=neurons[n]['choice']
                    if choice is None:self.emit('no_training_update',image_ordinal=ordinal,train_index=index,neuron=n)
                    else:
                        wl,bl=logic.address(n,choice['feature'])
                        self.pulse(wl,bl,choice['sign'],'training_update',image_ordinal=ordinal,train_index=index,
                            neuron=n,feature=choice['feature'],eligibility=choice['eligibility'])
        self.emit('image_complete',phase=phase,ordinal=ordinal,train_index=index,impacts=self.calls,pulses=self.pulses)
        self.writer.checkpoint(reason='complete_image',ordinal=ordinal,phase=phase,impacts=self.calls,pulses=self.pulses)
        indices=inputs.groups(self.scope)[phase];done=indices.index(index)+1
        if done%16==0 or done==len(indices):
            label={'train':'STDP','readout_train':'Readout train collection','official_test':'Test response collection'}.get(phase,phase)
            self.progress('%s %d/%d; impacts=%d pulses=%d'%(label,done,len(indices),self.calls,self.pulses))
        return result

    def summary(self, status: str) -> Summary:
        return dict(status=status,scope=self.scope,impacts=self.calls,pulses=self.pulses,decisions=self.decisions,
            measured_images=self.ordinal,gain=self.gain,calibration=self.calibration,outcomes=self.outcomes,
            updates=self.updates,checkpoints=self.checkpoints,image_cell_participation=self.image_cell_participation,
            input_event_counts=self.input_event_counts,separate_control_bank=False,classification_accuracy_measured=False)

    def run(self) -> Summary:
        status='failed';error=None;opened=False
        try:
            self.emit('session_begin',protocol=SPEC,scope=self.scope,parent=self.parent,gain=self.gain,limits=self.limits)
            opened=True;result=self.connector.open_port();self.emit('port_open',result=safe(result))
            if not result:raise ValueError('Public open failed')
            self.checkpoint('initial' if self.scope=='train' else 'start')
            if self.scope=='train':
                for step,sign in enumerate(SPEC['preparation_signs']):
                    for bl in range(32):
                        for wl in (range(8) if (step+bl)%2==0 else reversed(range(8))):
                            self.pulse(wl,bl,sign,'preparation',preparation_step=step)
                self.checkpoint('after_preparation')
                fit=[self.image('fit',i) for i in inputs.TRAIN_GROUPS['fit']]
                self.calibration=logic.fit(fit,self.emit);self.gain=self.calibration['gains']
                self.emit('gain_frozen',last_native_task_id=self.calls,**self.calibration)
                if self.gain is None:
                    self.emit('calibration_stop',reason='No individual fixed-grid eligible gain; no probes/training')
                    status='uninformative_calibration'
                else:
                    for phase in ('preprobe','train','postprobe','retentionprobe'):
                        if phase=='retentionprobe':
                            self.emit('retention_begin',seconds=60.);start=self.monotonic();self.sleep(60.)
                            self.emit('retention_end',elapsed_s=self.monotonic()-start)
                        for i in inputs.TRAIN_GROUPS[phase]:self.image(phase,i)
                    self.checkpoint('final');status='completed'
            else:
                for phase,indices in inputs.ACCURACY_GROUPS.items():
                    for i in indices:self.image(phase,i)
                self.checkpoint('end');status='completed'
        except (Exception,KeyboardInterrupt) as exc:error=repr(exc);raise
        finally:
            try:
                if opened:
                    result=self.connector.close_port();self.emit('port_close',result=safe(result))
                    if not result:status='failed';raise ValueError('Public close failed')
            finally:
                self.emit('session_end',status=status,error=error,impacts=self.calls,pulses=self.pulses,decisions=self.decisions)
                self.writer.finish()
        return self.summary(status)
