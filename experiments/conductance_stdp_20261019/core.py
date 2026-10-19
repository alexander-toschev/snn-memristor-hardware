"""Bounded STDP collector. All physical access uses an injected public Connector."""
import contextlib
from datetime import datetime, timezone
import io
import math
import time

BL=(6,8,9,14)
WL=(5,7)
SPEC=dict(schema='conductance-stdp-v1',bl=list(BL),wl=list(WL),amplitude_v=.05,
          reset_v=1.4,set_v=1.7,maximum_impacts=6712,maximum_pulses=128,
          training_windows=16,probe_windows_per_phase=4,timesteps=16,
          lif_leak=.5,lif_gain=100.,lif_threshold=1.,trace_tau=4.,
          depression_factor=1.05,eligibility_threshold=.25,retention_seconds=60.,
          simulator_pulse_width_s=.0001,hardware_submitted_width_s=0.,
          hardware_physical_width_s=None,seed=20261019)


def safe(value):
    if value is None or type(value) in (bool,int,str):return value
    if type(value) is float:return value if math.isfinite(value) else repr(value)
    if isinstance(value,(tuple,list)):return [safe(x) for x in value]
    return repr(value)


def utc():return datetime.now(timezone.utc).isoformat()


class DriverLog(io.TextIOBase):
    def __init__(self,owner,stream):self.owner,self.stream=owner,stream
    def write(self,value):
        if value:self.owner.emit('driver_output',task_id=self.owner.calls,stream=self.stream,text=value)
        return len(value)
    def flush(self):pass


class Neuron:
    def __init__(self):self.u=0.;self.a=[0.]*4;self.b=0.;self.e=[0.]*4
    def step(self,pre,voltage):
        if len(pre)!=4 or any(type(x) is not int or x not in (0,1) for x in pre) or not math.isfinite(voltage):
            raise ValueError('Invalid digital input')
        before=self.u;unreset=.5*before+100.*voltage;post=int(unreset>=1.)
        self.u=0. if post else unreset
        decay=math.exp(-.25)
        self.a=[v*decay for v in self.a];self.b*=decay
        increment=[v*post-1.05*self.b*s for v,s in zip(self.a,pre)]
        self.e=[v+d for v,d in zip(self.e,increment)]
        self.a=[v+s for v,s in zip(self.a,pre)];self.b+=post
        return dict(u_before=before,u_unreset=unreset,u_after=self.u,post=post,
                    pre=list(pre),voltage_v=voltage,eligibility=list(self.e),
                    eligibility_increment=increment,pre_trace=list(self.a),post_trace=self.b)


def timing(phase,index):
    if phase=='train':return (2,3,10,11) if index%2==0 else (10,11,2,3)
    if phase not in ('pretest','posttest','retention'):raise ValueError('Unknown window phase')
    return (1,4,9,12) if index%2==0 else (9,12,1,4)


class Collector:
    def __init__(self,connector,converters,writer,source,sleep=time.sleep,monotonic=time.monotonic):
        if source not in ('hardware','simulator') or connector.cb_type!=('real' if source=='hardware' else 'simulator'):
            raise ValueError('Connector/source mismatch')
        self.connector,self.converters,self.writer=connector,converters,writer
        self.source,self.sleep,self.monotonic=source,sleep,monotonic
        self.calls=self.pulses=self.decisions=0;self.checkpoints=[];self.windows=[];self.updates=[]

    def emit(self,event,**fields):
        self.writer.write(dict(event=event,source=self.source,utc=utc(),monotonic_s=self.monotonic(),**fields))

    def call(self,command,phase,**meta):
        if self.calls>=6712:raise ValueError('Impact budget exhausted')
        self.calls+=1;task={**command,'id':self.calls if command['mode_flag']==7 else 0}
        metadata=dict(task_id=self.calls,command_si=task,phase=phase,**meta)
        self.emit('request',**metadata);started=self.monotonic()
        try:
            with contextlib.redirect_stdout(DriverLog(self,'stdout')),contextlib.redirect_stderr(DriverLog(self,'stderr')):
                native=self.connector.impact({**task,'vol':list(task['vol']) if isinstance(task['vol'],list) else task['vol']})
            self.emit('result',**metadata,native_result=safe(native),latency_s=self.monotonic()-started)
            if not isinstance(native,(list,tuple)) or len(native)!=3:raise ValueError('Malformed native response')
            r,echo,adc=native
            if type(echo) is not int or echo!=task['id'] or type(adc) is not int:raise ValueError('Invalid echo/ADC type')
            if task['mode_flag']==7:
                if type(r) not in (int,float) or not math.isfinite(r) or r<=1e-8 or not 20<=adc<=16382:
                    raise ValueError('Invalid native R/ADC')
            elif not 0<=adc<=16382:raise ValueError('Invalid MVM ADC')
            return r,adc
        except (Exception,KeyboardInterrupt) as exc:
            self.emit('call_error',task_id=self.calls,error=type(exc).__name__+': '+str(exc));raise

    def read_r(self,wl,bl,phase,**meta):
        if wl not in WL or bl not in BL:raise ValueError('Out-of-scope cell')
        r,adc=self.call(dict(mode_flag=7,wl=wl,bl=bl,vol=0.,sign=0,pulse_width=0.),phase,**meta)
        return dict(r=r,adc=adc,task_id=self.calls)

    def pulse(self,wl,bl,sign,phase,**meta):
        if phase not in ('preparation','training_update') or sign not in (0,1):raise ValueError('Invalid pulse phase/sign')
        if phase=='training_update' and wl!=5:raise ValueError('Control cannot receive learning pulse')
        if self.decisions>=128:raise ValueError('Decision budget exhausted')
        self.decisions+=1
        level=1.7 if sign else 1.4
        sham=self.read_r(7,bl,'paired_sham',**meta) if phase=='training_update' else None
        guard=self.read_r(wl,bl,'pulse_guard',intended_phase=phase,**meta)
        admitted=level/guard['r']<=(.002 if sign else .04)
        event=dict(wl=wl,bl=bl,sign=sign,voltage_v=level,phase=phase,guard=guard,sham=sham,
                   admitted=admitted,decision=self.decisions,**meta)
        self.emit('pulse_decision',**event)
        if admitted:
            if self.pulses>=128:raise ValueError('Pulse budget exhausted')
            self.pulses+=1
            r,adc=self.call(dict(mode_flag=7,wl=wl,bl=bl,vol=level,sign=sign,
                pulse_width=.0001 if self.source=='simulator' else 0.),phase,**meta)
            event.update(pulse_task_id=self.calls,pulse_native_r=r,pulse_adc=adc)
        self.updates.append(event);self.emit('pulse_complete',**event)
        return event

    def vector(self,wl,mask,phase,**meta):
        if wl not in WL or len(mask)>1 or any(b not in BL for b in mask):raise ValueError('Invalid vector address')
        guards=[self.read_r(wl,b,'vector_guard',vector_phase=phase,**meta) for b in mask]
        dac=self.converters.convert_volt_to_dac(12,5,.05)
        amplitude=max(.05,self.converters.convert_dac_to_volt(12,5,dac))
        current=sum(amplitude/g['r'] for g in guards)
        if current>.005 or 330*current>=3.3:raise ValueError('MVM current/output guard')
        _,adc=self.call(dict(mode_flag=10,wl=wl,vol=[.05 if b in mask else 0. for b in range(32)]),phase,mask=list(mask),**meta)
        voltage=self.converters.convert_adc_to_volt(1,14,5,adc)
        if not math.isfinite(voltage) or voltage>4.:raise ValueError('Measured ADC headroom')
        return dict(task_id=self.calls,adc=adc,voltage_v=voltage,guards=guards)

    def bracket(self,wl,mask,phase,**meta):
        before=self.vector(wl,[],phase+'_zero_before',**meta)
        signal=self.vector(wl,mask,phase+'_signal',**meta)
        after=self.vector(wl,[],phase+'_zero_after',**meta)
        result=dict(wl=wl,mask=list(mask),before=before,signal=signal,after=after,
                    centered_v=signal['voltage_v']-(before['voltage_v']+after['voltage_v'])/2)
        self.emit('bracket',phase=phase,**result,**meta)
        return result

    def checkpoint(self,wl,bl,phase,**meta):
        reads=[self.read_r(wl,bl,'checkpoint_r',checkpoint_phase=phase,repeat=i,**meta) for i in range(3)]
        responses=[self.bracket(wl,[bl],'checkpoint_mvm',checkpoint_phase=phase,repeat=i,**meta) for i in range(3)]
        result=dict(wl=wl,bl=bl,phase=phase,reads=reads,responses=responses,
                    q=[330/x['r'] for x in reads],native_inverse_r=[1/x['r'] for x in reads],**meta)
        self.checkpoints.append(result);self.emit('checkpoint',**result)

    def all_cells(self,phase,**meta):
        for wl in WL:
            for bl in BL:self.checkpoint(wl,bl,phase,**meta)

    def window(self,phase,index):
        times=timing(phase,index);neurons={wl:Neuron() for wl in WL};steps=[]
        for t in range(16):
            pre=[int(t==v) for v in times];mask=[b for b,s in zip(BL,pre) if s]
            for wl in (WL if t%2==0 else WL[::-1]):
                measurement=self.bracket(wl,mask,'window',window_phase=phase,window_index=index,t=t)
                step=dict(wl=wl,t=t,measurement=measurement,**neurons[wl].step(pre,measurement['centered_v']))
                steps.append(step);self.emit('lif_step',window_phase=phase,window_index=index,**step)
        result=dict(phase=phase,index=index,times=list(times),steps=steps,eligibility={str(wl):neurons[wl].e for wl in WL})
        self.windows.append(result);self.emit('window_complete',**result)
        if phase=='train':
            for bl,e in zip(BL,neurons[5].e):
                if abs(e)>=.25:self.pulse(5,bl,int(e>=.25),'training_update',window_index=index,eligibility=e)
                else:self.emit('no_update',wl=5,bl=bl,window_index=index,eligibility=e)
        return result

    def run(self):
        status='failed';opened=False;error=None
        try:
            self.emit('session_begin',protocol=SPEC)
            opened=True  # Even a partial open that raises must attempt public close.
            result=self.connector.open_port();self.emit('port_open',result=safe(result))
            if not result:raise ValueError('Connector open failed')
            self.all_cells('initial')
            for decision in range(8):
                for j,bl in enumerate(BL):
                    sign=int((j<2)==(decision<4))
                    for wl in (WL if (decision+j)%2==0 else WL[::-1]):
                        self.pulse(wl,bl,sign,'preparation',preparation_step=decision)
                        self.checkpoint(wl,bl,'after_preparation_decision',preparation_step=decision)
            self.all_cells('after_preparation')
            for i in range(4):self.window('pretest',i)
            self.all_cells('baseline_before_training')
            for i in range(16):
                self.window('train',i);self.all_cells('after_training_window',window_index=i)
                print('STDP training window %d/16; impacts=%d pulses=%d'%(i+1,self.calls,self.pulses),flush=True)
            for i in range(4):self.window('posttest',i)
            self.all_cells('after_posttest')
            self.emit('retention_begin',seconds=60.)
            started=self.monotonic();self.sleep(60.);elapsed=self.monotonic()-started
            if elapsed<60.:raise ValueError('Retention wait ended early')
            self.emit('retention_end',elapsed_s=elapsed)
            for i in range(4):self.window('retention',i)
            self.all_cells('after_retention');status='completed'
        except (Exception,KeyboardInterrupt) as exc:
            error=type(exc).__name__+': '+str(exc)
            raise
        finally:
            try:
                if opened:
                    try:
                        closed=self.connector.close_port();self.emit('port_close',result=safe(closed))
                        if not closed:raise ValueError('Connector close failed')
                    except (Exception,KeyboardInterrupt) as exc:
                        status='failed';error=type(exc).__name__+': '+str(exc);raise
            finally:
                try:self.emit('session_end',status=status,error=error,impacts=self.calls,pulses=self.pulses,decisions=self.decisions)
                finally:self.writer.finish()
        counts={phase:dict(decisions=sum(x['phase']==phase for x in self.updates),
                           admitted=sum(x['phase']==phase and x['admitted'] for x in self.updates),
                           current_skipped=sum(x['phase']==phase and not x['admitted'] for x in self.updates))
                for phase in ('preparation','training_update')}
        lif={str(wl):dict(spikes=sum(s['post'] for w in self.windows for s in w['steps'] if s['wl']==wl),
                          windows_with_no_spikes=sum(not any(s['post'] for s in w['steps'] if s['wl']==wl) for w in self.windows)) for wl in WL}
        return dict(status=status,impacts=self.calls,pulses=self.pulses,decisions=self.decisions,
                    counts=counts,lif_summary=lif,uninformative_all_silent=all(v['spikes']==0 for v in lif.values()),
                    qualification='descriptive algorithmic STDP pilot, no accuracy/intrinsic-STDP claim',
                    checkpoints=self.checkpoints,windows=self.windows,updates=self.updates)
