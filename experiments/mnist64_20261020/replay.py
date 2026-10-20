"""Independent64-trace math plus strict sequential raw command replay."""
import math
from collections import Counter
import numpy as np
from tools.audit_mnist_conductance import Replay as PhysicalReplay,require,same,exact_command,GAINS
from experiments.mnist64_20261020 import inputs


def address(neuron,feature):return neuron+4*(feature//32),feature%32


def trajectory(schedule,voltages,gain):
    require(len(schedule)==32 and len(voltages)==32 and gain in GAINS,'LIF inputs')
    a=np.zeros(64);e=np.zeros(64);b=v=0.;steps=[];decay=math.exp(-.25)
    for t in range(32):
        pre=np.zeros(64,dtype=int);pre[schedule[t]]=1
        before=v;u=.5*v+gain*voltages[t];post=int(u>=1.);v=u if not post else 0.
        a*=decay;b*=decay;delta=a*post-1.05*b*pre;e+=delta;a+=pre;b+=post
        steps.append(dict(t=t,pre=pre.tolist(),voltage_v=voltages[t],gain=gain,u_before=before,u_unreset=u,
            u_after=v,post=post,pre_trace=a.tolist(),post_trace=b,eligibility_increment=delta.tolist(),eligibility=e.tolist()))
    feature=int(np.argmax(np.abs(e)))
    choice=None if abs(e[feature])<.25 else dict(feature=feature,eligibility=float(e[feature]),sign=int(e[feature]>0))
    return steps,choice


class Replay(PhysicalReplay):
    def __init__(self,records,fixture,source,gain=None,parent=None):
        self.rows=iter(records);self.source=source;self.reference=None;self.parent=parent
        self.images=inputs.verify(fixture);self.scope=fixture['scope'];self.limits=inputs.budget(fixture)
        self.gain=gain;self.calibration=None
        self.calls=self.pulses=self.decisions=self.ordinal=self.records=0;self.last_monotonic=None
        self.outcomes=[];self.updates=[];self.checkpoints=[]
        self.image_cell_participation=[[0]*32 for _ in range(8)]
        self.input_event_counts={p:[0]*64 for p in inputs.groups(self.scope)}
        self.mode_counts=Counter();self.pulse_counts=Counter();self.max_mvm_current=0.

    def read_r(self,wl,bl,phase,**meta):return self.read(wl,bl,phase,**meta)

    def call(self, command, phase, **metadata):
        self.calls += 1
        require(self.calls <= self.limits['maximum_impacts'], 'New64 exact impact cap')
        task = dict(command, id=self.calls if command['mode_flag']==7 else 0)
        fields = dict(task_id=self.calls, command_si=task, phase=phase, **metadata)
        request = self.expect('request', **fields)
        exact_command(request['command_si'],task)
        require(type(request['task_id']) is int,'Request task ID type differs')
        result = self.take()
        while result['event'] == 'driver_output':
            require(set(self.payload(result)) == {'task_id','stream','text'} and result['task_id']==self.calls and result['stream'] in ('stdout','stderr') and isinstance(result['text'],str), 'Unbound driver output')
            result = self.take()
        require(result['event']=='result', 'Native call lacks result')
        p = self.payload(result)
        require(set(p)==set(fields)|{'native_result','latency_s'}, 'Native result fields differ')
        same({key:p[key] for key in fields},fields,'result request ownership')
        exact_command(p['command_si'],task)
        require(type(p['task_id']) is int,'Result task ID type differs')
        latency=p['latency_s']
        require(type(latency) in (int,float) and math.isfinite(latency) and 0 <= latency <= result['monotonic_s']-request['monotonic_s']+1e-6, 'Invalid native latency')
        native=p['native_result']
        require(isinstance(native,list) and len(native)==3, 'Invalid native tuple')
        resistance,echo,adc=native
        require(type(echo) is int and echo==task['id'] and type(adc) is int, 'Native echo/ADC type differs')
        if command['mode_flag']==7:
            require(type(resistance) in (float,int) and math.isfinite(resistance) and resistance>1e-8 and 20<=adc<=16382, 'Invalid native R/ADC')
        else:
            require(command['mode_flag']==10 and 0<=adc<=16382, 'Invalid MVM ADC')
        self.mode_counts[command['mode_flag']] += 1
        return dict(r=resistance,adc=adc,task_id=self.calls)

    def pulse(self,wl,bl,sign,phase,**meta):
        if self.scope!='train' or self.decisions>=2560 or phase not in ('preparation','training_update'):raise ValueError('Write scope/cap')
        level=1.7 if sign else 1.4
        guard=self.read_r(wl,bl,'pulse_guard',intended_phase=phase,**meta)
        self.decisions+=1;admitted=level/guard['r']<=(.002 if sign else .04)
        result=dict(wl=wl,bl=bl,sign=sign,voltage_v=level,phase=phase,decision=self.decisions,guard=guard,admitted=admitted,**meta)
        self.expect('pulse_decision',**result)
        if admitted:
            self.pulses+=1
            result['pulse']=self.call(dict(mode_flag=7,wl=wl,bl=bl,vol=level,sign=sign,
                pulse_width=.0001 if self.source=='simulator' else 0.),phase,**meta)
            result['post']=self.read_r(wl,bl,'pulse_post_r',intended_phase=phase,**meta)
            result['delta_g_s']=1/result['post']['r']-1/guard['r']
        self.updates.append(result);self.expect('pulse_complete',**result)

    def image(self,phase,index):
        if (phase=='fit')!=(self.gain is None):raise ValueError('Image/gain freeze order')
        row=self.images[index];schedule=row['schedule'];ordinal=self.ordinal;self.ordinal+=1
        meta=dict(image_phase=phase,image_ordinal=ordinal,train_index=index)
        self.expect('image_begin',phase=phase,ordinal=ordinal,train_index=index,image_sha256=row['image_sha256'],
                  schedule_sha256=row['schedule_sha256'],schedule=schedule)
        order=list(range(8)) if ordinal%2==0 else list(reversed(range(8)))
        before={wl:self.vector(wl,[],'image_zero_before',**meta) for wl in order};signals=[]
        for tick,mask in enumerate(schedule):
            for f in mask:self.input_event_counts[phase][f]+=1
            for tile in (0,1):
                active=[f%32 for f in mask if f//32==tile]
                for micro,start in enumerate(range(0,len(active),4)):
                    chunk=active[start:start+4]
                    neurons=range(4) if (ordinal+tick+tile+micro)%2==0 else reversed(range(4))
                    for neuron in neurons:
                        wl=4*tile+neuron
                        result=self.vector(wl,chunk,'image_signal',t=tick,tile=tile,micro=micro,neuron=neuron,**meta)
                        signals.append(dict(wl=wl,neuron=neuron,tile=tile,t=tick,micro=micro,mask=chunk,**result))
                        for bl in chunk:self.image_cell_participation[wl][bl]+=1
        after={wl:self.vector(wl,[],'image_zero_after',**meta) for wl in order[::-1]}
        volts=[[0.]*32 for _ in range(4)]
        for s in signals:
            # Sum both tile planes and all microvectors before ONE leak/threshold.
            volts[s['neuron']][s['t']]+=s['voltage_v']-(before[s['wl']]['voltage_v']+after[s['wl']]['voltage_v'])/2
        result=dict(phase=phase,ordinal=ordinal,train_index=index,schedule=schedule,voltages=volts,
            zero_before=before,zero_after=after,signals=signals,zero_drift_v=[after[w]['voltage_v']-before[w]['voltage_v'] for w in range(8)])
        self.expect('image_measurements_complete',**result)
        if phase!='fit':
            neurons=[]
            for n in range(4):
                steps,choice=trajectory(schedule,volts[n],self.gain[n])
                for step in steps:self.expect('lif_step',image_ordinal=ordinal,image_phase=phase,train_index=index,neuron=n,**step)
                neurons.append(dict(neuron=n,posts=[s['post'] for s in steps],eligibility=steps[-1]['eligibility'],choice=choice))
            outcome=dict(phase=phase,ordinal=ordinal,train_index=index,neurons=neurons,zero_drift_v=result['zero_drift_v'])
            self.outcomes.append(outcome);self.expect('image_feedback_complete',**outcome)
            if phase=='train':
                for n in (range(4) if index%2==0 else reversed(range(4))):
                    choice=neurons[n]['choice']
                    if choice is None:self.expect('no_training_update',image_ordinal=ordinal,train_index=index,neuron=n)
                    else:
                        wl,bl=address(n,choice['feature'])
                        self.pulse(wl,bl,choice['sign'],'training_update',image_ordinal=ordinal,train_index=index,
                            neuron=n,feature=choice['feature'],eligibility=choice['eligibility'])
        self.expect('image_complete',phase=phase,ordinal=ordinal,train_index=index,impacts=self.calls,pulses=self.pulses)
        return result

    def summary(self,status):
        return dict(status=status,scope=self.scope,impacts=self.calls,pulses=self.pulses,decisions=self.decisions,
            measured_images=self.ordinal,gain=self.gain,calibration=self.calibration,outcomes=self.outcomes,
            updates=self.updates,checkpoints=self.checkpoints,image_cell_participation=self.image_cell_participation,
            input_event_counts=self.input_event_counts,separate_control_bank=False,classification_accuracy_measured=False)

    def fit(self,images):
        candidates=[]
        for gain in GAINS:
            per=[]
            for n in range(4):
                counts=[];empty=0
                for image in images:
                    steps,_=trajectory(image['schedule'],image['voltages'][n],gain)
                    for step in steps:self.expect('fit_candidate_lif_step',image_ordinal=image['ordinal'],neuron=n,**step)
                    counts.append(sum(s['post'] for s in steps))
                    empty+=sum(s['post'] for s in steps if not any(s['pre']))
                eligible=8<=sum(counts)<=64 and sum(c>0 for c in counts)>=4 and max(counts)<=8 and empty==0
                per.append(dict(neuron=n,post_counts=counts,total_posts=sum(counts),active_images=sum(c>0 for c in counts),
                                empty_input_posts=empty,eligible=eligible))
            c=dict(gain=gain,per_neuron=per);candidates.append(c);self.expect('gain_candidate',**c)
        gains=[]
        for n in range(4):
            options=[(c['gain'],c['per_neuron'][n]['total_posts']) for c in candidates if c['per_neuron'][n]['eligible']]
            if not options:break
            gains.append(min(options,key=lambda x:(abs(x[1]-32),x[0]))[0])
        self.calibration=dict(gains=gains if len(gains)==4 else None,candidates=candidates,
            selection='per logical neuron min(abs(totalfitposts-32),gain)',labels_used=False)
        self.gain=self.calibration['gains']
        self.expect('gain_frozen',last_native_task_id=self.calls,**self.calibration)

    def run(self,protocol):
        require(protocol['logical_shape']==[64,4] and protocol['tile_sum_before_lif'] is True and
                protocol['amplitude_v']==.05 and protocol['maximum_mask']==4 and
                protocol['train_max_pulses']==2560 and protocol['accuracy_max_pulses']==0 and
                protocol['separate_control_bank'] is False,'Wrong64protocol')
        self.expect('session_begin',protocol=protocol,scope=self.scope,parent=self.parent,gain=self.gain,limits=self.limits)
        require(bool(self.take('port_open')['result']),'Open not confirmed')
        self.checkpoint('initial' if self.scope=='train' else 'start')
        if self.scope=='train':
            for step,sign in enumerate([0]*4+[1]*4):
                for bl in range(32):
                    for wl in (range(8) if (step+bl)%2==0 else reversed(range(8))):
                        self.pulse(wl,bl,sign,'preparation',preparation_step=step)
            self.checkpoint('after_preparation')
            fit=[self.image('fit',i) for i in inputs.TRAIN_GROUPS['fit']]
            self.fit(fit)
            if self.gain is None:
                self.expect('calibration_stop',reason='No individual fixed-grid eligible gain; no probes/training')
                require(self.calls<=self.limits['noeligible_maximum_impacts'],'Noeligible cap')
                status='uninformative_calibration'
            else:
                for phase in ('preprobe','train','postprobe','retentionprobe'):
                    if phase=='retentionprobe':
                        begin=self.expect('retention_begin',seconds=60.)
                        end=self.take('retention_end')
                        require(end['elapsed_s']>=60. and end['monotonic_s']-begin['monotonic_s']>=end['elapsed_s'],'Retention elapsed differs')
                    for i in inputs.TRAIN_GROUPS[phase]:self.image(phase,i)
                self.checkpoint('final');status='completed'
        else:
            for phase,indices in inputs.ACCURACY_GROUPS.items():
                for i in indices:self.image(phase,i)
            self.checkpoint('end');status='completed'
        require(bool(self.take('port_close')['result']),'Close not confirmed')
        self.expect('session_end',status=status,error=None,impacts=self.calls,pulses=self.pulses,decisions=self.decisions)
        require(next(self.rows,None) is None,'Trailing raw events')
        expected=32 if status=='uninformative_calibration' else 208 if self.scope=='train' else 228
        require(self.ordinal==expected and self.pulses<=self.limits['maximum_pulses'],'Coverage/pulse cap')
        return self.summary(status)

