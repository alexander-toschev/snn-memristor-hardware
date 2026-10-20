"""Independent diagonal-gain replay; native request/guard/ownership math inherited unchanged."""
import math
from tools import audit_mnist_conductance as previous
from tools.audit_mnist_conductance import require,same,GAINS,GROUPS,digest,replay_neuron,choose_update


def expected_protocol():
    return {**previous.expected_protocol(),
        'schema':'mnist-pooled32-diagonal-gain-pilot128-v1',
        'gain_objective':'per learning WL min(abs(total fit posts-32),gain); paired control shares gain',
        'gain_transform':'diag(g0,g1,g2,g3,g0,g1,g2,g3), frozen after fresh fit',
        'reference_sim_enabled':False,
        'reference_semantics':'No new reference run in this variant; original scalar main/reference remain historical evidence'}


class Replay(previous.Replay):
    def image(self, phase, index):
        require((phase=='fit' and self.gain is None) or (phase!='fit' and isinstance(self.gain,list) and len(self.gain)==8 and all(g in GAINS for g in self.gain) and self.gain[:4]==self.gain[4:]),'Acquisition before gain freeze')
        ordinal=self.ordinal;self.ordinal+=1
        schedule=self.schedules[index]
        self.expect('image_begin',phase=phase,ordinal=ordinal,train_index=index,image_sha256=self.images[index]['image_sha256'],schedule_sha256=digest(schedule),schedule=schedule)
        metadata=dict(image_phase=phase,image_ordinal=ordinal,train_index=index)
        order=list(range(8)) if ordinal%2==0 else list(reversed(range(8)))
        before={wl:self.vector(wl,[],'image_zero_before',**metadata) for wl in order}
        signals=[]
        for tick, mask in enumerate(schedule):
            for bl in mask:self.input_events[phase][bl]+=1
            for micro, offset in enumerate(range(0,len(mask),4)):
                chunk=mask[offset:offset+4]
                wls=list(range(8)) if (ordinal+tick+micro)%2==0 else list(reversed(range(8)))
                for wl in wls:
                    signal=self.vector(wl,chunk,'image_signal',t=tick,micro=micro,**metadata)
                    signals.append(dict(wl=wl,t=tick,micro=micro,mask=chunk,**signal))
                    for bl in chunk:self.participation[wl][bl]+=1
        after={wl:self.vector(wl,[],'image_zero_after',**metadata) for wl in order[::-1]}
        voltages=[[0.]*32 for _ in range(8)]
        for signal in signals:
            wl=signal['wl']
            voltages[wl][signal['t']]+=signal['voltage_v']-(before[wl]['voltage_v']+after[wl]['voltage_v'])/2
        result=dict(phase=phase,ordinal=ordinal,train_index=index,schedule=schedule,voltages=voltages,
            zero_before=before,zero_after=after,signals=signals,zero_drift_v=[after[wl]['voltage_v']-before[wl]['voltage_v'] for wl in range(8)])
        self.expect('image_measurements_complete',**result)
        if phase!='fit':
            neurons=[]
            for wl in range(8):
                steps=replay_neuron(schedule,voltages[wl],self.gain[wl])
                for step in steps:
                    self.expect('lif_step',image_ordinal=ordinal,image_phase=phase,train_index=index,wl=wl,**step)
                neurons.append(dict(wl=wl,posts=[s['post'] for s in steps],eligibility=steps[-1]['eligibility'],choice=choose_update(steps[-1]['eligibility'])))
            outcome=dict(phase=phase,ordinal=ordinal,train_index=index,neurons=neurons,zero_drift_v=result['zero_drift_v'])
            self.expect('image_feedback_complete',**outcome);self.outcomes.append(outcome)
            if phase=='train':
                for wl in (range(4) if index%2==0 else reversed(range(4))):
                    choice=neurons[wl]['choice']
                    if choice is None:self.expect('no_training_update',image_ordinal=ordinal,train_index=index,wl=wl)
                    else:self.pulse(wl,choice['bl'],choice['sign'],'training_update',image_ordinal=ordinal,train_index=index,eligibility=choice['eligibility'])
        self.expect('image_complete',phase=phase,ordinal=ordinal,train_index=index,impacts=self.calls,pulses=self.pulses)
        return result

    def gain_fit(self, images):
        candidates=[]
        for gain in GAINS:
            rows=[]
            for wl in range(4):
                counts=[];empty=0
                for image in images:
                    steps=replay_neuron(image['schedule'],image['voltages'][wl],gain)
                    for step in steps:
                        self.expect('fit_candidate_lif_step',image_ordinal=image['ordinal'],image_phase='fit',train_index=image['train_index'],wl=wl,**step)
                    counts.append(sum(s['post'] for s in steps))
                    empty+=sum(s['post'] for s in steps if not any(s['pre']))
                eligible=8<=sum(counts)<=64 and sum(n>0 for n in counts)>=4 and max(counts)<=8 and empty==0
                rows.append(dict(wl=wl,post_counts=counts,total_posts=sum(counts),images_with_posts=sum(n>0 for n in counts),empty_input_posts=empty,eligible=eligible))
            total=sum(row['total_posts'] for row in rows)
            candidate=dict(gain=gain,per_wl=rows,total_posts=total,eligible=all(row['eligible'] for row in rows),objective=[abs(total-128),gain])
            self.expect('gain_candidate',**candidate);candidates.append(candidate)
        common=[c for c in candidates if c['eligible']]
        shared=min(common,key=lambda row:tuple(row['objective']))['gain'] if common else None
        selected=[];gains=[]
        for wl in range(4):
            options=[(c['gain'],c['per_wl'][wl]) for c in candidates if c['per_wl'][wl]['eligible']]
            if not options:break
            g,row=min(options,key=lambda p:(abs(p[1]['total_posts']-32),p[0]))
            gains.append(g);selected.append(dict(wl=wl,gain=g,total_posts=row['total_posts']))
        self.gain=gains+gains if len(gains)==4 else None
        self.calibration=dict(selected_gain=self.gain,candidates=candidates,per_wl_selected=selected,
            selection='per_learning_WL_abs_posts_minus32_then_gain',paired_control_gain=True,
            shared_gain_diagnostic=shared,fit_image_ordinals=list(range(32)),
            fit_train_indices=GROUPS['fit'],control_used=False,labels_used=False)
        self.expect('gain_frozen',last_native_task_id=self.calls,**self.calibration)

    def run(self, protocol):
        same(protocol,expected_protocol(),'registered independent protocol')
        self.expect('session_begin',protocol=protocol,reference=self.reference)
        opened=self.take('port_open')
        require(set(self.payload(opened))=={'result'} and bool(opened['result']), 'Open not confirmed')
        self.checkpoint('initial')
        for decision in range(8):
            for bl in range(32):
                for wl in (range(8) if (decision+bl)%2==0 else reversed(range(8))):
                    self.pulse(wl,bl,int(decision>=4),'preparation',preparation_step=decision)
        self.checkpoint('after_preparation')
        fit=[self.image('fit',index) for index in GROUPS['fit']]
        self.checkpoint('after_fit')
        if self.reference is None:
            self.gain_fit(fit)
        else:
            self.gain=self.reference['gain']
            require(self.gain in GAINS,'Reference gain outside fixed grid')
            self.calibration=dict(selected_gain=self.gain,candidates=[],selection='bound main fit only; reference fit does not change gain',reference=self.reference,fit_image_ordinals=list(range(32)),fit_train_indices=GROUPS['fit'])
            self.expect('gain_frozen',last_native_task_id=self.calls,**self.calibration)
        if self.gain is None:
            self.expect('calibration_stop',reason='No fixed candidate eligible; no probes or training')
            require(self.calls<=34560 and self.pulses<=2048, 'No-eligible cap exceeded')
            status='uninformative_calibration'
        else:
            for index in GROUPS['probe']:self.image('preprobe',index)
            self.checkpoint('after_preprobe')
            for index in GROUPS['train']:
                self.image('train',index)
                if (index+1)%16==0:self.checkpoint('after_train_%03d'%(index+1))
            for index in GROUPS['probe']:self.image('postprobe',index)
            self.checkpoint('after_postprobe')
            begin=self.expect('retention_begin',seconds=60.)
            end=self.take('retention_end');elapsed=end['elapsed_s']
            require(set(self.payload(end))=={'elapsed_s'} and type(elapsed) in (int,float) and math.isfinite(elapsed) and elapsed>=60. and end['monotonic_s']-begin['monotonic_s']>=elapsed, 'Retention wait invalid')
            for index in GROUPS['probe']:self.image('retentionprobe',index)
            self.checkpoint('after_retentionprobe');status='completed'
        closed=self.take('port_close')
        require(set(self.payload(closed))=={'result'} and bool(closed['result']), 'Close not confirmed')
        self.expect('session_end',status=status,error=None,impacts=self.calls,pulses=self.pulses,decisions=self.decisions)
        try:
            next(self.rows)
        except StopIteration:
            pass
        else:
            raise ValueError('Events after terminal session_end')
        require(self.ordinal==(208 if status=='completed' else 32) and len(self.checkpoints)==(14 if status=='completed' else 3), 'Incomplete image/checkpoint coverage')
        return self.summary(status)
