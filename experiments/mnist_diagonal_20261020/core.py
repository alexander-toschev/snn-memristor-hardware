"""Diagonal digital gain; unchanged physical I/O, fixture and bounded STDP schedule."""
from experiments.mnist_conductance_20261019 import core as previous
GROUPS=previous.GROUPS
safe=previous.safe
GAINS=previous.GAINS
SPEC={**previous.SPEC,
    'schema':'mnist-pooled32-diagonal-gain-pilot128-v1',
    'gain_objective':'per learning WL min(abs(total fit posts-32),gain); paired control shares gain',
    'gain_transform':'diag(g0,g1,g2,g3,g0,g1,g2,g3), frozen after fresh fit',
    'reference_sim_enabled':False,
    'reference_semantics':'No new reference run in this variant; original scalar main/reference remain historical evidence'}
digest=previous.digest


def select_diagonal(candidates):
    selected=[]
    for wl in range(4):
        eligible=[(row['gain'],row['per_wl'][wl]) for row in candidates if row['per_wl'][wl]['eligible']]
        if not eligible:return None,selected
        gain,row=min(eligible,key=lambda p:(abs(p[1]['total_posts']-32),p[0]))
        selected.append(dict(wl=wl,gain=gain,total_posts=row['total_posts']))
    gains=[row['gain'] for row in selected]
    return gains+gains,selected


def choose_gain(images,emit):
    shared=previous.choose_gain(images,emit)
    gains,selected=select_diagonal(shared['candidates'])
    return dict(selected_gain=gains,candidates=shared['candidates'],per_wl_selected=selected,
        selection='per_learning_WL_abs_posts_minus32_then_gain',paired_control_gain=True,
        shared_gain_diagnostic=shared['selected_gain'],
        fit_image_ordinals=shared['fit_image_ordinals'],fit_train_indices=GROUPS['fit'],control_used=False,labels_used=False)


class Collector(previous.Collector):
    def __init__(self,*args,**kwargs):
        if kwargs.get('reference') is not None:raise ValueError('Diagonal reference mode is not implemented')
        super().__init__(*args,**kwargs)

    def image(self,phase,index):
        if phase not in self.input_event_counts or (phase=='fit' and self.gain is not None) or (phase!='fit' and self.gain is None):
            raise ValueError('Invalid image phase/gain state')
        fixture=self.images[index]; schedule=fixture['schedule']; ordinal=self.ordinal; self.ordinal+=1
        self.emit('image_begin',phase=phase,ordinal=ordinal,train_index=index,
                  image_sha256=fixture['image_sha256'],schedule_sha256=fixture['schedule_sha256'],schedule=schedule)
        order=list(range(8)) if ordinal%2==0 else list(reversed(range(8)))
        before={wl:self.vector(wl,[],'image_zero_before',image_phase=phase,image_ordinal=ordinal,train_index=index) for wl in order}
        signals=[]
        for t,mask in enumerate(schedule):
            for b in mask:self.input_event_counts[phase][b]+=1
            for micro,offset in enumerate(range(0,len(mask),4)):
                chunk=mask[offset:offset+4]
                wls=list(range(8)) if (ordinal+t+micro)%2==0 else list(reversed(range(8)))
                for wl in wls:
                    signal=self.vector(wl,chunk,'image_signal',image_phase=phase,image_ordinal=ordinal,
                                       train_index=index,t=t,micro=micro)
                    signals.append(dict(wl=wl,t=t,micro=micro,mask=chunk,**signal))
                    for b in chunk:self.image_cell_participation[wl][b]+=1
        after={wl:self.vector(wl,[],'image_zero_after',image_phase=phase,image_ordinal=ordinal,train_index=index) for wl in order[::-1]}
        voltages=[[0.]*32 for _ in range(8)]
        for signal in signals:
            wl=signal['wl']; voltages[wl][signal['t']]+=signal['voltage_v']-(before[wl]['voltage_v']+after[wl]['voltage_v'])/2
        result=dict(phase=phase,ordinal=ordinal,train_index=index,schedule=schedule,voltages=voltages,
            zero_before=before,zero_after=after,signals=signals,
            zero_drift_v=[after[wl]['voltage_v']-before[wl]['voltage_v'] for wl in range(8)])
        self.emit('image_measurements_complete',**result)
        if phase!='fit':
            neurons=[previous.replay_image(result,self.gain[wl],wl,self.emit) for wl in range(8)]
            outcome=dict(phase=phase,ordinal=ordinal,train_index=index,neurons=neurons,zero_drift_v=result['zero_drift_v'])
            self.outcomes.append(outcome); self.emit('image_feedback_complete',**outcome)
            if phase=='train':
                for wl in (range(4) if index%2==0 else reversed(range(4))):
                    choice=neurons[wl]['choice']
                    if choice is None:self.emit('no_training_update',image_ordinal=ordinal,train_index=index,wl=wl)
                    else:self.pulse(wl,choice['bl'],choice['sign'],'training_update',
                        image_ordinal=ordinal,train_index=index,eligibility=choice['eligibility'])
        self.emit('image_complete',phase=phase,ordinal=ordinal,train_index=index,impacts=self.calls,pulses=self.pulses)
        self.writer.checkpoint(reason='complete_image',image_phase=phase,ordinal=ordinal,train_index=index,impacts=self.calls,pulses=self.pulses)
        return result

    def run(self):
        status='failed'; opened=False; error=None
        try:
            self.emit('session_begin',protocol=SPEC,reference=self.reference)
            opened=True; result=self.connector.open_port(); self.emit('port_open',result=safe(result))
            if not result:raise ValueError('Public Connector open failed')
            self.checkpoint('initial')
            for decision,sign in enumerate(SPEC['preparation_signs']):
                for bl in range(32):
                    for wl in (range(8) if (decision+bl)%2==0 else reversed(range(8))):
                        self.pulse(wl,bl,sign,'preparation',preparation_step=decision)
            self.checkpoint('after_preparation')
            fit=[self.image('fit',index) for index in GROUPS['fit']]
            self.checkpoint('after_fit')
            if self.reference is None:
                self.calibration=choose_gain(fit,self.emit)
            else:
                self.calibration=dict(selected_gain=self.reference['gain'],candidates=[],
                    selection='bound main fit only; reference fit does not change gain',reference=self.reference,
                    fit_image_ordinals=[r['ordinal'] for r in fit],fit_train_indices=GROUPS['fit'])
            self.gain=self.calibration['selected_gain']
            self.emit('gain_frozen',last_native_task_id=self.calls,**self.calibration)
            if self.gain is None:
                if self.calls>34560 or self.pulses>2048:raise ValueError('No-eligible budget exceeded')
                self.emit('calibration_stop',reason='No fixed candidate eligible; no probes or training')
                status='uninformative_calibration'
            else:
                for index in GROUPS['probe']:self.image('preprobe',index)
                self.checkpoint('after_preprobe')
                for index in GROUPS['train']:
                    self.image('train',index)
                    if (index+1)%16==0:
                        self.checkpoint('after_train_%03d'%(index+1))
                        print('MNIST conductance train %d/128; impacts=%d pulses=%d'%(index+1,self.calls,self.pulses),flush=True)
                for index in GROUPS['probe']:self.image('postprobe',index)
                self.checkpoint('after_postprobe')
                self.emit('retention_begin',seconds=60.); begin=self.monotonic(); self.sleep(60.)
                elapsed=self.monotonic()-begin
                if elapsed<60.:raise ValueError('Retention wait ended early')
                self.emit('retention_end',elapsed_s=elapsed)
                for index in GROUPS['probe']:self.image('retentionprobe',index)
                self.checkpoint('after_retentionprobe'); status='completed'
        except (Exception,KeyboardInterrupt) as exc:
            error=repr(exc); raise
        finally:
            try:
                if opened:
                    try:
                        closed=self.connector.close_port(); self.emit('port_close',result=safe(closed))
                        if not closed:raise ValueError('Public Connector close failed')
                    except (Exception,KeyboardInterrupt) as exc:
                        status='failed'; error=repr(exc); raise
            finally:
                try:self.emit('session_end',status=status,error=error,impacts=self.calls,pulses=self.pulses,decisions=self.decisions)
                finally:self.writer.finish()
        return self.summary(status)
