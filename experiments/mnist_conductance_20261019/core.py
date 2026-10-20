"""New pooled32→8 conductance pilot; every physical operation uses public Connector."""
import contextlib
import hashlib
import json
import math
import statistics
import time

from experiments.conductance_stdp_20261019.core import DriverLog, safe, utc
from tools.prepare_mnist_conductance_fixture import GROUPS

GAINS = (1., 2., 4., 8., 10., 20., 40., 80., 100., 200., 400., 800., 1600.)
SPEC = dict(schema='mnist-pooled32-conductance-pilot128-v1', timesteps=32,
    shape=[32, 8], learning_wl=[0, 1, 2, 3], control_wl=[4, 5, 6, 7],
    groups=GROUPS, amplitude_v=.05, maximum_mask=4, reset_v=1.4, set_v=1.7,
    preparation_signs=[0]*4+[1]*4, seed=20261019, gain_candidates=list(GAINS),
    gain_fit='learning WL only; each8..64posts,>=4active images,max8/image,no empty-input posts',
    gain_objective='min(abs(total learning posts-128),gain)', lif_leak=.5, lif_threshold=1.,
    trace_decay=math.exp(-.25), depression=1.05, eligibility_threshold=.25,
    updates='one max-absolute-E BL per learningWL after complete image/zero_after; lowerBL tie',
    checkpoints=14, checkpoint_calls=3840, retention_seconds=60.,
    maximum_impacts=172288, maximum_pulses=2560, maximum_decisions=2560,
    no_eligible_maximum_impacts=34560, no_eligible_maximum_pulses=2048,
    simulator_pulse_width_s=.0001, hardware_submitted_width_s=0., hardware_physical_width_s=None,
    maximum_raw_expanded_bytes=2*1024**3, raw_chunk_bytes=8*1024**2,
    raw_minimum_free_bytes=2*1024**3, initial_minimum_free_bytes=6*1024**3,
    labels_used=False, classification_accuracy_measured=False, reference_sim_enabled=True,
    reference_maximum_impacts=171776, reference_maximum_pulses=2048,
    reference_semantics='SIM only: same initial array/preparation/inputs, main gain frozen; own shadow choices,4reads,no training pulse')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


class Neuron:
    def __init__(self, gain):
        if gain not in GAINS:
            raise ValueError('Gain outside fixed grid')
        self.gain=gain; self.u=0.; self.a=[0.]*32; self.b=0.; self.e=[0.]*32

    def step(self, pre, voltage):
        if len(pre)!=32 or any(type(s) is not int or s not in (0,1) for s in pre) or not math.isfinite(voltage):
            raise ValueError('Invalid digital input')
        before=self.u; unreset=.5*self.u+self.gain*voltage; post=int(unreset>=1.)
        self.u=0. if post else unreset
        self.a=[a*SPEC['trace_decay'] for a in self.a]; self.b*=SPEC['trace_decay']
        increment=[a*post-1.05*self.b*s for a,s in zip(self.a,pre)]
        self.e=[e+d for e,d in zip(self.e,increment)]
        self.a=[a+s for a,s in zip(self.a,pre)]; self.b+=post
        return dict(pre=list(pre), voltage_v=voltage, gain=self.gain, u_before=before,
            u_unreset=unreset, u_after=self.u, post=post, pre_trace=list(self.a),
            post_trace=self.b, eligibility_increment=increment, eligibility=list(self.e))


def shadow_choice(eligibility):
    if len(eligibility)!=32 or any(not math.isfinite(e) for e in eligibility):
        raise ValueError('Invalid eligibility')
    bl=min(range(32), key=lambda b:(-abs(eligibility[b]), b))
    return None if abs(eligibility[bl])<.25 else dict(bl=bl, eligibility=eligibility[bl], sign=int(eligibility[bl]>0))


def replay_image(image, gain, wl, emit):
    neuron=Neuron(gain); posts=[]
    for t in range(32):
        mask=image['schedule'][t]; pre=[int(b in mask) for b in range(32)]
        step=neuron.step(pre, image['voltages'][wl][t]); posts.append(step['post'])
        emit('lif_step', image_ordinal=image['ordinal'], image_phase=image['phase'],
             train_index=image['train_index'], wl=wl, t=t, **step)
    return dict(wl=wl, posts=posts, eligibility=list(neuron.e), choice=shadow_choice(neuron.e))


def choose_gain(images, emit):
    if [row['train_index'] for row in images]!=GROUPS['fit'] or any(row['phase']!='fit' for row in images):
        raise ValueError('Gain selection requires exactly32 ordered fit images')
    candidates=[]
    for gain in GAINS:
        per_wl=[]
        for wl in range(4):
            counts=[]; zero_posts=0
            for image in images:
                def fit_emit(event, **fields):
                    emit('fit_candidate_'+event, **fields)
                result=replay_image(image, gain, wl, fit_emit)
                counts.append(sum(result['posts']))
                zero_posts+=sum(p for t,p in enumerate(result['posts']) if not image['schedule'][t])
            eligible=8<=sum(counts)<=64 and sum(n>0 for n in counts)>=4 and max(counts)<=8 and zero_posts==0
            per_wl.append(dict(wl=wl, post_counts=counts, total_posts=sum(counts),
                images_with_posts=sum(n>0 for n in counts), empty_input_posts=zero_posts, eligible=eligible))
        total=sum(row['total_posts'] for row in per_wl)
        candidate=dict(gain=gain, per_wl=per_wl, total_posts=total,
            eligible=all(row['eligible'] for row in per_wl), objective=[abs(total-128),gain])
        candidates.append(candidate); emit('gain_candidate', **candidate)
    eligible=[row for row in candidates if row['eligible']]
    selected=min(eligible,key=lambda row:tuple(row['objective']))['gain'] if eligible else None
    return dict(selected_gain=selected,candidates=candidates,fit_image_ordinals=[r['ordinal'] for r in images],
                fit_train_indices=GROUPS['fit'],control_used=False,labels_used=False)


class Collector:
    def __init__(self, connector, converters, writer, source, fixture, sleep=time.sleep, monotonic=time.monotonic, reference=None):
        if source not in ('hardware','simulator') or connector.cb_type!=('real' if source=='hardware' else 'simulator'):
            raise ValueError('Connector/source mismatch')
        self.connector,self.converters,self.writer,self.source=connector,converters,writer,source
        if reference is not None and (source!='simulator' or reference.get('gain') not in GAINS):
            raise ValueError('No-update reference requires SIM and qualified main gain')
        self.reference=reference
        self.images={row['train_index']:row for row in fixture['images']}
        if set(self.images)!={i for group in GROUPS.values() for i in group}:
            raise ValueError('Fixture coverage differs')
        self.sleep,self.monotonic=sleep,monotonic
        self.calls=self.pulses=self.decisions=self.ordinal=0; self.gain=None; self.calibration=None
        self.outcomes=[]; self.updates=[]; self.checkpoints=[]
        self.image_cell_participation=[[0]*32 for _ in range(8)]
        self.input_event_counts={phase:[0]*32 for phase in ('fit','preprobe','train','postprobe','retentionprobe')}

    def emit(self,event,**fields):
        self.writer.write(dict(event=event,source=self.source,utc=utc(),monotonic_s=self.monotonic(),**fields))

    def call(self,command,phase,**meta):
        if self.calls>=(SPEC['reference_maximum_impacts'] if self.reference else SPEC['maximum_impacts']):
            raise ValueError('Impact budget exhausted')
        self.calls+=1; task={**command,'id':self.calls if command['mode_flag']==7 else 0}
        fields=dict(task_id=self.calls,command_si=task,phase=phase,**meta)
        self.emit('request',**fields); started=self.monotonic()
        try:
            with contextlib.redirect_stdout(DriverLog(self,'stdout')),contextlib.redirect_stderr(DriverLog(self,'stderr')):
                result=self.connector.impact({**task,'vol':list(task['vol']) if isinstance(task['vol'],list) else task['vol']})
            self.emit('result',**fields,native_result=safe(result),latency_s=self.monotonic()-started)
            if not isinstance(result,(list,tuple)) or len(result)!=3:
                raise ValueError('Malformed native response')
            resistance,echo,adc=result
            if type(echo) is not int or echo!=task['id'] or type(adc) is not int:
                raise ValueError('Invalid native echo/ADC type')
            if task['mode_flag']==7:
                if type(resistance) not in (float,int) or not math.isfinite(resistance) or resistance<=1e-8 or not 20<=adc<=16382:
                    raise ValueError('Invalid native R/ADC')
            elif not 0<=adc<=16382:
                raise ValueError('Invalid native MVM ADC')
            return dict(r=resistance,adc=adc,task_id=self.calls)
        except (Exception,KeyboardInterrupt) as exc:
            self.emit('call_error',task_id=self.calls,error=repr(exc)); raise

    def read_r(self,wl,bl,phase,**meta):
        if type(wl) is not int or type(bl) is not int or not 0<=wl<8 or not 0<=bl<32:
            raise ValueError('Out-of-scope cell')
        return self.call(dict(mode_flag=7,wl=wl,bl=bl,vol=0.,sign=0,pulse_width=0.),phase,**meta)

    def vector(self,wl,mask,phase,**meta):
        if type(wl) is not int or not 0<=wl<8 or len(mask)>4 or mask!=sorted(set(mask)) or any(type(b) is not int or not 0<=b<32 for b in mask):
            raise ValueError('Invalid vector address')
        guards=[self.read_r(wl,b,'vector_guard',vector_phase=phase,**meta) for b in mask]
        dac=self.converters.convert_volt_to_dac(12,5,.05)
        amplitude=max(.05,self.converters.convert_dac_to_volt(12,5,dac))
        current=sum(amplitude/g['r'] for g in guards)
        if not math.isfinite(amplitude) or current>.005 or 330*current>=3.3:
            raise ValueError('MVM current/output guard')
        result=self.call(dict(mode_flag=10,wl=wl,vol=[.05 if b in mask else 0. for b in range(32)]),
                         phase,mask=list(mask),**meta)
        voltage=self.converters.convert_adc_to_volt(1,14,5,result['adc'])
        if not math.isfinite(voltage) or voltage>4.:
            raise ValueError('Measured ADC headroom')
        return dict(task_id=result['task_id'],adc=result['adc'],voltage_v=voltage,guards=guards)

    def pulse(self,wl,bl,sign,phase,**meta):
        if phase not in ('preparation','training_update') or sign not in (0,1) or (phase=='training_update' and wl not in range(4)):
            raise ValueError('Invalid pulse phase/address/sign')
        if self.decisions>=SPEC['maximum_decisions']:
            raise ValueError('Decision budget exhausted')
        self.decisions+=1; level=1.7 if sign else 1.4
        sham=self.read_r(wl+4,bl,'paired_control_guard',**meta) if phase=='training_update' else None
        guard=self.read_r(wl,bl,'pulse_guard',intended_phase=phase,**meta)
        admitted=level/guard['r']<=(.002 if sign else .04)
        reference_suppressed=self.reference is not None and phase=='training_update'
        result=dict(wl=wl,bl=bl,sign=sign,voltage_v=level,phase=phase,decision=self.decisions,
                    guard=guard,control_guard=sham,guard_admissible=admitted,
                    admitted=admitted and not reference_suppressed,reference_suppressed=reference_suppressed,**meta)
        self.emit('pulse_decision',**result)
        if reference_suppressed:
            post=self.read_r(wl,bl,'pulse_post_r',intended_phase=phase,**meta)
            control_post=self.read_r(wl+4,bl,'paired_control_post_r',**meta)
            result.update(post=post,control_post=control_post,delta_inverse_r=1/post['r']-1/guard['r'])
        elif admitted:
            if self.pulses>=SPEC['maximum_pulses']:
                raise ValueError('Pulse budget exhausted')
            self.pulses+=1
            pulse=self.call(dict(mode_flag=7,wl=wl,bl=bl,vol=level,sign=sign,
                pulse_width=.0001 if self.source=='simulator' else 0.),phase,**meta)
            post=self.read_r(wl,bl,'pulse_post_r',intended_phase=phase,**meta)
            control_post=self.read_r(wl+4,bl,'paired_control_post_r',**meta) if phase=='training_update' else None
            result.update(pulse=pulse,post=post,control_post=control_post,
                          delta_inverse_r=1/post['r']-1/guard['r'])
        self.updates.append(result); self.emit('pulse_complete',**result)
        self.writer.checkpoint(reason='complete_update_pair',impacts=self.calls,pulses=self.pulses,decisions=self.decisions)
        return result

    def checkpoint(self,phase):
        means=[]; self.emit('checkpoint_begin',phase=phase)
        for wl in range(8):
            row=[]
            for bl in range(32):
                reads=[self.read_r(wl,bl,'checkpoint_r',checkpoint_phase=phase,repeat=r) for r in range(3)]
                brackets=[]
                for repeat in range(3):
                    meta=dict(checkpoint_phase=phase,cell_bl=bl,repeat=repeat)
                    before=self.vector(wl,[],'checkpoint_zero_before',**meta)
                    signal=self.vector(wl,[bl],'checkpoint_signal',**meta)
                    after=self.vector(wl,[],'checkpoint_zero_after',**meta)
                    brackets.append(dict(before=before,signal=signal,after=after,
                        centered_v=signal['voltage_v']-(before['voltage_v']+after['voltage_v'])/2))
                row.append(statistics.mean(1/r['r'] for r in reads))
                self.emit('checkpoint_cell',phase=phase,wl=wl,bl=bl,reads=reads,responses=brackets)
            means.append(row)
            self.writer.checkpoint(reason='checkpoint_row_complete',checkpoint_phase=phase,wl=wl,impacts=self.calls)
        result=dict(phase=phase,mean_inverse_r=means)
        self.checkpoints.append(result); self.emit('checkpoint_complete',**result)

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
            neurons=[replay_image(result,self.gain,wl,self.emit) for wl in range(8)]
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

    def summary(self,status):
        return dict(status=status,impacts=self.calls,pulses=self.pulses,decisions=self.decisions,
            reference=self.reference,diagnostics=diagnostics(self.outcomes,self.checkpoints,self.updates),
            measured_images=self.ordinal,gain=self.gain,calibration=self.calibration,
            image_cell_participation=self.image_cell_participation,input_event_counts=self.input_event_counts,
            outcomes=self.outcomes,updates=self.updates,checkpoints=self.checkpoints,
            classification_accuracy_measured=False,physical_lif_integrator=False,
            learning_cells=128,read_cells=256,full10k_executed=False,
            summary_interpretation='Physical conductance updates with digitally evaluated measured-input LIF/eligibility; no accuracy or intrinsic delta-t claim')

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


def diagnostics(outcomes,checkpoints,updates):
    phases={}
    for phase in ('preprobe','train','postprobe','retentionprobe'):
        rows=[row for row in outcomes if row['phase']==phase]
        if not rows:continue
        posts=[[row['neurons'][wl]['posts'] for row in rows] for wl in range(8)]
        counts=[[sum(trace) for trace in bank] for bank in posts]
        pairs=[]
        for a in range(8):
            for b in range(a+1,8):
                mismatch=sum(x!=y for left,right in zip(posts[a],posts[b]) for x,y in zip(left,right))
                intersection=sum(x and y for left,right in zip(posts[a],posts[b]) for x,y in zip(left,right))
                union=sum(x or y for left,right in zip(posts[a],posts[b]) for x,y in zip(left,right))
                active=[(left,right) for left,right in zip(posts[a],posts[b]) if any(left) or any(right)]
                pairs.append(dict(wl_pair=[a,b],exact_matching_images=sum(left==right for left,right in zip(posts[a],posts[b])),
                    active_images=len(active),matching_active_images=sum(left==right for left,right in active),
                    hamming_bits=mismatch,jaccard=intersection/union if union else None,empty_union=not union))
        phases[phase]=dict(images=len(rows),post_counts_by_wl=counts,
            silent_images_by_wl=[sum(n==0 for n in values) for values in counts],
            unique_eight_neuron_outputs=len({tuple(p for neuron in row['neurons'] for p in neuron['posts']) for row in rows}),
            all_learning_traces_identical=all(row['neurons'][0]['posts']==row['neurons'][wl]['posts'] for row in rows for wl in range(1,4)),
            pairwise=pairs,
            eligibility_signs_by_wl=[dict(positive=sum(e>0 for row in rows for e in row['neurons'][wl]['eligibility']),
                negative=sum(e<0 for row in rows for e in row['neurons'][wl]['eligibility'])) for wl in range(8)])
    conductance=[]
    for checkpoint in checkpoints:
        banks=checkpoint['mean_inverse_r']; pairs=[]
        for a in range(8):
            for b in range(a+1,8):
                denom=math.sqrt(sum(x*x for x in banks[a])*sum(x*x for x in banks[b]))
                pairs.append(dict(wl_pair=[a,b],cosine=sum(x*y for x,y in zip(banks[a],banks[b]))/denom if denom else None))
        conductance.append(dict(phase=checkpoint['phase'],row_mean=[statistics.mean(row) for row in banks],
            row_cv=[statistics.pstdev(row)/statistics.mean(row) for row in banks],pairwise_cosine=pairs))
    occupancy=[[[0,0] for _ in range(32)] for _ in range(8)]
    admitted=[[[0,0] for _ in range(32)] for _ in range(8)]
    for update in updates:
        if update['phase']=='training_update':
            occupancy[update['wl']][update['bl']][update['sign']]+=1
            admitted[update['wl']][update['bl']][update['sign']]+=int(update['admitted'])
    training=[u for u in updates if u['phase']=='training_update']
    train_outcomes=[row for row in outcomes if row['phase']=='train']
    learning_posts=sum(sum(row['neurons'][wl]['posts']) for row in train_outcomes for wl in range(4))
    nonzero_e=sum(e!=0 for row in train_outcomes for wl in range(4) for e in row['neurons'][wl]['eligibility'])
    return dict(phases=phases,conductance=conductance,training_decisions_by_cell_reset_set=occupancy,
        training_pulses_by_cell_reset_set=admitted,training_attempts=len(training),
        training_admitted_pulses=sum(u['admitted'] for u in training),
        guard_skipped=sum(not u['guard_admissible'] for u in training),
        reference_suppressed=sum(u['reference_suppressed'] for u in training),
        mechanism=dict(learning_bank_train_posts=learning_posts,nonzero_learning_eligibility=nonzero_e,
            admitted_updates_with_post_r=sum(u['admitted'] and 'post' in u for u in training),
            uninformative_no_learning_posts=learning_posts==0,uninformative_no_eligibility=nonzero_e==0,
            no_admitted_training_pulses=not any(u['admitted'] for u in training)),
        admitted_delta_g_signs=dict(positive=sum(u.get('delta_inverse_r',0)>0 for u in training if u['admitted']),
            negative=sum(u.get('delta_inverse_r',0)<0 for u in training if u['admitted']),
            unchanged=sum(u.get('delta_inverse_r',0)==0 for u in training if u['admitted'])))
