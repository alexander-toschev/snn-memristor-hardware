"""Fit-only digital-gain STDP v2; hardware I/O delegates to frozen public-Connector collector."""
import hashlib
import math
from pathlib import Path

from experiments.conductance_stdp_20261019 import core as previous

PREVIOUS_CORE_SHA='005586394be54aad0ac7d116c729d7b733628161ef7a442975ee9b94209aa86f'
if hashlib.sha256(Path(previous.__file__).read_bytes()).hexdigest()!=PREVIOUS_CORE_SHA:
    raise ValueError('Frozen public-Connector collector changed')

BL=previous.BL
WL=previous.WL
GAINS=(10.,20.,40.,80.,100.,200.,400.,800.,1600.)
SPEC={**previous.SPEC,
      'schema':'conductance-stdp-fit-gain-v2',
      'lif_gain':None,'gain_candidates':list(GAINS),
      'gain_fit_wl':5,'gain_fit_windows':2,'pretest_windows':2,
      'posttest_windows':4,'retention_windows':4,
      'gain_eligibility':'1..3 post spikes per fit window; zero spikes at every zero-input step',
      'gain_selection':'min(abs(total_fit_posts-4), gain) among eligible candidates',
      'gain_shared_with_control':True,'gain_freeze_before_pretest':True,
      'fit_times':[[1,5,9,13],[13,9,5,1]],
      'train_times':[[3,10,11,2],[2,11,10,3]],
      'probe_times':[[1,4,9,12],[9,12,1,4]],
      'fit_failure':'uninformative_calibration; stop before pretest or training',
      'maximum_fit_failure_impacts':1536,
      'design_origin':'new preregistered variant motivated by all-silent v1 native SIM; no heldout tuning'}
SPEC.pop('probe_windows_per_phase')  # v2 has two pretest and four later probe windows.


def timing(phase,index):
    if phase=='fit':return tuple(SPEC['fit_times'][index%2])
    if phase=='train':return tuple(SPEC['train_times'][index%2])
    if phase in ('pretest','posttest','retention'):return tuple(SPEC['probe_times'][index%2])
    raise ValueError('Unknown window phase')


class Neuron:
    def __init__(self,gain):
        if gain not in GAINS:raise ValueError('Gain outside frozen grid')
        self.gain=gain;self.u=0.;self.a=[0.]*4;self.b=0.;self.e=[0.]*4

    def step(self,pre,voltage):
        if len(pre)!=4 or any(type(x) is not int or x not in (0,1) for x in pre) or not math.isfinite(voltage):
            raise ValueError('Invalid digital input')
        before=self.u;unreset=.5*before+self.gain*voltage;post=int(unreset>=1.)
        self.u=0. if post else unreset
        decay=math.exp(-.25)
        self.a=[value*decay for value in self.a];self.b*=decay
        increment=[value*post-1.05*self.b*spike for value,spike in zip(self.a,pre)]
        self.e=[value+delta for value,delta in zip(self.e,increment)]
        self.a=[value+spike for value,spike in zip(self.a,pre)];self.b+=post
        return dict(gain=self.gain,u_before=before,u_unreset=unreset,u_after=self.u,post=post,
                    pre=list(pre),voltage_v=voltage,eligibility=list(self.e),
                    eligibility_increment=increment,pre_trace=list(self.a),post_trace=self.b)


def choose_gain(fit_windows):
    """Only the two WL5 fit recordings enter selection; candidates never affect I/O."""
    if len(fit_windows)!=2 or [w['index'] for w in fit_windows]!=[0,1] or any(w['phase']!='fit' for w in fit_windows):
        raise ValueError('Exactly two ordered fresh fit windows required')
    candidates=[]
    for gain in GAINS:
        windows=[]
        for window in fit_windows:
            records=[step for step in window['steps'] if step['wl']==5]
            if [step['t'] for step in records]!=list(range(16)):raise ValueError('Fit WL5 coverage incomplete')
            neuron=Neuron(gain);steps=[]
            for record in records:
                expected=[int(record['t']==value) for value in timing('fit',window['index'])]
                if record['pre']!=expected or record['measurement']['wl']!=5:
                    raise ValueError('Wrong fit timing/bank')
                result=neuron.step(expected,record['measurement']['centered_v'])
                steps.append(dict(t=record['t'],signal_task_id=record['measurement']['signal']['task_id'],**result))
            posts=sum(step['post'] for step in steps)
            zero_posts=sum(step['post'] for step in steps if not any(step['pre']))
            windows.append(dict(index=window['index'],posts=posts,zero_input_posts=zero_posts,steps=steps))
        total=sum(window['posts'] for window in windows)
        eligible=all(1<=window['posts']<=3 and window['zero_input_posts']==0 for window in windows)
        candidates.append(dict(gain=gain,eligible=eligible,total_posts=total,
                               objective=[abs(total-4),gain],windows=windows))
    eligible=[candidate for candidate in candidates if candidate['eligible']]
    selected=min(eligible,key=lambda candidate:tuple(candidate['objective']))['gain'] if eligible else None
    return dict(selected_gain=selected,candidates=candidates,fit_wl=5,
                fit_window_indices=[0,1],control_used_for_selection=False,
                qualification='fit-only readout operating-point calibration, not classification or physical delta-t STDP')


class Collector(previous.Collector):
    """All native reads/writes/guards are unchanged inherited public Connector calls."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.gain=None;self.calibration=None

    def window(self,phase,index):
        if phase!='fit' and self.gain is None:raise ValueError('Digital gain not frozen')
        times=timing(phase,index)
        neurons={wl:Neuron(self.gain) for wl in WL} if phase!='fit' else None
        steps=[]
        for t in range(16):
            pre=[int(t==value) for value in times];mask=[bl for bl,spike in zip(BL,pre) if spike]
            for wl in (WL if t%2==0 else WL[::-1]):
                measurement=self.bracket(wl,mask,'window',window_phase=phase,window_index=index,t=t)
                if phase=='fit':
                    step=dict(wl=wl,t=t,pre=pre,measurement=measurement)
                    self.emit('fit_measurement_step',window_phase=phase,window_index=index,**step)
                else:
                    step=dict(wl=wl,t=t,measurement=measurement,
                              **neurons[wl].step(pre,measurement['centered_v']))
                    self.emit('lif_step',window_phase=phase,window_index=index,**step)
                steps.append(step)
        result=dict(phase=phase,index=index,times=list(times),steps=steps,
                    eligibility=None if neurons is None else {str(wl):neurons[wl].e for wl in WL},
                    gain=None if phase=='fit' else self.gain)
        self.windows.append(result);self.emit('window_complete',**result)
        if phase=='train':
            for bl,eligibility in zip(BL,neurons[5].e):
                if abs(eligibility)>=.25:
                    self.pulse(5,bl,int(eligibility>=.25),'training_update',window_index=index,eligibility=eligibility)
                else:self.emit('no_update',wl=5,bl=bl,window_index=index,eligibility=eligibility)
        return result

    def summary(self,status):
        counts={phase:dict(decisions=sum(value['phase']==phase for value in self.updates),
                           admitted=sum(value['phase']==phase and value['admitted'] for value in self.updates),
                           current_skipped=sum(value['phase']==phase and not value['admitted'] for value in self.updates))
                for phase in ('preparation','training_update')}
        evaluated=[window for window in self.windows if window['phase']!='fit']
        lif={str(wl):dict(spikes=sum(step['post'] for window in evaluated for step in window['steps'] if step['wl']==wl),
                          windows_with_no_spikes=sum(not any(step['post'] for step in window['steps'] if step['wl']==wl) for window in evaluated),
                          windows_with_one_post_per_input=sum(all(step['post']==int(any(step['pre'])) for step in window['steps'] if step['wl']==wl) for window in evaluated))
             for wl in WL}
        return dict(status=status,impacts=self.calls,pulses=self.pulses,decisions=self.decisions,
                    counts=counts,lif_summary=lif,calibration=self.calibration,gain=self.gain,
                    evaluated_windows=len(evaluated),uninformative_all_silent=not evaluated or all(value['spikes']==0 for value in lif.values()),
                    qualification='fit-calibrated logical STDP pilot; no intrinsic delta-t STDP or accuracy claim',
                    checkpoints=self.checkpoints,windows=self.windows,updates=self.updates)

    def run(self):
        status='failed';opened=False;error=None
        try:
            self.emit('session_begin',protocol=SPEC)
            opened=True
            result=self.connector.open_port();self.emit('port_open',result=previous.safe(result))
            if not result:raise ValueError('Connector open failed')
            self.all_cells('initial')
            for decision in range(8):
                for j,bl in enumerate(BL):
                    sign=int((j<2)==(decision<4))
                    for wl in (WL if (decision+j)%2==0 else WL[::-1]):
                        self.pulse(wl,bl,sign,'preparation',preparation_step=decision)
                        self.checkpoint(wl,bl,'after_preparation_decision',preparation_step=decision)
            self.all_cells('after_preparation')
            fits=[self.window('fit',index) for index in range(2)]
            self.calibration=choose_gain(fits);self.gain=self.calibration['selected_gain']
            self.emit('gain_frozen',last_fit_task_id=self.calls,**self.calibration)
            if self.gain is None:
                status='uninformative_calibration'
                self.emit('calibration_stop',reason='No candidate meets preregistered fit eligibility; no pretest/training executed')
            else:
                for index in range(2):self.window('pretest',index)
                self.all_cells('baseline_before_training')
                for index in range(16):
                    self.window('train',index);self.all_cells('after_training_window',window_index=index)
                    print('STDP-v2 training window %d/16; gain=%g impacts=%d pulses=%d'%(index+1,self.gain,self.calls,self.pulses),flush=True)
                for index in range(4):self.window('posttest',index)
                self.all_cells('after_posttest')
                self.emit('retention_begin',seconds=60.)
                started=self.monotonic();self.sleep(60.);elapsed=self.monotonic()-started
                if elapsed<60.:raise ValueError('Retention wait ended early')
                self.emit('retention_end',elapsed_s=elapsed)
                for index in range(4):self.window('retention',index)
                self.all_cells('after_retention');status='completed'
        except (Exception,KeyboardInterrupt) as exc:
            error=type(exc).__name__+': '+str(exc);raise
        finally:
            try:
                if opened:
                    try:
                        closed=self.connector.close_port();self.emit('port_close',result=previous.safe(closed))
                        if not closed:raise ValueError('Connector close failed')
                    except (Exception,KeyboardInterrupt) as exc:
                        status='failed';error=type(exc).__name__+': '+str(exc);raise
            finally:
                try:self.emit('session_end',status=status,error=error,impacts=self.calls,pulses=self.pulses,decisions=self.decisions)
                finally:self.writer.finish()
        return self.summary(status)
