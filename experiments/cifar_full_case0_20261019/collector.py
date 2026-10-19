"""Full-case collector; launch via this directory's experiment.py.

Only an injected public MemriBoard Connector performs I/O. Native converters are
injected unchanged. Registered numerical protocol requires native SIM qualification
and an explicitly selected source; registration is not a claim of HW accuracy.
"""
import hashlib
import json
import math
import random
import statistics
import time
from pathlib import Path
import contextlib
import io

from . import plan

APPROVED_PROTOCOL_SHA256 = frozenset({'2cb1d57689508725f7155bf7b1d0d17fec9d4f6b65782171e2b85481bc9aa558'})
BOARD = dict(dac_bit=12,vol_ref_dac=5,adc_bit=14,vol_ref_adc=5,gain=11,
             res_load=3000,vol_read=.3,res_switches=10,sum_gain=330,soft_cc=.002)
WEIGHTS_SHA='8be312b9eafb9de23f8340b17f848e7e0e2e7884293c826a6d9f6cbb5125d440'


def json_safe(value):
    if value is None or type(value) in (bool,int,str):return value
    if type(value) is float:return value if math.isfinite(value) else repr(value)
    if isinstance(value,(tuple,list)):return [json_safe(v) for v in value]
    return repr(value)


class DriverOutput(io.TextIOBase):
    def __init__(self,collector,stream):self.collector,self.stream=collector,stream
    def write(self,text):
        if text:self.collector.emit('driver_output',task_id=self.collector.seq,stream=self.stream,text=text)
        return len(text)
    def flush(self):pass


def load_assets():
    import numpy as np
    weight_path=plan.ROOT/'weights/conv1_weights.npy'
    if hashlib.sha256(weight_path.read_bytes()).hexdigest()!=WEIGHTS_SHA:raise ValueError('Weight file SHA mismatch')
    weights=np.load(weight_path,allow_pickle=False)
    if weights.dtype!=np.float32 or list(weights.shape)!=[64,3,5,5] or not np.isfinite(weights).all():raise ValueError('Wrong finite Conv1 weights')
    return plan.fixture()['image_chw_uint8'],weights.reshape(64,75)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def validate_protocol(protocol, frozen_sha, require_approval=True):
    if digest(protocol) != frozen_sha:
        raise ValueError('Protocol SHA mismatch')
    if require_approval and frozen_sha not in APPROVED_PROTOCOL_SHA256:
        raise ValueError('Full-case protocol is not registered after review/native SIM')
    if protocol.get('schema') != 'full-case0-bounded-v1' or protocol.get('max_active') != 4 or protocol.get('amplitude_v') != .05:
        raise ValueError('Unreviewed geometry/amplitude')
    for key in ('reset_levels_v','set_levels_v'):
        levels=protocol[key]
        if not 1<=len(levels)<=8 or any(type(v) not in (int,float) or not math.isfinite(v) or not .3<=v<=3 for v in levels) or sorted(set(levels)) != levels:
            raise ValueError('Require 1..8 increasing levels in 0.3..3 V')
    expected={'reset_stop_q':.005,'weight_absolute_tolerance_v':.00061,
              'weight_relative_tolerance':.05,'maximum_impacts':15000000,
              'maximum_programming_pulses':115200,'maximum_matrix_gain':10.,
              'maximum_matrix_condition':1000000.,'calibration_ridge':.0001,
              'hardware_gain':.125,'simulator_gain':.015,'seed':20261019,
              'programming_quality_failure':'retain_approximate_and_continue',
              'current_model_limit_a':.005,'gui_output_limit_v':3.3,
              'basis_headroom_factor':2.,'basis_sd_multiplier':3.}
    if any(protocol.get(k)!=v for k,v in expected.items()):
        raise ValueError('Protocol differs from bounded numerical contract')
    return protocol


def calibration_masks(n, seed):
    rng=random.Random(seed)
    available=[]
    used=set()
    # Four-input maximum agrees with actual inference splitting.
    for count in (64,32,32):
        group=[]
        while len(group)<count:
            density=rng.choice((1,2,3,4))
            mask=tuple(sorted(rng.sample(range(n),density)))
            if mask not in used:
                used.add(mask);group.append(list(mask))
        available.append(group)
    return dict(fit=available[0],selection=available[1],heldout=available[2])


def fit_matrix(basis, fit, selection, protocol):
    """Fit physical singleton additivity, with independent model selection."""
    import numpy as np
    zero=np.array(basis[0]['mean'])
    h=np.array([b['mean'] for b in basis[1:]])-zero
    def arrays(rows):
        return (np.array([r['centered_v'] for r in rows]),
                np.array([h[r['mask']].sum(0) for r in rows]))
    x,y=arrays(fit);sx,sy=arrays(selection)
    ridge=protocol['calibration_ridge']
    models=[('identity',np.eye(8))]
    diagonal=np.diag((x*y).sum(0)/((x*x).sum(0)+ridge))
    models.append(('diagonal',diagonal))
    rank=int(np.linalg.matrix_rank(x));reference_rank=int(np.linalg.matrix_rank(y))
    norms=np.linalg.norm(x,axis=0)
    raw_condition=float(np.linalg.cond(x/np.where(norms>0,norms,1.)))
    condition=raw_condition if math.isfinite(raw_condition) else None
    gram_condition=float(np.linalg.cond(x.T@x+ridge*np.eye(8)))
    if rank==reference_rank==8 and condition is not None and condition<=protocol['maximum_matrix_condition']:
        models.append(('full',np.linalg.solve(x.T@x+ridge*np.eye(8),x.T@y).T))
    candidates=[]
    for name,c in models:
        gain=float(np.max(np.sum(np.abs(c),axis=1)))
        if not np.isfinite(c).all() or gain>protocol['maximum_matrix_gain']:
            continue
        error=sx@c.T-sy
        score=float(np.sqrt(np.mean(error**2)))
        candidates.append(dict(name=name,matrix=c.tolist(),selection_rmse_v=score,operator_gain=gain))
    if not candidates:
        raise ValueError('No finite bounded calibration matrix')
    chosen=min(candidates,key=lambda c:(round(c['selection_rmse_v'],12),c['name']!='identity',c['name']))
    return dict(**chosen,candidates=candidates,fit_rank=rank,reference_rank=reference_rank,
                condition=condition,condition_definition='2-norm of unregularized column-L2-normalized fit design',
                regularized_gram_condition=gram_condition,targets='measured_singleton_superposition',
                trained_target_weights_used=False)


class Collector:
    def __init__(self,connector,converters,writer,protocol,frozen_sha,source):
        validate_protocol(protocol,frozen_sha)
        if source not in ('hardware','simulator') or connector.cb_type != ('real' if source=='hardware' else 'simulator'):
            raise ValueError('Connector/source mismatch')
        self.connector,self.converters,self.writer=connector,converters,writer
        self.p,self.source,self.protocol_sha=protocol,source,frozen_sha
        self.gain=protocol[source+'_gain']
        self.seq=self.pulses=0
        self.basis=None
        self.stage=-1
        self.latest_r={}

    def emit(self,event,**fields):
        self.writer.write(dict(event=event,source=self.source,stage=self.stage,**fields))

    def call(self,command,phase,**meta):
        if self.seq>=self.p['maximum_impacts']:
            raise ValueError('Impact budget exhausted')
        self.seq+=1
        task={**command,'vol':list(command['vol']) if isinstance(command['vol'],list) else command['vol']}
        if task['mode_flag']==7:
            task['id']=self.seq
        self.last_call_started=time.monotonic()
        row=dict(task_id=self.seq,command_si=task,phase=phase,monotonic_s=self.last_call_started,**meta)
        self.emit('request',**row)
        started=time.monotonic()
        try:
            with contextlib.redirect_stdout(DriverOutput(self,'stdout')),contextlib.redirect_stderr(DriverOutput(self,'stderr')):
                result=self.connector.impact({**task,'vol':list(task['vol']) if isinstance(task['vol'],list) else task['vol']})
            self.emit('result',**row,native_result=json_safe(result),latency_s=time.monotonic()-started)
            if not isinstance(result,(tuple,list)) or len(result)!=3:
                raise ValueError('Invalid native tuple')
            r,echo,adc=result
            if type(echo) is not int or echo!=task['id'] or type(adc) is not int:
                raise ValueError('Invalid native ID/ADC')
            if task['mode_flag']==7:
                if not 20<=adc<=16382 or type(r) not in (int,float) or not math.isfinite(r) or r<=1e-8:
                    raise ValueError('Invalid native R/ADC')
            elif not 0<=adc<=16382:
                raise ValueError('Invalid MVM ADC')
            return r,adc
        except (Exception,KeyboardInterrupt) as exc:
            self.emit('abort',task_id=self.seq,error=type(exc).__name__+': '+str(exc))
            raise

    def read_r(self,wl,bl,phase,**meta):
        r,_=self.call(dict(mode_flag=7,vol=0.,pulse_width=0.,sign=0,wl=wl,bl=bl,id=0),phase,**meta)
        self.latest_r[wl,bl]=r
        return r

    def pulse(self,wl,bl,level,sign,phase,**meta):
        r=self.read_r(wl,bl,'pulse_guard',intended_phase=phase,**meta)
        limit=.04 if sign==0 else BOARD['soft_cc']
        if level/r>limit:
            self.emit('write_skipped',wl=wl,bl=bl,voltage_v=level,sign=sign,native_r=r,phase=phase,**meta)
            return r,False
        if self.pulses>=self.p['maximum_programming_pulses']:
            raise ValueError('Programming budget exhausted')
        self.pulses+=1
        width=.000125 if self.source=='simulator' else 0.
        r,_=self.call(dict(mode_flag=7,vol=level,pulse_width=width,sign=sign,wl=wl,bl=bl,id=0),phase,**meta)
        return r,True

    def vector(self,mask,phase,use_basis=True,**meta):
        if len(mask)>4 or len(mask)!=len(set(mask)) or any(type(b) is not int or not 0<=b<32 for b in mask):
            raise ValueError('Invalid physical input mask')
        amplitude=self.p['amplitude_v']
        decoded=self.converters.convert_dac_to_volt(12,5,self.converters.convert_volt_to_dac(12,5,amplitude))
        resistances={}
        for wl in range(8):
            for bl in mask:
                resistances[wl,bl]=self.read_r(wl,bl,'vector_guard',vector_phase=phase,mask=mask,**meta)
        rows=[]
        for wl in range(8):
            current=max(amplitude,decoded)*sum(1/resistances[wl,b] for b in mask)
            if current>.005 or 330*current>=3.3:
                raise ValueError('Fresh native-R current/output guard')
            if use_basis and self.basis is not None:
                zero=self.basis[0]
                upper=zero['mean_adc'][wl]+3*zero['sd_adc'][wl]
                for bl in mask:
                    one=self.basis[bl+1]
                    upper+=2*max(0.,one['mean_adc'][wl]-zero['mean_adc'][wl]+3*math.hypot(one['sd_adc'][wl],zero['sd_adc'][wl]))
                if upper>.8*16383:
                    raise ValueError('Basis headroom guard')
            _,adc=self.call(dict(mode_flag=10,vol=[amplitude if bl in mask else 0. for bl in range(32)],wl=wl,id=0),
                            phase,mask=mask,**meta)
            voltage=self.converters.convert_adc_to_volt(1,14,5,adc)
            if not math.isfinite(voltage) or voltage>4.:
                raise ValueError('Observed ADC headroom')
            rows.append(dict(adc=adc,v=voltage,task_id=self.seq,monotonic_s=self.last_call_started))
        self.emit('vector_complete',phase=phase,mask=mask,rows=rows,**meta)
        return rows

    def bracket(self,mask,phase,use_basis=True,**meta):
        before=self.vector([],phase+'_zero_before',use_basis,**meta)
        signal=self.vector(mask,phase,use_basis,**meta)
        after=self.vector([],phase+'_zero_after',use_basis,**meta)
        return dict(mask=mask,centered_v=[s['v']-(a['v']+b['v'])/2 for a,s,b in zip(before,signal,after)])

    def program_stage(self,weights):
        lo,hi=plan.RANGES[self.stage%3]
        self.basis=None
        for pair in range(4):
            f=(self.stage//3)*4+pair
            for bl,tap in enumerate(range(lo,hi)):
                w=float(weights[f][tap]);target=w*self.gain
                meta=dict(filter=f,tap=tap,bl=bl,trained_weight=w,target_v=target)
                self.emit('coefficient_begin',**meta)
                for wl in (2*pair,2*pair+1):
                    for step,level in enumerate(self.p['reset_levels_v']):
                        r,admitted=self.pulse(wl,bl,level,0,'program_reset',filter=f,tap=tap,step=step)
                        if admitted and 330/r<=self.p['reset_stop_q']:
                            break
                # Fresh physical feedback across every row; no lookup basis.
                response=self.bracket([bl],'program_feedback',False,filter=f,tap=tap)
                value=response['centered_v'][2*pair]-response['centered_v'][2*pair+1]
                tolerance=max(self.p['weight_absolute_tolerance_v'],abs(target)*self.p['weight_relative_tolerance'])
                stop='initial_overshoot_or_within_tolerance'
                for step,level in enumerate(self.p['set_levels_v']):
                    direction=1 if target>=0 else -1
                    if abs(value-target)<=tolerance or direction*value>direction*target:
                        break
                    wl=2*pair+(target<0)
                    self.pulse(wl,bl,level,1,'program_set',filter=f,tap=tap,step=step)
                    response=self.bracket([bl],'program_feedback',False,filter=f,tap=tap)
                    value=response['centered_v'][2*pair]-response['centered_v'][2*pair+1]
                    stop='bounded_set_attempts'
                self.emit('coefficient_complete',**meta,actual_v=value,within_tolerance=abs(value-target)<=tolerance,
                          stop=stop,approximate_is_valid_result=True)
        self.emit('programming_complete',coefficients=4*(hi-lo))

    def calibrate(self):
        n=plan.RANGES[self.stage%3][1]-plan.RANGES[self.stage%3][0]
        basis=[]
        for index,mask in enumerate([[]]+[[b] for b in range(n)]):
            rows=[self.vector(mask,'basis',False,basis_index=index,repeat=r) for r in range(3)]
            basis.append(dict(mean=[statistics.mean(r[wl]['v'] for r in rows) for wl in range(8)],
                              mean_adc=[statistics.mean(r[wl]['adc'] for r in rows) for wl in range(8)],
                              sd_adc=[statistics.stdev(r[wl]['adc'] for r in rows) for wl in range(8)]))
        self.basis=basis
        self.emit('basis_complete',basis=basis)
        # Qualify every four-input subset conservatively before long collection.
        for wl in range(8):
            zero=basis[0]
            terms=[max(0.,b['mean_adc'][wl]-zero['mean_adc'][wl]+3*math.hypot(b['sd_adc'][wl],zero['sd_adc'][wl])) for b in basis[1:]]
            upper=zero['mean_adc'][wl]+3*zero['sd_adc'][wl]+2*sum(sorted(terms,reverse=True)[:4])
            current=self.p['amplitude_v']*sum(sorted((1/self.latest_r[wl,b] for b in range(n)),reverse=True)[:4])
            if upper>.8*16383 or current>.005 or 330*current>=3.3:
                raise ValueError('Post-write worst-four-input headroom/current guard')
        self.emit('worst_four_guard_passed',basis_sha256=digest(basis))
        masks=calibration_masks(n,self.p['seed']+self.stage)
        observations={}
        for phase in ('fit','selection'):
            observations[phase]=[self.bracket(mask,phase,mask_index=i,repeat=r)
                                 for i,mask in enumerate(masks[phase]) for r in range(3)]
        matrix=fit_matrix(basis,observations['fit'],observations['selection'],self.p)
        self.emit('calibration_frozen',matrix=matrix,calibration_sha256=digest(matrix),
                  basis_sha256=digest(basis),fit_sha256=digest(observations),gain_v_per_weight=self.gain)
        # Independent heldout observations follow freeze; no refit.
        heldout=[self.bracket(mask,'heldout',mask_index=i,repeat=r)
                 for i,mask in enumerate(masks['heldout']) for r in range(3)]
        self.emit('heldout_complete',observations=heldout,calibration_sha256=digest(matrix),refit_performed=False)
        return matrix

    def collect_position(self,image,y,x,matrix):
        self.emit('position_begin',y=y,x=x,calibration_sha256=digest(matrix))
        before=self.vector([],'inference_zero_before',y=y,x=x)
        for t,parts in enumerate(plan.masks(image,self.stage%3,y,x)):
            for split,mask in enumerate(parts):
                self.vector(mask,'inference_signal',y=y,x=x,t=t,split=split)
        after=self.vector([],'inference_zero_after',y=y,x=x)
        self.emit('position_complete',y=y,x=x,calibration_sha256=digest(matrix),
                  zero_before_task_ids=[r['task_id'] for r in before],zero_after_task_ids=[r['task_id'] for r in after])

    def run(self,image,weights,scope='full'):
        import numpy as np
        expected_image,expected_weights=load_assets()
        if image!=expected_image or not np.array_equal(np.asarray(weights),expected_weights):
            raise ValueError('Caller inputs differ from pinned image/weights')
        if scope not in ('full','smoke'):
            raise ValueError('Unknown scope')
        stages=range(48) if scope=='full' else range(3)
        positions=[(y,x) for y in range(28) for x in range(28)] if scope=='full' else [(0,0)]
        self.emit('session_begin',scope=scope,protocol=self.p,protocol_sha256=self.protocol_sha,
                  image_sha256=hashlib.sha256(bytes(image)).hexdigest(),checkpoint_sha256=plan.CHECKPOINT_SHA,
                  conv1_weights_file_sha256=WEIGHTS_SHA,
                  physical_pulse_duration_s=None,downstream_lif='digital',cpu_fill_allowed=False)
        status='failed'
        try:
            if not self.connector.open_port():
                raise ConnectionError('Connector.open_port failed')
            for self.stage in stages:
                print('CIFAR stage '+str(self.stage+1)+'/'+str(len(stages))+' programming',flush=True)
                self.emit('stage_begin')
                self.program_stage(weights)
                matrix=self.calibrate()
                for count,(y,x) in enumerate(positions):
                    self.collect_position(image,y,x,matrix)
                    if count%16==0:print('CIFAR stage '+str(self.stage+1)+' position '+str(count+1)+'/'+str(len(positions)),flush=True)
                n=plan.RANGES[self.stage%3][1]-plan.RANGES[self.stage%3][0]
                for index,mask in enumerate([[]]+[[b] for b in range(n)]):
                    for repeat in range(3):
                        self.vector(mask,'final_basis',basis_index=index,repeat=repeat)
                self.emit('stage_complete')
            status='completed'
        finally:
            try:
                if not self.connector.close_port():
                    status='failed'
                    raise ConnectionError('Connector.close_port failed')
            finally:
                try:
                    self.emit('session_complete',status=status,scope=scope,impact_attempts=self.seq,
                              programming_pulses=self.pulses,full_classifier_accuracy_measured=False)
                finally:
                    self.writer.finish()
        return dict(status=status,scope=scope,impacts=self.seq,pulses=self.pulses)
