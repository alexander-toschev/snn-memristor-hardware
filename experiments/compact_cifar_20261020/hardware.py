"""Compact physical executor. Reuses public-Connector guards; never supplies CPU currents."""
import math
import numpy as np
from experiments.stable_cell_calibration_20261020 import core as io, metrics, plan as calibration


def stage_layout(pairs,bls):
    if (not pairs or not bls or len({w for p in pairs for w in p})!=2*len(pairs)
        or any(len(p)!=2 or any(type(w)is not int or not 0<=w<8 for w in p) for p in pairs)
        or sorted(set(bls))!=bls or any(type(b)is not int or not 0<=b<32 for b in bls)):
        raise ValueError('Invalid stable rectangle')
    stages=[]
    for start in range(0,8,len(pairs)):
        neurons=list(range(start,min(start+len(pairs),8)))
        for low in range(0,48,len(bls)):
            inputs=list(range(low,min(low+len(bls),48)))
            stages.append(dict(stage=len(stages),neurons=neurons,pairs=pairs[:len(neurons)],
                               inputs=inputs,bls=bls[:len(inputs)]))
    coordinates=[(j,k) for s in stages for j in s['neurons'] for k in s['inputs']]
    if sorted(coordinates)!=[(j,k) for j in range(8) for k in range(48)]:
        raise ValueError('Mapping must cover all384 coefficients exactly once')
    return stages


def input_masks(times,stage,t):
    active=[bl for k,bl in zip(stage['inputs'],stage['bls']) if times[k]==t]
    return [active[i:i+4] for i in range(0,len(active),4)] or [[]]


def bounds(stages,times):
    # Every coefficient:16 RESET+8 SET decisions,guard/write each;9 feedbacks×3 repeats×2WL×4calls.
    programming=384*(48+9*3*2*4)
    inference=cal=retention=0
    for s in stages:
        d=2*len(s['pairs']);k=len(s['bls'])
        basis=d*(9+12*k)
        masks=calibration.masks(s['bls'])
        cal+=basis+sum(3*d*(3+len(mask)) for group in masks.values() for mask in group)
        retention+=5*basis  # After each20 images, including final100.
        inference+=sum(d*(3+len(mask)) for sample in times for t in range(12)
                       for mask in input_masks(sample,s,t))
    return dict(maximum_impacts=programming+inference+cal+retention,maximum_pulses=9216,
                programming_maximum=programming,inference_exact=inference,
                calibration_maximum=cal,retention_exact=retention,soft_deadline_s=14400)


def select_gain(map_value,weights):
    selected=map_value['selection']['selected']
    if selected is None:raise ValueError('No stable or diagnostic selected rectangle')
    q=map_value['pair_qualification']
    if not q or not q['mandatory_pass']:raise ValueError('Both measured signed orientations required')
    lower=[]
    for cell in q['cells']:
        for sign,name in ((1,'plus'),(-1,'minus')):
            for role in ('immediate','delayed'):
                s=cell['configurations'][name][role]
                lower.append(sign*s['mean']-3*s['sd']/math.sqrt(s['count']))
    voltage=min(lower)*5/16383
    maximum=float(np.max(np.abs(weights)))
    if not math.isfinite(voltage) or voltage<=0 or maximum<=0:
        raise ValueError('No resolved signed dynamic range')
    return .8*voltage/maximum


class Collector(io.Collector):
    def __init__(self,connector,converters,writer,source,protocol,weights,times,**kwargs):
        super().__init__(connector,converters,writer,source,**kwargs)
        self.p=protocol;self.weights=weights;self.times=times;self.outputs=[]
        self.gain=protocol['gain_v_per_weight']

    def budget(self):
        if self.clock()-self.started>=self.p['limits']['soft_deadline_s']:
            raise TimeoutError('Finite compact acquisition deadline')
        if self.calls>=self.p['limits']['maximum_impacts']:raise ValueError('Compact impact cap')

    def measure(self,stage,mask,role,**meta):
        wls=[w for p in stage['pairs'] for w in p]
        rows=[self.bracket(w,mask,stage=stage['stage'],operation=role,**meta) for w in wls]
        if any(x['status']!='measured' for x in rows):raise ValueError('Missing physical vector; no CPU fill')
        return dict(mask=mask,centered_v=[x['centered_v'] for x in rows],
                    signal_task_ids=[x['signal']['task_id'] for x in rows])

    def feedback(self,stage,pair,bl,**meta):
        small={**stage,'pairs':[pair]}
        values=[self.measure(small,[bl],'program_feedback',repeat=r,**meta) for r in range(3)]
        return sum(r['centered_v'][0]-r['centered_v'][1] for r in values)/3

    def program(self,stage):
        self.phase='B'
        for neuron,pair in zip(stage['neurons'],stage['pairs']):
            for k,bl in zip(stage['inputs'],stage['bls']):
                target=float(self.weights[neuron,k])*self.gain
                meta=dict(stage=stage['stage'],neuron=neuron,input=k,target_v=target)
                for wl in pair:
                    for step,level in enumerate(calibration.LEVELS):
                        self.pulse(wl,bl,0,level,step=step,**meta)
                value=self.feedback(stage,pair,bl,neuron=neuron,input=k)
                tolerance=max(.00061,.05*abs(target));direction=1 if target>=0 else -1
                for step,level in enumerate(calibration.LEVELS):
                    if abs(value-target)<=tolerance or direction*value>direction*target:break
                    self.pulse(pair[0 if target>=0 else 1],bl,1,level,step=step,**meta)
                    value=self.feedback(stage,pair,bl,neuron=neuron,input=k)
                self.emit('coefficient_complete',**meta,actual_v=value,
                          within_tolerance=abs(value-target)<=tolerance,approximate_is_valid_result=True)
                self.durable(stage=stage['stage'],neuron=neuron,input=k)

    def singleton_basis(self,stage,role):
        result={}
        for mask in [[]]+[[b] for b in stage['bls']]:
            rows=[self.measure(stage,mask,role,repeat=r) for r in range(3)]
            if mask:result[mask[0]]=np.mean([r['centered_v'] for r in rows],axis=0).tolist()
        self.emit('compact_basis_complete',stage=stage['stage'],role=role,basis=result)
        return result

    def calibrate(self,stage):
        self.phase='C'
        basis=self.singleton_basis(stage,'basis')
        masks=calibration.masks(stage['bls'])
        self.emit('compact_mask_split',stage=stage['stage'],masks=masks)
        rows={}
        for split in ('fit','selection'):
            rows[split]=[self.measure(stage,m,split,mask_index=i,repeat=r)
                         for i,m in enumerate(masks[split]) for r in range(3)]
        if not rows['fit'] or not rows['selection']:
            self.emit('compact_matrix_unavailable',stage=stage['stage'],reason='insufficient_disjoint_masks')
            return None
        matrix=metrics.fit_matrix(basis,rows['fit'],rows['selection'])
        self.emit('compact_matrix_frozen',stage=stage['stage'],matrix=matrix,gain_v_per_weight=self.gain)
        held=[self.measure(stage,m,'heldout',mask_index=i,repeat=r)
              for i,m in enumerate(masks['heldout']) for r in range(3)]
        wls=[w for p in stage['pairs'] for w in p]
        validation=metrics.heldout(matrix,basis,held,stage['pairs'],wls)
        self.emit('compact_heldout_complete',stage=stage['stage'],validation=validation)
        return np.asarray(matrix['matrix'])

    def collect_stage(self,stage,matrix):
        raw=np.full((100,12,len(stage['pairs'])),np.nan,dtype=np.float64)
        corrected=np.full_like(raw,np.nan) if matrix is not None else None
        for image,times in enumerate(self.times):
            for t in range(12):
                observed=[];calibrated=[]
                for micro,mask in enumerate(input_masks(times,stage,t)):
                    row=self.measure(stage,mask,'inference',image=image,t=t,micro=micro)
                    x=np.asarray(row['centered_v']);observed.append(x[0::2]-x[1::2])
                    if matrix is not None:
                        y=matrix@x;calibrated.append(y[0::2]-y[1::2])
                    self.emit('compact_inference_contribution',stage=stage['stage'],image=image,t=t,
                              micro=micro,observation=row)
                raw[image,t]=np.sum(observed,axis=0)/self.gain
                if corrected is not None:corrected[image,t]=np.sum(calibrated,axis=0)/self.gain
            self.durable(stage=stage['stage'],image=image)
            if (image+1)%20==0:self.singleton_basis(stage,'retention_after_'+str(image+1))
        result=dict(stage=stage['stage'],neurons=stage['neurons'],inputs=stage['inputs'],
                    raw=raw.tolist(),calibrated=None if corrected is None else corrected.tolist())
        self.emit('compact_stage_complete',stage=stage['stage'],samples=100)
        self.outputs.append(result)

    def run_compact(self):
        self.emit('compact_session_begin',protocol=self.p,protocol_sha256=calibration.digest(self.p))
        self.status='failed'
        try:
            if not self.connector.open_port():raise ConnectionError('Connector open failed')
            for stage in self.p['stages']:
                self.emit('compact_stage_begin',stage=stage)
                self.program(stage);matrix=self.calibrate(stage);self.collect_stage(stage,matrix)
            if self.pulses>9216:raise ValueError('Pulse cap mismatch')
            self.status='completed'
        except (Exception,KeyboardInterrupt) as exc:self.error=type(exc).__name__+': '+str(exc)
        finally:
            try:
                if not self.connector.close_port():raise ConnectionError('Connector close failed')
            except Exception as exc:self.status='failed';self.error=repr(exc)
            self.emit('compact_session_end',status=self.status,error=self.error,impacts=self.calls,pulses=self.pulses)
        return dict(status=self.status,error=self.error,impacts=self.calls,pulses=self.pulses,
                    completed_stages=len(self.outputs),stages=self.outputs,source=self.source,
                    classification_performed=False)


def aggregate(protocol,outputs,branch):
    if len(outputs)!=len(protocol['stages']):raise ValueError('Partial physical acquisition')
    currents=np.zeros((100,12,8),dtype=np.float32);coverage=np.zeros((8,48),dtype=np.uint8)
    for expected,row in zip(protocol['stages'],outputs):
        if any(row[k]!=expected[k] for k in ('stage','neurons','inputs')):raise ValueError('Wrong tile coverage/order')
        if row[branch] is None:raise ValueError('Calibrated branch unavailable; do not replace with raw')
        values=np.asarray(row[branch],dtype=np.float32)
        if values.shape!=(100,12,len(expected['neurons'])) or not np.isfinite(values).all():
            raise ValueError('Missing/invalid measured contributions')
        for i,j in enumerate(expected['neurons']):
            currents[:,:,j]+=values[:,:,i]
            for k in expected['inputs']:coverage[j,k]+=1
    if not (coverage==1).all():raise ValueError('Missing/duplicated coefficient tile')
    return currents


def readout(currents,A,b):
    if currents.shape!=(100,12,8) or not np.isfinite(currents).all():raise ValueError('Complete measured currents required')
    v=np.zeros((100,8),dtype=np.float32);spikes=[]
    for t in range(12):
        u=.5*v+currents[:,t];z=(u>=1).astype(np.float32);v=(1-z)*u;spikes.append(z)
    spikes=np.stack(spikes,axis=1)
    return spikes.sum(axis=1)/12@A.T+b,spikes
