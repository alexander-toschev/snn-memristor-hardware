"""A/B/C scheduler. Device access is exclusively public Connector methods."""
import contextlib
from datetime import datetime, timezone
import io
import math
import time
from . import metrics, plan


def safe(value):
    if value is None or type(value) in (str,int,bool):return value
    if type(value) is float:return value if math.isfinite(value) else repr(value)
    if isinstance(value,(list,tuple)):return [safe(v) for v in value]
    return repr(value)


class DriverOutput(io.TextIOBase):
    def __init__(self,owner,channel):self.owner,self.channel=owner,channel
    def write(self,text):
        if text:self.owner.emit('driver_output',channel=self.channel,text=text,task_id=self.owner.calls)
        return len(text)
    def flush(self):pass


class Collector:
    def __init__(self,connector,converters,writer,source,monotonic=time.monotonic,sleep=time.sleep):
        if source not in ('simulator','hardware'):raise ValueError('Unknown source')
        self.connector,self.cv,self.writer,self.source=connector,converters,writer,source
        self.clock,self.sleep=monotonic,sleep
        self.calls=self.pulses=self.decisions=0
        self.phase='A';self.phase_calls=dict(A=0,B=0,C=0);self.phase_pulses=dict(A=0,B=0,C=0)
        self.started=self.clock();self.waited=0.;self.last_decision=None
        self.checkpoints=[];self.updates=[];self.pair_observations=[]
        self.cells=[];self.selection=None;self.pair_qualification=None;self.calibration={}
        self.status='failed';self.error=None;self.lifecycle_assumed=False

    def emit(self,event,**fields):
        self.writer.write(dict(event=event,source=self.source,phase=self.phase,
            utc=datetime.now(timezone.utc).isoformat(),monotonic_s=self.clock(),**fields))

    def budget(self):
        if self.clock()-self.started>=3600:raise TimeoutError('Soft acquisition deadline reached')
        if self.calls>=411536 or self.phase_calls[self.phase]>=plan.SPEC['phase_'+self.phase.lower()+'_maximum_impacts']:
            raise ValueError('Impact budget exhausted')

    def durable(self,**fields):
        self.writer.checkpoint(phase=self.phase,impacts=self.calls,pulses=self.pulses,**fields)

    def call(self,command,**metadata):
        self.budget()
        self.calls+=1;self.phase_calls[self.phase]+=1
        task={**command,'id':self.calls if command['mode_flag']==7 else 0}
        fields=dict(task_id=self.calls,command_si=task,metadata=metadata)
        self.emit('request',**fields);started=self.clock()
        try:
            with contextlib.redirect_stdout(DriverOutput(self,'stdout')),contextlib.redirect_stderr(DriverOutput(self,'stderr')):
                result=self.connector.impact({**task,'vol':list(task['vol']) if isinstance(task['vol'],list) else task['vol']})
            self.emit('result',**fields,native_result=safe(result),latency_s=self.clock()-started)
            if not isinstance(result,(list,tuple)) or len(result)!=3:raise ValueError('Malformed native result')
            r,echo,adc=result
            if type(echo) is not int or echo!=task['id'] or type(adc) is not int:raise ValueError('Invalid native ID/ADC type')
            if task['mode_flag']==7:
                if type(r) not in (float,int) or not math.isfinite(r) or r<=1e-8 or not 20<=adc<=16382:
                    raise ValueError('Invalid native R/ADC')
            elif not 0<=adc<=16382:raise ValueError('Invalid MVM ADC')
            return dict(task_id=self.calls,resistance_ohm=r,adc=adc)
        except (Exception,KeyboardInterrupt) as exc:
            self.emit('call_error',task_id=self.calls,error=type(exc).__name__+': '+str(exc));raise

    def read_r(self,wl,bl,**metadata):
        if type(wl) is not int or type(bl) is not int or not 0<=wl<8 or not 0<=bl<32:
            raise ValueError('Invalid native address')
        return self.call(dict(mode_flag=7,wl=wl,bl=bl,vol=0.,sign=0,pulse_width=0.),**metadata)

    def pulse(self,wl,bl,sign,level,**metadata):
        if self.phase not in ('A','B') or sign not in (0,1) or level not in plan.LEVELS:raise ValueError('Unreviewed programming decision')
        if self.decisions>=20480:raise ValueError('Programming decision budget exhausted')
        guard=self.read_r(wl,bl,role='write_guard',sign=sign,level=level,**metadata)
        self.decisions+=1
        predicted=level/guard['resistance_ohm'];admitted=predicted<=(.002 if sign else .04)
        record=dict(phase=self.phase,wl=wl,bl=bl,sign=sign,level=level,guard=guard,
                    predicted_current_a=predicted,admitted=admitted,decision=self.decisions,**metadata)
        self.emit('pulse_decision',record=record)
        self.last_decision=self.clock()
        if admitted:
            if self.pulses>=20480 or self.phase_pulses[self.phase]>=plan.SPEC['phase_'+self.phase.lower()+'_maximum_pulses']:
                raise ValueError('Pulse budget exhausted')
            self.pulses+=1;self.phase_pulses[self.phase]+=1
            record['response']=self.call(dict(mode_flag=7,wl=wl,bl=bl,vol=level,sign=sign,
                pulse_width=.000125 if self.source=='simulator' else 0.),role='write',**metadata)
        self.updates.append(record);self.emit('pulse_complete',record=record)

    def vector(self,wl,mask,**metadata):
        if type(wl) is not int or not 0<=wl<8 or len(mask)>4 or sorted(set(mask))!=mask or any(type(b) is not int or not 0<=b<32 for b in mask):
            raise ValueError('Invalid MVM vector')
        guards=[self.read_r(wl,b,role='vector_guard',mask=mask,**metadata) for b in mask]
        code=self.cv.convert_volt_to_dac(12,5,.05)
        voltage=max(.05,self.cv.convert_dac_to_volt(12,5,code))
        current=sum(voltage/r['resistance_ohm'] for r in guards)
        if current>.005 or 330*current>=3.3:
            self.emit('vector_skipped',wl=wl,mask=mask,guards=guards,predicted_current_a=current,
                      reason='native_r_current_or_output_guard',metadata=metadata)
            return None
        response=self.call(dict(mode_flag=10,wl=wl,vol=[.05 if b in mask else 0. for b in range(32)]),
                           role='mvm',mask=mask,**metadata)
        v=self.cv.convert_adc_to_volt(1,14,5,response['adc'])
        if not math.isfinite(v) or v>4.:raise ValueError('Measured ADC headroom exceeded')
        return dict(**response,voltage_v=v,guards=guards,predicted_current_a=current)

    def bracket(self,wl,mask,**metadata):
        before=self.vector(wl,[],bracket_position='before',**metadata)
        signal=self.vector(wl,mask,bracket_position='signal',**metadata)
        after=self.vector(wl,[],bracket_position='after',**metadata)
        result=dict(wl=wl,mask=mask,metadata=metadata,before=before,signal=signal,after=after,
                    status='measured' if signal is not None else 'guard_skipped')
        if signal is not None:
            result.update(centered_adc=signal['adc']-(before['adc']+after['adc'])/2,
                centered_v=signal['voltage_v']-(before['voltage_v']+after['voltage_v'])/2,
                before_adc=before['adc'],after_adc=after['adc'],
                resistance_ohm=signal['guards'][0]['resistance_ohm'] if len(mask)==1 else None)
        self.emit('bracket_complete',bracket=result)
        return result

    def checkpoint(self,wl,bl,cycle,state,role):
        identifier=len(self.checkpoints)+1
        rows=[self.bracket(wl,[bl],checkpoint=identifier,cycle=cycle,state=state,
                           checkpoint_role=role,repeat=r) for r in range(10)]
        record=dict(id=identifier,wl=wl,bl=bl,cycle=cycle,state=state,role=role,
                    monotonic_s=self.clock(),brackets=rows,summary=metrics.checkpoint(rows),
                    bracket_signal_task_ids=[r['signal']['task_id'] if r['signal'] else None for r in rows])
        self.checkpoints.append(record);self.emit('checkpoint_complete',checkpoint=record)
        self.durable(checkpoint=identifier,wl=wl,bl=bl)

    def wait_until(self,origin,delay):
        remaining=max(0.,origin+delay-self.clock())
        self.budget()
        if self.waited+remaining>600.+1e-7:raise ValueError('Wait budget exhausted')
        if self.clock()-self.started+remaining>=3600:raise TimeoutError('Wait would cross soft deadline')
        self.emit('wait_begin',origin_monotonic_s=origin,minimum_delay_s=delay,requested_wait_s=remaining)
        before=self.clock()
        if remaining:self.sleep(remaining)
        self.waited+=remaining
        self.emit('wait_end',elapsed_s=self.clock()-before,actual_since_origin_s=self.clock()-origin)

    def phase_a(self):
        for wl,bl in plan.CELLS:self.checkpoint(wl,bl,-1,'initial','baseline')
        for cycle in range(2):
            for state in ('R','M','H'):
                for wl,bl in plan.CELLS:
                    for step,(sign,level) in enumerate(plan.recipe(state,incremental=True)):
                        self.pulse(wl,bl,sign,level,cycle=cycle,state=state,step=step)
                    self.checkpoint(wl,bl,cycle,state,'immediate')
                origin=self.last_decision
                for delay in (0,10,60):
                    self.wait_until(origin,delay)
                    for wl,bl in plan.CELLS:self.checkpoint(wl,bl,cycle,state,'delay_'+str(delay))
                self.emit('state_complete',cycle=cycle,state=state)
        for wl,bl in plan.CELLS:self.checkpoint(wl,bl,-1,'H','final')
        self.emit('phase_a_complete',checkpoint_count=len(self.checkpoints))
        self.cells=[metrics.cell_summary(w,b,self.checkpoints,self.updates) for w,b in plan.CELLS]
        self.selection=metrics.select_rectangle(self.cells)
        self.emit('selection_frozen',selection=self.selection,cells=self.cells)
        self.durable(selection_sha256=self.selection['selection_sha256'])

    def paired_checkpoint(self,pair,bl,configuration,role):
        rows=[[self.bracket(w,[bl],configuration=configuration,pair=pair,paired_role=role,repeat=r)
               for w in pair] for r in range(10)]
        measured=all(x['status']=='measured' for row in rows for x in row)
        record=dict(pair=pair,bl=bl,configuration=configuration,role=role,
                    status='measured' if measured else 'guard_skipped',
                    differences_adc=[row[0]['centered_adc']-row[1]['centered_adc'] for row in rows] if measured else [],
                    signal_task_ids=[[x['signal']['task_id'] if x['signal'] else None for x in row] for row in rows],
                    monotonic_s=self.clock())
        self.pair_observations.append(record);self.emit('paired_checkpoint_complete',checkpoint=record)
        self.durable(pair=pair,bl=bl,configuration=configuration,role=role)

    def phase_b(self):
        self.phase='B';selected=self.selection['selected']
        for config in ('plus','minus','zero','checkerboard'):
            for pair in selected['pairs']:
                for index,bl in enumerate(selected['bls']):
                    for wl,state in zip(pair,plan.pair_states(config,index)):
                        for step,(sign,level) in enumerate(plan.recipe(state)):
                            self.pulse(wl,bl,sign,level,configuration=config,pair=pair,state=state,step=step)
                    self.paired_checkpoint(pair,bl,config,'immediate')
            self.wait_until(self.last_decision,60)
            for pair in selected['pairs']:
                for bl in selected['bls']:self.paired_checkpoint(pair,bl,config,'delayed')
        self.emit('phase_b_measurements_complete',selection_sha256=self.selection['selection_sha256'])
        self.pair_qualification=metrics.qualify_pairs(selected,self.cells,self.pair_observations)
        self.emit('pairs_qualified',qualification=self.pair_qualification)
        self.durable(phase_b_qualified=self.pair_qualification['qualified'])

    def basis(self,selected,role):
        wls=sorted(w for p in selected['pairs'] for w in p);observations=[]
        for mask in [[]]+[[b] for b in selected['bls']]:
            for repeat in range(3):
                row=[self.bracket(w,mask,basis_role=role,repeat=repeat) for w in wls]
                observations.append(dict(mask=mask,repeat=repeat,brackets=row))
        self.emit('basis_measurements_complete',role=role,wls=wls,observations=observations)
        self.durable(basis_role=role)
        return observations

    def mask_rows(self,selected,phase,masks):
        wls=sorted(w for pair in selected['pairs'] for w in pair);observations=[]
        for index,mask in enumerate(masks):
            for repeat in range(3):
                row=[self.bracket(w,mask,calibration_phase=phase,mask_index=index,repeat=repeat) for w in wls]
                observation=dict(mask=mask,repeat=repeat,index=index,brackets=row)
                observations.append(observation)
                self.emit('calibration_vector_complete',calibration_phase=phase,observation=observation)
            self.durable(calibration_phase=phase,mask_index=index)
        return observations

    def phase_c(self):
        self.phase='C';selected=self.selection['selected'];masks=plan.masks(selected['bls'])
        self.emit('mask_schedule_frozen',masks=masks,mask_sha256=plan.digest(masks))
        self.calibration['before']=self.basis(selected,'before')
        for split in ('fit','selection'):
            self.calibration[split]=self.mask_rows(selected,split,masks[split])
        try:
            basis=metrics.basis_values(self.calibration['before'])
            fit=metrics.vector_values(self.calibration['fit'])
            selection=metrics.vector_values(self.calibration['selection'])
            if not fit or not selection:raise ValueError('Insufficient disjoint masks')
            matrix=metrics.fit_matrix(basis,fit,selection)
        except ValueError as exc:
            self.calibration.update(status='unavailable',reason=str(exc))
            self.calibration['after']=self.basis(selected,'after')
            return
        self.emit('matrix_frozen',matrix=matrix,matrix_sha256=plan.digest(matrix))
        self.calibration['heldout']=self.mask_rows(selected,'heldout',masks['heldout'])
        self.calibration['after']=self.basis(selected,'after')
        try:
            held=metrics.vector_values(self.calibration['heldout'])
            metrics.basis_values(self.calibration['after'])
            validation=metrics.heldout(matrix,basis,held,selected['pairs'],sorted(w for p in selected['pairs'] for w in p))
            self.calibration.update(status='qualified' if validation['qualified'] else 'joint_unqualified',
                                    matrix=matrix,validation=validation)
        except ValueError as exc:self.calibration.update(status='unavailable',reason=str(exc),matrix=matrix)
        self.emit('calibration_complete',summary={k:v for k,v in self.calibration.items()
                  if k not in ('before','fit','selection','heldout','after')})

    def result(self):
        by={(c['wl'],c['bl']):c for c in self.cells}
        cells=[by.get((w,b),dict(wl=w,bl=b,eligible=False,reasons=['missing_or_interrupted_phase_A']))
               for w,b in plan.CELLS]
        return dict(status=self.status,error=self.error,impacts=self.calls,pulses=self.pulses,decisions=self.decisions,
            phase_impacts=self.phase_calls,phase_pulses=self.phase_pulses,explicit_wait_requested_s=self.waited,
            checkpoints=self.checkpoints,updates=self.updates,cells=cells,selection=self.selection,
            pair_qualification=self.pair_qualification,pair_observations=self.pair_observations,
            calibration={k:v for k,v in self.calibration.items() if k not in ('before','fit','selection','heldout','after')},
            qualified=self.status=='completed' and self.calibration.get('status')=='qualified'
                      and bool(self.selection and self.selection.get('quality_qualified'))
                      and bool(self.pair_qualification and self.pair_qualification['qualified']),
            diagnostic_best_available=self.status=='completed' and not (
                self.selection.get('quality_qualified') and self.pair_qualification['qualified']
                and self.calibration.get('status')=='qualified'),
            physical_pulse_duration_s=None,private_runtime_status=plan.SPEC['private_runtime_status'],
            cifar_accuracy_measured=False,initial_array_descriptive_only=True)

    def run(self):
        self.emit('session_begin',protocol=plan.SPEC,protocol_sha256=plan.digest(plan.SPEC))
        self.lifecycle_assumed=True
        try:
            opened=self.connector.open_port();self.emit('port_open',result=safe(opened))
            if not opened:raise ConnectionError('Connector open failed')
            self.phase_a()
            if self.selection['selected'] is None:self.status='no_stable_rectangle'
            else:
                self.phase_b()
                if not self.pair_qualification['mandatory_pass']:self.status='pair_qualification_failed'
                else:
                    self.phase_c();self.status='completed'
        except (Exception,KeyboardInterrupt) as exc:
            self.status='failed';self.error=type(exc).__name__+': '+str(exc)
        finally:
            try:
                closed=self.connector.close_port();self.emit('port_close',result=safe(closed))
                if not closed:raise ConnectionError('Connector close failed')
            except Exception as exc:
                self.status='failed';self.error='close: '+type(exc).__name__+': '+str(exc)
            self.emit('session_end',status=self.status,error=self.error,impacts=self.calls,pulses=self.pulses,
                      phase_impacts=self.phase_calls)
        return self.result()
