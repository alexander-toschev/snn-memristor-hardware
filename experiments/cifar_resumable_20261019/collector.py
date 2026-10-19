"""One fresh bounded stage using unchanged reviewed MemriBoard operation methods."""
import hashlib
import statistics
from experiments.cifar_full_case0_20261019 import collector as original, plan


def stage_bounds(image,stage,scope='full'):
    if type(stage) is not int or stage not in range(48) or scope not in ('full','smoke'):raise ValueError('Stage/scope outside protocol')
    n=(32,32,11)[stage%3];positions=[(y,x) for y in range(28) for x in range(28)] if scope=='full' else [(0,0)]
    signals=sum(8*len(parts) for y,x in positions for parts in plan.masks(image,stage%3,y,x))
    guards=sum(8*len(mask) for y,x in positions for parts in plan.masks(image,stage%3,y,x) for mask in parts)
    masks=original.calibration_masks(n,20261019+stage)
    calibration_mode10=2*(n+1)*3*8+128*3*24
    calibration_guards=2*n*3*8+sum(3*8*len(mask) for rows in masks.values() for mask in rows)
    programming=4*n*(32+16+9*32)
    return dict(stage=stage,maximum_impacts=signals+len(positions)*16+guards+calibration_mode10+calibration_guards+programming,
                maximum_pulses=4*n*24,positions=len(positions),programming_impacts_upper=programming)


class Collector(original.Collector):
    """All impact/current/ADC/programming/calibration methods inherited unchanged."""
    def __init__(self,*args,stage,scope='full',interrupt_after_positions=None,process_loss=False,**kwargs):
        super().__init__(*args,**kwargs)
        if interrupt_after_positions is not None and (self.source!='simulator' or stage!=1 or interrupt_after_positions!=17 or scope!='full'):
            raise ValueError('Controlled interruption is fixed SIM stage1/full/17 only')
        self.interrupt_after_positions=interrupt_after_positions
        if process_loss and interrupt_after_positions!=17:raise ValueError('Process-loss injection requires fixed17position SIM qualification')
        self.process_loss=process_loss
        self.stage_id=stage;self.scope=scope;self.completed_coefficients=0;self.completed_positions=0
        self.bounds=None
        self.lifecycle_assumed=False

    def call(self,command,phase,**meta):
        if self.bounds is None or self.seq>=self.bounds['maximum_impacts']:raise ValueError('Per-stage impact budget exhausted')
        return super().call(command,phase,**meta)

    def pulse(self,*args,**kwargs):
        if self.bounds is None or self.pulses>=self.bounds['maximum_pulses']:raise ValueError('Per-stage programming budget exhausted')
        return super().pulse(*args,**kwargs)

    def emit(self,event,**fields):
        super().emit(event,**fields)
        checkpoint=False
        if event=='coefficient_complete':
            self.completed_coefficients+=1;checkpoint=self.completed_coefficients%16==0
        if event=='position_complete':
            self.completed_positions+=1;checkpoint=self.completed_positions%16==0
        if checkpoint or event in ('programming_complete','calibration_frozen','heldout_complete','stage_complete'):
            self.writer.checkpoint(reason=event,stage=self.stage,impact_attempts=self.seq,programming_pulses=self.pulses,
                completed_coefficients=self.completed_coefficients,completed_positions=self.completed_positions,
                resume_inside_stage_allowed=False)

    def run_stage(self,image,weights):
        import numpy as np
        expected_image,expected_weights=original.load_assets()
        if image!=expected_image or not np.array_equal(np.asarray(weights),expected_weights):raise ValueError('Pinned image/weights mismatch')
        self.bounds=stage_bounds(image,self.stage_id,self.scope)
        # Start marker keeps the original stage=-1 convention; the range is explicit.
        self.emit('session_begin',scope=self.scope,protocol=self.p,protocol_sha256=self.protocol_sha,
            stage_range=[self.stage_id,self.stage_id+1],acquisition_kind='stage_segment',
            image_sha256=hashlib.sha256(bytes(image)).hexdigest(),checkpoint_sha256=plan.CHECKPOINT_SHA,
            conv1_weights_file_sha256=original.WEIGHTS_SHA,physical_pulse_duration_s=None,
            downstream_lif='digital',cpu_fill_allowed=False)
        status='failed'
        self.lifecycle_assumed=True
        try:
            if not self.connector.open_port():raise ConnectionError('Connector.open_port failed')
            self.stage=self.stage_id;self.emit('stage_begin')
            self.program_stage(weights);matrix=self.calibrate()
            positions=[(y,x) for y in range(28) for x in range(28)] if self.scope=='full' else [(0,0)]
            for count,(y,x) in enumerate(positions):
                self.collect_position(image,y,x,matrix)
                if self.interrupt_after_positions==count+1:
                    if self.process_loss:
                        from .durable import simulated_process_loss
                        simulated_process_loss(self.writer,self.source,self.stage,count+1,self.seq)
                    self.writer.checkpoint(reason='controlled_qualification_interruption',stage=self.stage,
                                           completed_positions=count+1,resume_inside_stage_allowed=False)
                    raise RuntimeError('Planned nativeSIM recovery qualification interruption after position17')
                if (count+1)%16==0 or count+1==len(positions):print('CIFAR stage %d positions %d/%d'%(self.stage+1,count+1,len(positions)),flush=True,file=getattr(self,'progress_stream',None))
            n=(32,32,11)[self.stage%3]
            for index,mask in enumerate([[]]+[[b] for b in range(n)]):
                for repeat in range(3):self.vector(mask,'final_basis',basis_index=index,repeat=repeat)
            self.emit('stage_complete');status='completed'
        finally:
            try:
                if not self.connector.close_port():status='failed';raise ConnectionError('Connector.close_port failed')
            finally:
                try:self.emit('session_complete',status=status,scope=self.scope,impact_attempts=self.seq,
                              programming_pulses=self.pulses,full_classifier_accuracy_measured=False)
                finally:self.writer.finish()
        return dict(status=status,stage=self.stage_id,scope=self.scope,impacts=self.seq,pulses=self.pulses,
                    reusable_only_after_independent_semantic_audit=True)
