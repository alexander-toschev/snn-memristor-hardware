"""Independent native-command/LIF replay; no Collector or Connector imports."""
from collections import Counter
from experiments.mnist_diagonal_20261020.replay import Replay as DiagonalReplay
from tools.audit_mnist_conductance import require,same,GAINS
from experiments.mnist_accuracy_20261020.inputs import verify,GROUPS,impact_count


class Replay(DiagonalReplay):
    def __init__(self,records,fixture,source,gain,parent):
        require(source in ('hardware','simulator'),'Unknown source')
        require(len(gain)==8 and gain[:4]==gain[4:] and all(g in GAINS for g in gain),'Invalid gain')
        self.rows=iter(records);self.source=source;self.reference=None;self.parent=parent
        self.schedules=verify(fixture);self.images={r['train_index']:r for r in fixture['images']}
        self.calls=self.pulses=self.decisions=self.ordinal=self.records=0
        self.last_monotonic=None;self.gain=list(gain);self.calibration=None
        self.updates=[];self.outcomes=[];self.checkpoints=[]
        self.participation=[[0]*32 for _ in range(8)]
        self.input_events={phase:[0]*32 for phase in GROUPS}
        self.mode_counts=Counter();self.pulse_counts=Counter();self.max_mvm_current=0.
        self.exact_impacts=impact_count(fixture)

    def call(self,command,phase,**metadata):
        require(self.calls<self.exact_impacts,'Exact read budget exceeded')
        require(command['mode_flag'] in (7,10) and (command['mode_flag']!=7 or command['vol']==0),'Write forbidden')
        return super().call(command,phase,**metadata)

    def pulse(self,*args,**kwargs):raise ValueError('Programming prohibited')

    def summary(self,status):
        return dict(status=status,impacts=self.calls,pulses=self.pulses,decisions=self.decisions,
            measured_images=self.ordinal,gain=self.gain,outcomes=self.outcomes,
            checkpoints=self.checkpoints,image_cell_participation=self.participation,
            input_event_counts=self.input_events,updates=[],classification_accuracy_measured=False)

    def run(self,protocol):
        self.expect('session_begin',protocol=protocol,parent=self.parent,gain=self.gain)
        opened=self.take('port_open')
        require(set(self.payload(opened))=={'result'} and bool(opened['result']),'Open not confirmed')
        self.checkpoint('start')
        for phase,indices in GROUPS.items():
            for i in indices:self.image(phase,i)
        self.checkpoint('end')
        closed=self.take('port_close')
        require(set(self.payload(closed))=={'result'} and bool(closed['result']),'Close not confirmed')
        self.expect('session_end',status='completed',error=None,impacts=self.calls,pulses=0,decisions=0)
        require(next(self.rows,None) is None,'Unexpected trailing record')
        require(self.calls==self.exact_impacts and self.ordinal==228 and len(self.checkpoints)==2,'Coverage mismatch')
        return self.summary('completed')
