"""Read-only frozen post-STDP features; native I/O inherited from measured pilot."""
import time
from experiments.mnist_diagonal_20261020.core import Collector as DiagonalCollector
from experiments.mnist_conductance_20261019.core import safe, GAINS
from experiments.mnist_accuracy_20261020.inputs import GROUPS, verify, impact_count

SPEC = dict(schema='mnist-frozen-post-stdp-accuracy128-100-v1',
    groups=GROUPS, shape=[32,8], timesteps=32, amplitude_v=.05, maximum_mask=4,
    lif_leak=.5, lif_threshold=1., gain='parent diagonal, unchanged, paired control',
    maximum_impacts=128064, maximum_pulses=0, checkpoints=['start','end'],
    checkpoint_calls=3840, labels_in_acquisition=False, updates=False,
    readout=dict(primary='WL0..3 spike raster,128 features',control='WL4..7 spike raster,128 features',
        standardization='train mean/population std; zero std replaced with1',
        model='one-hot10 linear ridge lambda1, unpenalized bias',lambda_value=1.,
        primary_metric='all100 denominator; silent raster or tied top score abstains and counts incorrect',
        tie_absolute_tolerance=1e-12, conventional_argmax_secondary=True,
        test='official test0..99 prefix; no tuning; Wilson95 descriptive pilot interval'))


def summary(owner,status):
    return dict(status=status,impacts=owner.calls,pulses=owner.pulses,decisions=owner.decisions,
        measured_images=owner.ordinal,gain=owner.gain,outcomes=owner.outcomes,
        checkpoints=owner.checkpoints,image_cell_participation=owner.image_cell_participation,
        input_event_counts=owner.input_event_counts,updates=[],classification_accuracy_measured=False)


class Collector(DiagonalCollector):
    def __init__(self,connector,converters,writer,source,fixture,gain,parent):
        verify(fixture)
        if source not in ('hardware','simulator') or connector.cb_type != ('real' if source=='hardware' else 'simulator'):
            raise ValueError('Public connector/source mismatch')
        if len(gain)!=8 or gain[:4]!=gain[4:] or any(g not in GAINS for g in gain):
            raise ValueError('Invalid frozen parent gain')
        self.connector,self.converters,self.writer,self.source=connector,converters,writer,source
        self.images={r['train_index']:r for r in fixture['images']}
        self.reference=None;self.parent=parent;self.gain=list(gain);self.calibration=None
        self.sleep=time.sleep;self.monotonic=time.monotonic
        self.calls=self.pulses=self.decisions=self.ordinal=0
        self.outcomes=[];self.updates=[];self.checkpoints=[]
        self.image_cell_participation=[[0]*32 for _ in range(8)]
        self.input_event_counts={phase:[0]*32 for phase in GROUPS}
        self.exact_impacts=impact_count(fixture)

    def call(self,command,phase,**meta):
        if self.calls>=self.exact_impacts: raise ValueError('Read-only exact budget exhausted')
        if command['mode_flag'] not in (7,10) or (command['mode_flag']==7 and command['vol']!=0):
            raise ValueError('Write command forbidden')
        return super().call(command,phase,**meta)

    def pulse(self,*args,**kwargs):
        raise ValueError('All programming/preparation/STDP writes forbidden')

    def summary(self,status): return summary(self,status)

    def run(self):
        status='failed';opened=False;error=None
        try:
            self.emit('session_begin',protocol=SPEC,parent=self.parent,gain=self.gain)
            opened=True;result=self.connector.open_port();self.emit('port_open',result=safe(result))
            if not result: raise ValueError('Public Connector open failed')
            self.checkpoint('start')
            for phase,indices in GROUPS.items():
                for i in indices:self.image(phase,i)
            self.checkpoint('end')
            if self.calls!=self.exact_impacts or self.pulses:raise ValueError('Incomplete read-only acquisition')
            status='completed'
        except (Exception,KeyboardInterrupt) as exc:
            error=repr(exc);raise
        finally:
            try:
                if opened:
                    result=self.connector.close_port();self.emit('port_close',result=safe(result))
                    if not result:status='failed';raise ValueError('Public close failed')
            finally:
                self.emit('session_end',status=status,error=error,impacts=self.calls,pulses=self.pulses,decisions=self.decisions)
                self.writer.finish()
        return self.summary(status)
