import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'.deps')]
import numpy as np
from experiments.mnist64_20261020 import core,inputs,logic,replay,readout
from tests.test_audit_mnist_conductance import FakeConnector,FakeConverters
from tests.test_mnist_diagonal import Sink

class Tests(unittest.TestCase):
    def fixture(self,s):return json.loads((inputs.HERE/('fixture_'+s+'.json')).read_text())
    def test_mapping_budget_and_sum_before_lif(self):
        self.assertEqual(len({logic.address(n,f) for n in range(4) for f in range(64)}),256)
        self.assertEqual(logic.address(3,63),(7,31))
        self.assertEqual(sum(inputs.budget(self.fixture(s))['maximum_impacts'] for s in ('train','accuracy')),104912)
        schedule=[[] for _ in range(32)];schedule[0]=[0,32];v=[.012]+[0.]*31
        a,c=logic.trajectory(schedule,v,100);b,d=replay.trajectory(schedule,v,100)
        self.assertEqual(a,b);self.assertEqual(c,d);self.assertEqual(a[0]['post'],1)
        self.assertEqual(sum(logic.trajectory(schedule,[.006]+[0.]*31,100)[0][i]['post'] for i in range(32)),0)
    def test_image_native_replay_and_mutation(self):
        f=self.fixture('accuracy');sink=Sink();r=core.Collector(FakeConnector(),FakeConverters,sink,'simulator',f,[100.]*4,{'session_id':'fake'})
        r.image('readout_train',0);rows=json.loads(json.dumps(sink.events))
        q=replay.Replay(iter(rows),f,'simulator',r.gain,r.parent);q.image('readout_train',0)
        self.assertEqual(q.outcomes,r.outcomes);self.assertEqual(r.pulses,0)
        self.assertEqual(len([e for e in rows if e['event']=='lif_step']),128)
        with self.assertRaises(ValueError):r.pulse(0,0,1,'training_update')
        bad=copy.deepcopy(rows)
        next(e for e in bad if e['event']=='lif_step')['post']^=1
        with self.assertRaises(ValueError):replay.Replay(iter(bad),f,'simulator',r.gain,r.parent).image('readout_train',0)
    def test_upper_plane_training_address_and_replay(self):
        f=self.fixture('train');sink=Sink();r=core.Collector(FakeConnector(),FakeConverters,sink,'simulator',f,[100.]*4)
        r.pulse(*logic.address(2,50),1,'training_update',neuron=2,feature=50,eligibility=.5)
        q=replay.Replay(iter(json.loads(json.dumps(sink.events))),f,'simulator',[100.]*4)
        q.pulse(6,18,1,'training_update',neuron=2,feature=50,eligibility=.5)
        self.assertEqual(q.updates,r.updates);self.assertEqual(r.pulses,1)
    def test_labels_open_after_model_saved(self):
        outcomes=[dict(phase=p,train_index=i,neurons=[dict(posts=[int((i+n+t)%7==0) for t in range(32)]) for n in range(4)]) for p,ids in inputs.ACCURACY_GROUPS.items() for i in ids]
        with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as temp:
            output=Path(temp);calls=[]
            def labels(split):
                calls.append(split)
                if split=='test':self.assertTrue((output/'READOUT.json').exists())
                return [i%10 for i in range(128 if split=='train' else 100)]
            with patch.object(inputs,'labels',labels):result=readout.classify({'outcomes':outcomes},output)
            self.assertEqual(calls,['train','test']);self.assertEqual(result['total'],100)
            self.assertFalse(result['separate_control_bank'])

if __name__=='__main__':unittest.main()
