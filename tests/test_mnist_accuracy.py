"""No native calls: frozen inputs, read-only I/O replay and fixed readout metrics."""
import copy
import json
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'.deps')]
import numpy as np
from experiments.mnist_accuracy_20261020 import core,inputs,replay,analyze,experiment
from experiments.mnist_conductance_20261019.core import Collector as Original
from tests.test_audit_mnist_conductance import FakeConnector,FakeConverters
from tests.test_mnist_diagonal import Sink


class Tests(unittest.TestCase):
    def fixture(self):return json.loads((inputs.HERE/'fixture.json').read_text())

    def test_fixed_inputs_and_parent_final_state(self):
        fixture=self.fixture();inputs.verify(fixture)
        self.assertEqual(inputs.impact_count(fixture),56336)
        self.assertEqual(len(inputs.labels('train')),128);self.assertEqual(len(inputs.labels('test')),100)
        self.assertEqual(fixture['images'][128]['dataset_split'],'test')
        _,_,_,parent,binding=experiment.preflight('simulator')
        self.assertEqual(experiment.sha(parent/'array.cb'),binding['final_array_sha256'])
        self.assertNotEqual(experiment.sha(parent/'array_initial.cb'),binding['final_array_sha256'])
        bad=copy.deepcopy(fixture);bad['images'][128]['dataset_split']='train'
        with self.assertRaises(ValueError):inputs.verify(bad)

    def test_image_replay_no_update_and_mutation_rejection(self):
        f=self.fixture();sink=Sink();parent={'session_id':'fake'}
        runner=core.Collector(FakeConnector(),FakeConverters,sink,'simulator',f,[200,200,200,100]*2,parent)
        for phase,index in [('readout_train',0),('official_test',100000)]:
            runner.image(phase,index)
        self.assertEqual(runner.pulses,0);self.assertEqual(runner.decisions,0)
        rows=json.loads(json.dumps(sink.events))
        checker=replay.Replay(iter(rows),f,'simulator',runner.gain,parent)
        checker.image('readout_train',0);checker.image('official_test',100000)
        self.assertEqual(checker.outcomes,runner.outcomes)
        self.assertIs(core.Collector.vector,Original.vector)
        self.assertIs(core.Collector.checkpoint,Original.checkpoint)
        with self.assertRaises(ValueError):runner.pulse(0,0,1,'training_update')
        with self.assertRaises(ValueError):runner.call(dict(mode_flag=7,vol=1.7),'bad')
        bad=copy.deepcopy(rows)
        next(r for r in bad if r['event']=='lif_step')['post']^=1
        with self.assertRaises(ValueError):
            replay.Replay(iter(bad),f,'simulator',runner.gain,parent).image('readout_train',0)

    def test_ridge_train_only_and_abstention(self):
        x=np.eye(10);y=np.arange(10);model=analyze.fit_ridge(x,y)
        self.assertEqual(analyze.predict(model,x).argmax(axis=1).tolist(),y.tolist())
        saved=copy.deepcopy(model);analyze.predict(model,np.full((3,10),1e6));self.assertEqual(saved,model)
        scores=np.array([[3.,1.],[1.,1.],[0.,3.]])
        features=np.array([[0.,0.],[1.,0.],[0.,1.]])
        result=analyze.evaluate(scores,features,[0,0,1])
        self.assertEqual(result['conventional_argmax_correct'],3)
        self.assertEqual(result['strict_correct'],1)
        self.assertEqual(result['abstained'],2)
        self.assertEqual(result['total'],3)
        self.assertEqual(result['coverage'],1/3)


if __name__=='__main__':unittest.main()
