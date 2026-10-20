import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'.deps')]
import numpy as np
from experiments.compact_cifar_20261020 import hardware,physical_experiment
from tests.test_stable_cell_calibration import Board,Converters,Sink,Clock


class PhysicalTests(unittest.TestCase):
    def test_all384_variable_geometry_and_zero_input_brackets(self):
        for pairs,bls in (([[0,2]],[1,3,8,9,12,14,20,30]),([[0,1],[2,3],[4,5]],list(range(32)))):
            stages=hardware.stage_layout(pairs,bls)
            self.assertEqual(sum(len(s['neurons'])*len(s['inputs']) for s in stages),384)
            self.assertEqual(hardware.input_masks([-1]*48,stages[0],0),[[]])
            limits=hardware.bounds(stages,[[-1]*48]*100)
            self.assertEqual(limits['maximum_pulses'],9216)
            self.assertGreater(limits['maximum_impacts'],limits['programming_maximum'])

    def test_one_real_fake_stage_program_calibrate_all100_and_no_substitution(self):
        stage=dict(stage=0,neurons=[0],pairs=[[0,1]],inputs=[0,1,2,3],bls=[0,1,2,3])
        times=np.full((100,48),-1,dtype=int);times[:,:4]=0
        weights=np.zeros((8,48),dtype=np.float32);weights[0,:4]=[.25,-.25,0,.5]
        protocol=dict(gain_v_per_weight=.005,stages=[stage],
                      limits=hardware.bounds([stage],times.tolist()))
        clock=Clock();sink=Sink()
        c=hardware.Collector(Board(),Converters,sink,'simulator',protocol,weights,times.tolist(),
                             monotonic=clock,sleep=clock.sleep)
        r=c.run_compact()
        self.assertEqual(r['status'],'completed',r['error'])
        self.assertLessEqual(r['impacts'],protocol['limits']['maximum_impacts'])
        self.assertLessEqual(r['pulses'],24*4)
        self.assertEqual(sink.events['compact_inference_contribution'],1200)
        self.assertEqual(sink.events['compact_matrix_frozen'],1)
        self.assertEqual(np.asarray(r['stages'][0]['raw']).shape,(100,12,1))
        with self.assertRaises(ValueError):hardware.aggregate(protocol,r['stages'],'raw')

    def test_aggregation_before_single_lif_and_missing_rejected(self):
        stages=hardware.stage_layout([[0,1],[2,3]],list(range(32)))
        rows=[dict(stage=s['stage'],neurons=s['neurons'],inputs=s['inputs'],
                   raw=np.full((100,12,len(s['neurons'])),.2).tolist(),calibrated=None) for s in stages]
        result=hardware.aggregate(dict(stages=stages),rows,'raw')
        np.testing.assert_allclose(result,.4)
        with self.assertRaises(ValueError):hardware.aggregate(dict(stages=stages),rows,'calibrated')
        with self.assertRaises(ValueError):hardware.aggregate(dict(stages=stages),rows[:-1],'raw')
        rows[0]['raw'][0][0][0]=float('nan')
        with self.assertRaises(ValueError):hardware.aggregate(dict(stages=stages),rows,'raw')

    def test_readout_matches_completed_cpu_reference(self):
        p=ROOT/'results/compact_cifar_20261020/1aa6dcc7-b729-450e-8f95-beb6344bc075'
        physical_experiment.load_inputs(p)
        logits,spikes=hardware.readout(np.load(p/'cpu_currents.npy',allow_pickle=False),
                                      np.load(p/'A.npy',allow_pickle=False),np.load(p/'b.npy',allow_pickle=False))
        saved=json.loads((p/'CPU_PILOT100.json').read_text())
        np.testing.assert_allclose(logits,np.asarray([r['logits'] for r in saved['cases']]),atol=1e-6)
        np.testing.assert_array_equal(spikes,np.load(p/'cpu_spikes.npy',allow_pickle=False))


if __name__=='__main__':unittest.main()
