"""Offline/fake-only qualification; no native Connector or hardware is opened."""
from __future__ import annotations
import unittest
import numpy as np
import torch
from bindsnet.network import Network
from bindsnet.network.nodes import Input,LIFNodes
from bindsnet.network.topology import Conv2dConnection
from bindsnet.learning import WeightDependentPostPre
from experiment_common.mnist_stdp import DigitalStep,initialization
from experiment_common.stdp_collector import Collector


class FakeConnector:
    def __init__(self,resistance=100000.):self.commands=[];self.resistance=resistance
    def impact(self,task):
        self.commands.append(dict(task))
        return [self.resistance,task['id'],100]

class FakeCV:
    def convert_volt_to_dac(self,*args):return 41
    def convert_dac_to_volt(self,*args):return .05
    def convert_adc_to_volt(self,*args):return .003

class FakeWriter:
    def __init__(self):self.rows=[]
    def write(self,row):self.rows.append(row)


class OnlineTests(unittest.TestCase):
    def test_fresh_initialization_is_repeatable_and_records_first_clip(self):
        a,b=initialization();c,d=initialization()
        self.assertTrue(torch.equal(a,c));self.assertTrue(torch.equal(b,d))
        self.assertEqual(a.shape,(64,1,5,5));self.assertTrue(torch.allclose(a.sum((1,2,3)),torch.full((64,),78.4)))
        self.assertTrue(bool((a>1).any()));self.assertTrue(torch.equal(b,a.clamp(0,1)))

    def test_digital_tick_exact_native_bindsnet_network_order(self):
        # Golden uses an ideal conv ONLY inside this test, to supply known
        # currents. Production DigitalStep never calls connection.compute.
        torch.set_num_threads(2);torch.manual_seed(19)
        net=Network(dt=1.)
        source=Input(shape=(1,28,28),traces=True,tc_trace=20.)
        target=LIFNodes(shape=(4,24,24),traces=True,tc_trace=20.,thresh=.38,rest=0.,reset=0.,refrac=2,tc_decay=150.)
        net.add_layer(source,'Input');net.add_layer(target,'Conv')
        connection=Conv2dConnection(source,target,kernel_size=5,stride=1,padding=0,w=torch.rand((4,1,5,5))*.4,wmin=0.,wmax=1.)
        connection.update_rule=WeightDependentPostPre(connection=connection,nu=(.001,.0001),weight_decay=1.)
        net.add_connection(connection,'Input','Conv');net.train(True)
        digital=DigitalStep();posts=0
        for tick in range(12):
            new=torch.zeros((1,1,28,28));new[:,:,tick%3::3,tick%4::4]=1
            with torch.no_grad():
                before=connection.w.clone()
                current=torch.nn.functional.conv2d(source.s.float(),before)
                observed,wanted=digital.step(current,new,before.reshape(4,25))
                net.run(inputs={'Input':new.unsqueeze(0)},time=1.)
            posts+=int(observed.sum())
            self.assertTrue(torch.equal(observed,target.s))
            self.assertTrue(torch.equal(wanted,connection.w.reshape(4,25)))
            self.assertTrue(torch.equal(digital.source.x,source.x))
            self.assertTrue(torch.equal(digital.target.x,target.x))
            self.assertTrue(torch.equal(digital.target.v,target.v))
            self.assertTrue(torch.equal(digital.target.refrac_count,target.refrac_count))
        self.assertGreater(posts,0)

    def test_native_pulse_direction_measured_feedback_and_cache_invalidation(self):
        for sign in (0,1):
            connector=FakeConnector();writer=FakeWriter()
            runner=Collector(connector,FakeCV(),writer,200,2,1e-4,lambda _:None)
            runner.measure([0],0,'test');cache=runner.active_cache
            self.assertTrue(cache.sealed)
            g=np.full((4,25,2),1e-5,dtype=np.float64);target=np.zeros((4,25),dtype=np.float32)
            if sign:target[0,0]=.1
            else:g[0,0,0]=2e-5
            runner.update(g,target,0,0,1)
            writes=[r for r in connector.commands if r['mode_flag']==7 and r['vol']!=0]
            self.assertEqual(len(writes),1);self.assertEqual(writes[0]['sign'],sign)
            self.assertEqual((writes[0]['wl'],writes[0]['bl']),(0,0))
            self.assertFalse(cache.sealed);self.assertIsNone(runner.active_cache)
            self.assertEqual(runner.pulses,1);self.assertEqual(runner.learning_stats['deadband'],99)
            self.assertEqual(len([r for r in writer.rows if r['event']=='learning_readback']),1)

    def test_deadband_and_guard_are_distinct_no_ideal_weight_commit(self):
        runner=Collector(FakeConnector(500.),FakeCV(),FakeWriter(),20,2,1e-4,lambda _:None)
        g=np.full((4,25,2),1e-5,dtype=np.float64);target=np.zeros((4,25),dtype=np.float32)
        target[0,0]=.0019
        runner.update(g,target,0,0,1);self.assertEqual(runner.calls,0)
        target[0,0]=.1
        runner.update(g,target,0,0,2)
        self.assertEqual(runner.pulses,0);self.assertEqual(runner.decisions,1)
        self.assertEqual(runner.calls,4);self.assertEqual(runner.learning_stats['guard_skipped'],1)
        self.assertTrue(np.array_equal(g,np.full((4,25,2),1e-5)))


if __name__=='__main__':unittest.main()
