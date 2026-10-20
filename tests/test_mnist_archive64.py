"""Focused offline tests: no native Connector is instantiated/opened."""
from __future__ import annotations
import sys
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.mnist_archive64_20261020.cache import ExactCache,Response
from experiments.mnist_archive64_20261020 import inputs,lif,readout,native

class ArchiveTests(unittest.TestCase):
    def test_fixed_train10000_protocol_and_pilot(self):
        from experiments.mnist_archive64_20261020.experiment import SPEC,sample_counts
        self.assertEqual(sample_counts(False),(10000,10000))
        self.assertEqual(sample_counts(True),(128,100))
        self.assertEqual(SPEC['train_images'],10000)
        self.assertEqual(SPEC['test_images'],10000)
        self.assertFalse(SPEC['test_tuning'])
        self.assertFalse(SPEC['physical_hardware'])
        self.assertFalse(SPEC['new_stdp_training'])

    def test_cache_exact_mvm_different_guard_and_invalidation(self):
        cache=ExactCache(2,[0,31])
        for repeat in range(3):
            for wl in range(8):
                for mask in (0,31):cache.observe(2,wl,mask,Response(mask,.01*mask,(1000.+repeat,),repeat+1))
        cache.seal();self.assertEqual(cache.get(2,3,31).adc,31)
        with self.assertRaises(ValueError):cache.get(2,3,1)
        with self.assertRaises(ValueError):cache.get(3,3,31)
        cache.invalidate()
        with self.assertRaises(ValueError):cache.get(2,3,31)
        bad=ExactCache(0,[1]);bad.observe(0,0,1,Response(10,.001,(),1))
        with self.assertRaises(ValueError):bad.observe(0,0,1,Response(11,.001,(),2))
        with self.assertRaises(ValueError):bad.seal()

    def test_native_full25_vector_no_micro_sums(self):
        class FakeConnector:
            def __init__(self):self.commands=[]
            def impact(self,task):
                self.commands.append(task)
                return [100000.,task['id'],100 if task['mode_flag']==7 else 10]
        class CV:
            def convert_volt_to_dac(self,*args):return 41
            def convert_dac_to_volt(self,*args):return .05
            def convert_adc_to_volt(self,*args):return .003
        class Writer:
            def __init__(self):self.rows=[]
            def write(self,row):self.rows.append(row)
        connector=FakeConnector();writer=Writer()
        collector=native.Collector(connector,CV(),writer,26,lambda _:None)
        response=collector.query(3,(1<<25)-1,0,0)
        self.assertEqual(collector.calls,26);self.assertEqual(len(response.guards_r),25)
        mvm=[x for x in connector.commands if x['mode_flag']==10]
        self.assertEqual(len(mvm),1);self.assertEqual(mvm[0]['vol'],[.05]*25+[0.]*7)
        with self.assertRaises(ValueError):collector.query(3,0,0,0)

    def test_public_converter_mutation_does_not_corrupt_raw_si(self):
        class MutatingConnector:
            def __init__(self):self.received=None
            def impact(self,task):
                self.received=json.loads(json.dumps(task))
                for i,v in enumerate(task['vol']):task['vol'][i]=abs(int(round(v*4095/5,0)))
                return [100000.,task['id'],10]
        class SnapshotWriter:
            def __init__(self):self.rows=[]
            def write(self,row):self.rows.append(json.loads(json.dumps(row)))
        connector=MutatingConnector();writer=SnapshotWriter()
        collector=native.Collector(connector,None,writer,1,lambda _:None)
        volts=[.05]*25+[0.]*7
        collector.call(dict(mode_flag=10,wl=2,vol=volts))
        request=next(r for r in writer.rows if r['event']=='request')
        result=next(r for r in writer.rows if r['event']=='result')
        expected=dict(mode_flag=10,wl=2,vol=[.05]*25+[0.]*7,id=0)
        self.assertEqual(connector.received,expected)
        self.assertEqual(request['command_si'],expected)
        self.assertEqual(result['command_si'],expected)
        self.assertEqual(volts,[.05]*25+[0.]*7)

    def test_poisson_batch_independent_and_patch_order(self):
        pixels=np.arange(2*784,dtype=np.uint16).reshape(2,1,28,28).astype(np.uint8)
        batch=inputs.encode(pixels)
        self.assertTrue(torch.equal(batch[:,:1],inputs.encode(pixels[:1])))
        self.assertTrue(torch.equal(batch[:,1:],inputs.encode(pixels[1:])))
        synthetic=torch.zeros((100,1,1,28,28));synthetic[0,0,0,0,0]=1;synthetic[0,0,0,4,4]=1
        masks=inputs.patch_masks(synthetic);self.assertEqual(int(masks[0,0,0,0]),1+(1<<24))

    def test_lif_exact_archived_network_delay_refrac_and_input_immutable(self):
        sys.path.insert(0,str(ROOT/'experiments/mnist_baseline_20261019/reference_source'))
        import csnn_mnist_net as archive
        from bindsnet.network.monitors import Monitor
        cfg=archive.CSCfg(c1_out=4,time=100,device='cpu',w_norm_enable=True,ei_enable=True,
                         wta_enable=False,nu_minus=0.,nu_plus=0.)
        torch.manual_seed(9);net,_,layer,conn=archive.build_csnn(cfg)
        with torch.no_grad():conn.w.copy_(torch.rand_like(conn.w)*.4)
        net.train(False);conn.update_rule.nu=(torch.tensor(0.),torch.tensor(0.))
        monitor=Monitor(layer,state_vars=['s'],time=100);net.add_monitor(monitor,name='equivalence')
        spikes=torch.zeros(100,2,1,28,28);spikes[::3,:,0,::4,::4]=1
        currents=torch.zeros(100,2,4,24,24)
        with torch.no_grad():
            for t in range(1,100):currents[t]=torch.nn.functional.conv2d(spikes[t-1],conn.w)
            net.run(inputs={'Input':spikes},time=100)
        before=currents.clone();counts,v,refrac=lif.run_currents(currents,.38)
        self.assertTrue(torch.equal(counts,monitor.get('s').sum(0)))
        self.assertTrue(torch.equal(v,layer.v));self.assertTrue(torch.equal(refrac,layer.refrac_count))
        self.assertTrue(torch.equal(before,currents));self.assertGreater(int(counts.sum()),0)

    def test_grouped_lif_equals_all64(self):
        rng=np.random.default_rng(15)
        masks=rng.integers(0,3,size=(100,2,24,24),dtype=np.uint32)
        keys=np.asarray([0,1,2],dtype=np.uint32)
        lut=rng.uniform(0,.5,size=(3,64)).astype(np.float32);lut[0]=0
        all64=lif.counts_for_masks(masks,keys,lut,.38)
        grouped=np.concatenate([lif.counts_for_masks(masks,keys,lut[:,s:s+4],.38) for s in range(0,64,4)],axis=1)
        np.testing.assert_array_equal(all64,grouped)

    def test_readout_test_labels_only_after_saved_head_and_strict_silence(self):
        parent=ROOT/'.tmp/archive64-unit';parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as name:
            out=Path(name);train=np.asarray([[0,1,0,2],[1,0,1,0],[2,0,0,1],[0,2,1,0]],dtype=np.uint8)
            accesses=[]
            def fake_labels(split,count):
                accesses.append(split)
                if split=='test':self.assertTrue((out/'readout.pt').is_file())
                return np.arange(count,dtype=np.int64)%10
            with patch.object(readout,'labels',side_effect=fake_labels):
                model=readout.fit(train,out,lambda _:None)
                self.assertEqual(accesses,['train'])
                result=readout.evaluate(model,np.zeros((2,4),dtype=np.uint8),out)
                self.assertEqual(result['strict_accuracy'],0.);self.assertEqual(result['silent'],2)
                self.assertEqual(accesses,['train','test'])

if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
