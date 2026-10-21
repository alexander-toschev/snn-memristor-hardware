"""Meaningful offline tests, no native MemriBoard session."""
from __future__ import annotations
import gzip
import json
from pathlib import Path
import tempfile
import time
import unittest
import numpy as np
import torch
from experiments.mnist_archive64_20261020 import inputs
from experiments.mnist_online_stdp64_20261021.logic import DigitalStep
from experiments.mnist_block_stdp64_20261021.core import Collector, EvidenceWriter, WallBudget, digital_block
from tests.test_mnist_online_stdp64 import FakeCV, FakeWriter, FakeConnector


class BlockTests(unittest.TestCase):
    def test_shadow_carries_between_images_but_forward_uses_fixed_measured_lut(self):
        torch.set_num_threads(2)
        pixels=inputs.images('train',2)
        seen={0}
        for _,masks in inputs.batches(pixels,2):seen.update(map(int,np.unique(masks)))
        keys=np.asarray(sorted(seen),dtype=np.uint32)
        lut=np.full((len(keys),4),.22,dtype=np.float32);lut[0]=0
        measured=np.full((4,25),.5,dtype=np.float64)
        observed,summary=digital_block(pixels,keys,lut,measured,lambda:None)
        step=DigitalStep();target=torch.full((4,25),.5)
        totals=[]
        for image in range(2):
            step.reset();spikes=inputs.encode(pixels[image:image+1]);masks=inputs.patch_masks(spikes);count=0
            for tick in range(100):
                current=torch.zeros((1,4,24,24)) if tick==0 else torch.from_numpy(lut[np.searchsorted(keys,masks[tick-1,0])].transpose(2,0,1).copy()).unsqueeze(0)
                post,target=step.step(current,spikes[tick],target);count+=int(post.sum())
            totals.append(count)
        self.assertTrue(np.array_equal(observed,target.numpy()))
        self.assertEqual(summary['spikes_per_image'],totals)
        self.assertGreater(summary['correlated_update_ticks'],0)
        self.assertGreater(sum(totals),0)
        self.assertTrue(np.array_equal(measured,np.full((4,25),.5)))
        self.assertTrue(np.array_equal(lut[1:],np.full((len(keys)-1,4),.22,dtype=np.float32)))

    def test_boundary_correction_is_not_called_plasticity(self):
        pixels=np.zeros((1,28,28),dtype=np.uint8)
        target,summary=digital_block(pixels,np.array([0],dtype=np.uint32),np.zeros((1,4),dtype=np.float32),
                                     np.full((4,25),1.2),lambda:None)
        self.assertEqual(summary['plasticity_nonzero'],0)
        self.assertEqual(summary['correction_nonzero'],100)
        self.assertTrue(np.array_equal(target,np.ones((4,25),dtype=np.float32)))

    def test_cache_endcheck_write_invalidation_and_deadline(self):
        conn=FakeConnector();runner=Collector(conn,FakeCV(),FakeWriter(),1000,4,1e-4,lambda _:None,time.monotonic()+60)
        runner.measure([0,1],0,'test');cache=runner.active_cache
        runner.verify_end(0)
        before=len(conn.commands)
        runner.pulse(0,0,1,1.7,.00001)
        self.assertFalse(cache.sealed);self.assertIsNone(runner.active_cache)
        self.assertGreater(len(conn.commands),before)
        runner.deadline=time.monotonic()-1
        before=len(conn.commands)
        with self.assertRaises(WallBudget):runner.read_r(0,0)
        self.assertEqual(len(conn.commands),before)

    def test_changed_native_output_fails_endcheck(self):
        class Changed(FakeCV):
            value=.003
            def convert_adc_to_volt(self,*args):return self.value
        cv=Changed();runner=Collector(FakeConnector(),cv,FakeWriter(),1000,4,1e-4,lambda _:None,time.monotonic()+60)
        runner.measure([0],0,'test');cv.value=.004
        with self.assertRaisesRegex(ValueError,'changed at block end'):runner.verify_end(0)

    def test_sampled_raw_declares_omission_and_preserves_control_section(self):
        root=Path('.tmp');root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as folder:
            writer=EvidenceWriter(Path(folder)/'raw')
            writer.write(dict(event='request',task_id=1))
            writer.retain_raw=True;writer.write(dict(event='request',task_id=2))
            writer.write(dict(event='block_complete',images=32));writer.finish()
            rows=[]
            for path in (Path(folder)/'raw').glob('*.gz'):
                rows.extend(json.loads(line) for line in gzip.open(path,'rt',encoding='utf-8'))
            self.assertEqual([r.get('task_id') for r in rows if r['event']=='request'],[2])
            self.assertEqual(writer.summary()['omitted_event_counts'],{'request':1})
            self.assertFalse(writer.summary()['full_native_raw_replay_available'])


if __name__=='__main__':unittest.main()
