import unittest
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT,ROOT/'.deps-classifier',ROOT/'.deps'):
    sys.path.insert(0,str(p))
import numpy as np
import torch
from experiments.compact_cifar_20261020 import data,model,experiment


class CompactTests(unittest.TestCase):
    def test_integer_pool_and_threshold(self):
        images=np.zeros((3,3,32,32),dtype=np.uint8);images[1]=255
        images[2,0,:8,:8]=25;images[2,0,:4,:8]=26
        sums,times,spikes=data.encode(images)
        self.assertTrue((times[0]==-1).all());self.assertTrue((times[1]==0).all())
        self.assertEqual(sums[2,0],1632);self.assertEqual(times[2,0],9)
        self.assertEqual(int(spikes[2].sum()),1)
        images[2,0,0,0]-=1
        self.assertEqual(data.encode(images)[1][2,0],-1)

    def test_surrogate_and_detached_reset(self):
        a=torch.tensor([-1.,0.,1.],requires_grad=True)
        model.Spike.apply(a).sum().backward()
        torch.testing.assert_close(a.grad,torch.tensor([1/36,1.,1/36]))
        n=model.Network()
        with torch.no_grad():n.W.zero_();n.A.zero_();n.b.zero_();n.W[0,0]=1;n.A[0,0]=1
        x=torch.zeros(1,12,48);x[0,0,0]=1
        logits,z,_,v=n(x,True)
        self.assertEqual(z.sum().item(),1);self.assertEqual(v[0,0,0].item(),0)
        logits[0,0].backward()
        self.assertAlmostEqual(n.W.grad[0,0].item(),1/12,places=7)
        with torch.no_grad():n.W[0,0]=-1
        self.assertEqual(n(x,True)[3][0,0,0].item(),-1)

    def test_init_reproducible_and_complete_48_sum(self):
        a=experiment.initialize();b=experiment.initialize()
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(a.parameters(),b.parameters())))
        x=torch.zeros(1,12,48);x[0,0]=1
        torch.testing.assert_close(a(x,True)[2][0,0],a.W.sum(1))

    def test_symbolic_reader_rejects_code_global(self):
        with self.assertRaises(ValueError):data.parse_official(b'\x80\x02cos\nsystem\n.')

    def test_actual_dataset_and_sealed_splits(self):
        d=data.prepare();data.load_dataset('meta')
        self.assertEqual(d['images'].shape,(5000,3,32,32))
        self.assertEqual(d['test_images'].shape,(100,3,32,32))
        self.assertFalse(set(d['official_test_ids'])&data.EXCLUDED)
        self.assertEqual(np.bincount(d['test_labels']).tolist(),[10]*10)
        self.assertEqual(len(set(d['train_positions'])&set(d['val_positions'])),0)


if __name__=='__main__':unittest.main()
