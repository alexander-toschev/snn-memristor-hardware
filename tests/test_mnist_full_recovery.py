"""Tiny synthetic math and coverage only; no MNIST network or full training."""
import inspect
import unittest
from experiments.mnist_full_recovery_20261019 import experiment as e


class RecoveryTests(unittest.TestCase):
    def test_exact_split_and_no_padded_tails(self):
        for size,batch,count,tail in [(30000,128,235,48),(10000,128,79,16),(30000,512,59,304),(10000,512,20,272)]:
            parts=e.batches(size,batch)
            self.assertEqual(len(parts),count);self.assertEqual(parts[-1][1]-parts[-1][0],tail)
            self.assertEqual([i for a,b in parts for i in range(a,b)],list(range(size)))

    def test_streaming_sample_sd_matches_reference_with_constant_columns(self):
        import numpy as np
        import torch
        x=np.array([[0,0,1,100],[0,1,2,99],[0,3,4,98],[0,7,8,97],[0,15,16,96]],dtype=np.float32)
        mean,sd=e.fit_scaler(x,batch_size=2)
        ref=torch.log1p(torch.from_numpy(x));torch.testing.assert_close(mean,ref.mean(0),rtol=1e-6,atol=1e-7)
        torch.testing.assert_close(sd,ref.std(0,correction=1).clamp_min(1e-6),rtol=1e-6,atol=1e-7)
        self.assertEqual(float(sd[0]),float(torch.tensor(1e-6)))
        self.assertEqual(list(inspect.signature(e.fit_scaler).parameters),['counts','batch_size'])

    def test_bad_counts_fail_closed(self):
        import numpy as np
        for value in [float('nan'),-1,101,.5]:
            with self.assertRaisesRegex(ValueError,'spike counts'):e.fit_scaler(np.array([[value],[0]],dtype=np.float32))

    def test_fixed_training_contract_and_closed_gate(self):
        self.assertEqual(e.SPEC['epochs']*len(e.batches(30000,512)),1475)
        self.assertEqual(e.SPEC['test_evaluations'],1);self.assertEqual(e.SPEC['epoch_selection'],'fixed final25')
        with self.assertRaisesRegex(ValueError,'review required'):e.main(['--run'])
        self.assertEqual(e.sha(e.PILOT),e.PILOT_SHA)


if __name__=='__main__':unittest.main()
