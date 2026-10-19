"""Offline contract and synthetic tensor tests; no SNN construction or run."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from experiments.mnist_stdp_retraining_20261019 import experiment as e


class STDPTests(unittest.TestCase):
    def test_both_model_launches_remain_gated(self):
        with patch.object(e,'APPROVED_SMOKE_SPEC_SHA256',frozenset()):
            with self.assertRaisesRegex(ValueError,'review required'):e.main(['--smoke-128'])
        with self.assertRaisesRegex(ValueError,'review required'):e.main(['--run'])

    def test_parent_positive_rates_and_fixed_budget(self):
        import json
        self.assertEqual(e.recovery.sha(e.SUMMARY),e.SUMMARY_SHA)
        cfg=json.loads(e.SUMMARY.read_text(encoding='utf-8'))['cfg']
        self.assertEqual((cfg['nu_minus'],cfg['nu_plus']),(.001,.0001))
        self.assertEqual(e.SPEC['full_training_samples']*e.SPEC['timesteps'],3000000)
        self.assertEqual(e.SPEC['training_batch'],1)
        self.assertEqual(e.recovery.sha(e.recovery.__file__),e.RECOVERY_SHA)

    def test_clipping_separated_from_later_plasticity_and_postrun_transform(self):
        import torch
        initial=torch.full((64,1,5,5),3.136)
        trace=torch.ones((100,64,1,5,5));trace[1:]-=.1
        result=e.assert_clipping_trace(initial,trace,trace[-1].clone())
        self.assertEqual(result['initial_out_of_bounds'],1600)
        self.assertEqual(result['post_first_step_changed_coefficients'],1600)
        with self.assertRaisesRegex(ValueError,'normalization'):e.assert_clipping_trace(initial,trace,trace[-1]+.01)
        trace[0,0,0,0,0]=.9
        with self.assertRaisesRegex(ValueError,'pure clipping'):e.assert_clipping_trace(initial,trace,trace[-1])

    def test_learning_flags_rates_bounds_not_silently_fixed(self):
        import torch
        Rule=type('WeightDependentPostPre',(),{})
        rule=Rule();rule.nu=torch.tensor([.001,.0001]);rule.weight_decay=0
        conn=SimpleNamespace(update_rule=rule,norm=None,Dales_rule=None,wmin=torch.tensor(0.),wmax=torch.tensor(1.),b=torch.zeros(64))
        net=SimpleNamespace(connections={('Input','Conv1'):conn},learning=True,training=True,
                            layers={'Input':SimpleNamespace(learning=True,training=True)})
        e.check_learning(net,conn)
        rule.nu=torch.tensor([-.001,.0001])
        with self.assertRaisesRegex(ValueError,'rates/sign'):e.check_learning(net,conn)
        rule.nu=torch.tensor([.001,.0001]);net.learning=False
        with self.assertRaisesRegex(ValueError,'learning'):e.check_learning(net,conn)


if __name__=='__main__':unittest.main()
