"""Tiny synthetic interrupted-prefix tests; never train/run a neural model."""
import copy
import unittest
from unittest import mock
from experiments.mnist_stdp_finish_20261019 import experiment as e, qualification as q
from tools import audit_mnist_stdp_finish as audit


def rows():
    result=[dict(kind='first_image_clipping_and_learning')]
    result += [dict(kind='stdp_image',image_id=i,steps=100,native_norm_none=True) for i in range(2)]
    result += [dict(kind='stdp_training_complete'),dict(kind='new_weight_features',split='train',start=0,end=2),
               dict(kind='feature_split_complete',split='train',rows=2,new_weights=True),dict(kind='scaler_complete',train_rows=2,test_access=False)]
    result += [dict(kind='head_epoch',epoch=i,updates=i,train_examples=2,test_access=False) for i in (1,2)]
    return result


class RecoveryTests(unittest.TestCase):
    def test_exact_completed_prefix(self):
        parsed=q.parse_prefix(rows(),size=2,epochs=2)
        self.assertEqual(len(parsed['images']),2);self.assertEqual(len(parsed['heads']),2)

    def test_missing_duplicate_or_later_test_rejected(self):
        fixtures=[]
        x=rows();del x[1];fixtures.append(x)
        x=rows();x.insert(1,copy.deepcopy(x[1]));fixtures.append(x)
        fixtures.append(rows()+[dict(kind='test_complete')])
        x=rows();x[4]['split']='test';fixtures.append(x)
        for x in fixtures:
            with self.assertRaises(ValueError):q.parse_prefix(x,size=2,epochs=2)

    def test_prefix_new_head_determinism(self):
        original=dict(epoch=1,updates=59,train_examples=30000,order_sha256='saved',test_access=False,train_loss=1.)
        observed=dict(original,train_loss=1.+1e-7);e.validate_prefix_epoch(observed,original)
        for key,value in [('order_sha256','changed'),('updates',60),('test_access',True),('train_loss',1.1)]:
            changed=dict(original);changed[key]=value
            with self.assertRaises(ValueError):e.validate_prefix_epoch(changed,original)

    def test_closed_gate_before_helpers_or_model(self):
        with mock.patch.object(e,'APPROVED_SPEC_SHA256',frozenset()), mock.patch.object(e,'helpers') as helper:
            with self.assertRaisesRegex(ValueError,'review and registration'):e.main(['--run'])
            helper.assert_not_called()

    def test_recovery_auditor_full_tail_and_double_eval_rejected(self):
        values=[dict(kind='inherited_prefix_qualified')]
        values += [dict(kind='head_epoch',epoch=i,updates=59*i,train_examples=30000,test_access=False) for i in (1,2)]
        values += [dict(kind='head_frozen_before_test'),dict(kind='restored_new_weight_network')]
        values += [dict(kind='new_weight_features',split='test',start=0,end=128),
                   dict(kind='new_weight_features',split='test',start=128,end=130),
                   dict(kind='feature_split_complete',split='test',rows=130,new_weights=True),dict(kind='test_complete')]
        self.assertEqual(len(audit.schedule(values,epochs=2,size=130)['features']),2)
        with self.assertRaises(ValueError):audit.schedule(values+[dict(kind='test_complete')],epochs=2,size=130)
        invalid=copy.deepcopy(values);invalid[3],invalid[5]=invalid[5],invalid[3]
        with self.assertRaises(ValueError):audit.schedule(invalid,epochs=2,size=130)

    def test_no_fake_optimizer_resume_or_training_in_recovery_events(self):
        with self.assertRaises(ValueError):audit.schedule([dict(kind='stdp_image')])

    def test_source_union_preserves_original_summary_and_rejects_conflict(self):
        original={'summary.json':'original','checkpoint.pt':'weights'}
        checked={'checkpoint.pt':'weights'};helper={'training.py':'code'};own={'finish.py':'newcode'}
        combined=e.merge_sources(helper,checked,original,own)
        self.assertEqual(combined,e.merge_sources(original,helper,own))
        self.assertIn('summary.json',combined)
        with self.assertRaises(ValueError):e.merge_sources(original,{'summary.json':'different'})


if __name__=='__main__':unittest.main()
