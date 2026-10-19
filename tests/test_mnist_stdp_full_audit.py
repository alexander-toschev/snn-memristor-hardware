"""Small synthetic evidence mutations only; no neural execution or training."""
import copy
import unittest
import numpy as np
from tools import audit_mnist_stdp_full as a


def fixture():
    rows = [dict(kind='first_image_clipping_and_learning')]
    rows += [dict(kind='stdp_image', image_id=i, steps=100, native_norm_none=True) for i in range(2)]
    rows += [dict(kind='stdp_training_complete')]
    for start, end in [(0, 128), (128, 130)]:
        rows.append(dict(kind='new_weight_features', split='train', start=start, end=end))
    rows += [dict(kind='feature_split_complete', split='train', rows=130, new_weights=True),
             dict(kind='scaler_complete', train_rows=130, test_access=False)]
    rows += [dict(kind='head_epoch', epoch=i, updates=i, train_examples=130, test_access=False) for i in [1, 2]]
    rows += [dict(kind='head_frozen_before_test'), dict(kind='new_weight_features', split='test', start=0, end=3),
             dict(kind='feature_split_complete', split='test', rows=3, new_weights=True), dict(kind='test_complete')]
    return rows


class FullAuditTests(unittest.TestCase):
    def check(self, rows): return a.schedule(rows, training=2, train=130, test=3, epochs=2)

    def test_tail_coverage_and_actual_counters(self):
        result = self.check(fixture())
        self.assertEqual(result['actual']['feature_rows'], {'train': 130, 'test': 3})
        self.assertEqual(result['actual']['head_updates_at_last_completed_epoch'], 2)
        self.assertEqual(result['actual']['training_images'], 2)

    def test_missing_duplicate_out_of_order_training(self):
        for change in ('missing', 'duplicate', 'order'):
            rows = fixture()
            if change == 'missing': del rows[1]
            elif change == 'duplicate': rows.insert(1, copy.deepcopy(rows[1]))
            else: rows[1]['image_id'] = 1
            with self.subTest(change=change), self.assertRaises(ValueError): self.check(rows)

    def test_feature_gap_overlap_tail_and_test_leakage(self):
        for edit in [('start', 127), ('end', 129), ('split', 'test')]:
            rows = fixture(); rows[5][edit[0]] = edit[1]
            with self.subTest(edit=edit), self.assertRaises(ValueError): self.check(rows)
        rows = fixture(); rows[7]['test_access'] = True
        with self.assertRaises(ValueError): self.check(rows)

    def test_test_before_freeze_and_second_evaluation_rejected(self):
        rows = fixture(); rows[10], rows[11] = rows[11], rows[10]
        with self.assertRaises(ValueError): self.check(rows)
        rows = fixture() + [dict(kind='test_complete')]
        with self.assertRaises(ValueError): self.check(rows)

    def test_accuracy_recomputed_and_mutation_rejected(self):
        scores = np.zeros((3, 10), np.float32); scores[0, 1] = 1; scores[1, 2] = 1
        predicted = np.array([1, 2, 0], np.int64); labels = np.array([1, 3, 0], np.int64)
        confusion = np.zeros((10, 10), np.int64); np.add.at(confusion, (labels, predicted), 1)
        self.assertEqual(a.predictions(scores, predicted, labels, confusion)['correct'], 2)
        wrong = predicted.copy(); wrong[2] = 1
        with self.assertRaises(ValueError): a.predictions(scores, wrong, labels, confusion)
        confusion[1, 1] = 0
        with self.assertRaises(ValueError): a.predictions(scores, predicted, labels, confusion)

    def test_paths_cannot_escape(self):
        with self.assertRaises(ValueError): a.inside(a.ROOT, '../elsewhere')

    def test_linear_replay_detects_wrong_head_or_logits(self):
        counts = np.array([[0., 1.], [2., 0.]], np.float32)
        mean = np.zeros(2, np.float32); sd = np.ones(2, np.float32)
        weight = np.eye(2, dtype=np.float32); bias = np.zeros(2, np.float32)
        saved = np.log1p(counts).astype(np.float32)
        self.assertTrue(a.linear_replay(counts, mean, sd, weight, bias, saved, batch=1)['passed'])
        wrong = saved.copy(); wrong[0, 0] = 10
        result = a.linear_replay(counts, mean, sd, weight, bias, wrong)
        self.assertFalse(result['passed']); self.assertEqual(result['argmax_disagreements'], 1)
        self.assertGreater(result['logit_tolerance_violations'], 0)

    def test_linear_near_tie_disagreement_never_silent_pass(self):
        counts = np.zeros((1, 2), np.float32)
        result = a.linear_replay(counts, np.zeros(2, np.float32), np.ones(2, np.float32),
                                 np.zeros((2, 2), np.float32), np.zeros(2, np.float32), np.array([[0., 1e-5]], np.float32))
        self.assertFalse(result['passed']); self.assertEqual(result['logit_tolerance_violations'], 0)
        self.assertEqual(result['argmax_disagreements'], 1)


if __name__ == '__main__': unittest.main()
