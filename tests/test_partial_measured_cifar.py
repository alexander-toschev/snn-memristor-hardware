"""Offline input-integrity and unchanged-filter checks; no board imports."""
import copy
import unittest
from tools import evaluate_partial_measured_cifar as e


class InputIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.trained = [[(f * 75 + t) / 1024 for t in range(75)] for f in range(64)]
        self.rows = [dict(filter=str(f), tap=str(t), weight=str(self.trained[f][t]), effective=str(-self.trained[f][t]))
                     for f, t in sorted(e.EXPECTED)]

    def test_exact_coverage_and_observed_zero_preserved(self):
        values = e.validate_rows(self.rows, self.trained)
        self.assertEqual(len(values), 300)
        self.assertEqual(values[0, 0], 0.)
        self.assertEqual(values[3, 74], -self.trained[3][74])

    def test_missing_duplicate_extra_and_nonfinite_rejected(self):
        invalid = [self.rows[:-1], self.rows + [self.rows[0]],
                   self.rows + [dict(filter="4", tap="0", weight="0", effective="0")]]
        for field, value in (("effective", "nan"), ("effective", "inf"), ("weight", "nan"), ("tap", "75"), ("filter", "-1"), ("weight", "123"), ("effective", "")):
            rows = copy.deepcopy(self.rows)
            rows[0][field] = value
            invalid.append(rows)
        for rows in invalid:
            with self.subTest(first=rows[0], count=len(rows)):
                with self.assertRaises(ValueError):
                    e.validate_rows(rows, self.trained)

    def test_only_authorized_weights_change_without_mutating_original(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch unavailable; pure input checks still run")
        original = torch.arange(4800, dtype=torch.float32).reshape(64, 3, 5, 5)
        before = original.clone()
        values = {key: -float(1 + key[0] * 75 + key[1]) for key in e.EXPECTED}
        modified = e.replace_weights(original, values)
        self.assertTrue(torch.equal(original, before))
        self.assertTrue(torch.equal(modified[4:], original[4:]))
        self.assertEqual(int((modified != original).sum()), 300)
        self.assertEqual(float(modified.reshape(64, 75)[3, 74]), -300.)
        with self.assertRaises(ValueError):
            e.replace_weights(original, {(0, 0): 0.})


if __name__ == "__main__":
    unittest.main()
