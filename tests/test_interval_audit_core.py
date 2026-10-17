"""Manufactured numerical cases, no experiment runtime or hardware."""
import math
import unittest
from interval_audit_core import MAX_DECISIONS,MAX_IMPACTS,expected_band,inside,readback_inside,screen_adc


class IndependentIntervalMathTests(unittest.TestCase):
    def test_bound_identity(self):
        self.assertEqual(MAX_DECISIONS,4878)
        self.assertEqual(MAX_IMPACTS,6120)

    def test_inverted_resistance_boundaries_inclusive(self):
        b=expected_band(.04)
        self.assertAlmostEqual(b['r_min'],8168.316831683168)
        self.assertAlmostEqual(b['r_max'],8333.333333333334)
        self.assertTrue(inside(b['r_min'],b));self.assertTrue(inside(b['r_max'],b))
        self.assertFalse(inside(math.nextafter(b['r_min'],-math.inf),b))
        self.assertFalse(inside(math.nextafter(b['r_max'],math.inf),b))
        self.assertFalse(inside(float('nan'),b))

    def test_mean_in_range_does_not_hide_failed_sample(self):
        b=expected_band(.04);middle=8250
        self.assertTrue(readback_inside([middle]*5,b,5))
        self.assertFalse(readback_inside([middle]*4+[b['r_max']+.01],b,5))
        self.assertFalse(readback_inside([middle]*4,b,5))

    def test_zero_and_nonlinearity_are_not_success(self):
        self.assertFalse(screen_adc([0]*5,[0]*5,[0]*5)['passes'])
        self.assertFalse(screen_adc([0]*5,[80]*5,[200]*5)['passes'])
        self.assertTrue(screen_adc([0]*5,[100]*5,[200]*5)['passes'])

    def test_offset_adjusted_screen(self):
        self.assertTrue(screen_adc([10]*5,[110]*5,[210]*5)['passes'])
        self.assertFalse(screen_adc([10]*5,[12]*5,[14]*5)['passes'])


if __name__=='__main__':unittest.main()
