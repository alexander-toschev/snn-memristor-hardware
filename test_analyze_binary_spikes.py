import math
import unittest
from analyze_binary_spikes import classify, coverage, pairings

RULES=dict(noise_floor_adc=1,minimum_snr=5,maximum_cv_high=.05,drift_fraction=.05,drift_se_multiplier=3)


class BinaryAnalysisTests(unittest.TestCase):
    def screen(self,level,zero=None,pre=None,post=None):
        return classify(level,zero if zero is not None else [0]*5,pre or [1]*3,post or [1]*3,RULES)

    def test_zero_and_negative_signal_are_not_low_noise_candidates(self):
        for row in (self.screen([0]*5),self.screen([1]*5,[2]*5)):
            self.assertEqual(row['status'],'reject');self.assertIsNone(row['cv'])
            self.assertIn('nonpositive_signal',row['reasons'])

    def test_noise_floor_boundary_inclusive_and_zero_noise(self):
        self.assertEqual(self.screen([5]*5)['status'],'provisional_candidate')
        self.assertEqual(self.screen([4]*5)['status'],'reject')

    def test_combined_sd_sample_not_standard_error_and_cv_gate(self):
        row=self.screen([96,98,100,102,104],[0,1,2,3,4])
        self.assertAlmostEqual(row['combined_sd_adc'],math.sqrt(12.5))
        self.assertAlmostEqual(row['signal_adc'],98)
        self.assertAlmostEqual(row['cv'],math.sqrt(12.5)/98)
        row=self.screen([80,90,100,110,120])
        self.assertGreater(row['snr'],5);self.assertIn('cv_above_0.05',row['reasons'])

    def test_drift_mean_uncertainty_and_inconsistent_state(self):
        row=self.screen([100]*5,pre=[.9,1,1.1],post=[1,1.1,1.2])
        self.assertAlmostEqual(row['g_fractional_tolerance'],3*math.sqrt(.01/3+.01/3))
        self.assertEqual(row['status'],'provisional_candidate')
        row=self.screen([100]*5,post=[1.06]*3)
        self.assertIn('pre_post_g_inconsistent',row['reasons'])

    def test_missing_and_invalid_readback_never_candidate(self):
        self.assertEqual(self.screen([100]*4)['status'],'unmeasured')
        self.assertEqual(self.screen([100]*5,pre=[0]*3)['status'],'unmeasured')
        self.assertEqual(self.screen([float('nan')]*5)['status'],'unmeasured')

    def test_all_105_partitions_unique_complete_and_pair_counts(self):
        partitions=list(pairings());self.assertEqual(len(partitions),105)
        self.assertEqual(len(set(partitions)),105)
        for pairs in partitions:self.assertEqual(sorted(x for p in pairs for x in p),list(range(8)))
        self.assertEqual(sum((0,1) in ps for ps in partitions),15)

    def test_capacity_uses_same_amplitude_coordinates_and_all_eight(self):
        c=coverage([{1,2} if w<4 else {2,3} for w in range(8)])
        self.assertEqual(c['shared_bl_all_eight_wl'],[2])
        self.assertEqual(c['maximum_capacity_sum'],8)
        self.assertEqual(c['maximum_minimum_pair_capacity'],2)
        self.assertTrue(all(p['shared_bl_all_four_pairs']==[2] for p in c['partitions']))


if __name__=='__main__':unittest.main()
