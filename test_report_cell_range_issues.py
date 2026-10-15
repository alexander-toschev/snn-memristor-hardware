"""Saved-data report arithmetic only; no native imports or I/O."""
import unittest
from report_cell_range_issues import mvm_transfer,PROTOCOL_FLAGS


def target(wl,q,scale=50.,offset=0.,source='hardware'):
    rows=[]
    for repeat in range(5):
        for position,v in enumerate((0.,.025,.05,.075,.1,0.)):
            output=offset+scale*q*v
            rows.append(dict(position=position,repeat=repeat,nominal_v=v,raw_adc=output*16383/5,output_voltage_v=output))
    return dict(session_id='fixture',source=source,wl=wl,bl=8,target_q=q,cycle_id=1,
                pre={'q':{'mean':q}},mvm=rows,interval_and_mvm_pass=True)


class ReportTests(unittest.TestCase):
    def test_ratio_and_nonzero_affine_fit(self):
        result=mvm_transfer([target(0,.05)])['states'][0]
        self.assertAlmostEqual(result['observed_output_over_q_times_input'],50.)
        self.assertAlmostEqual(result['nonzero_requested_input_affine_fit']['slope_v_per_v'],2.5)
        self.assertAlmostEqual(result['nonzero_requested_input_affine_fit']['intercept_v'],0.)
        self.assertFalse(result['physical_input_voltage_measured'])

    def test_pair_is_descriptive_not_qualified(self):
        data=[target(w,q,offset=.01 if w==0 else 0.) for w in (0,3) for q in (.045,.05)]
        pair=mvm_transfer(data)['descriptive_non_simultaneous_pairs'][0]
        high=pair['levels'][-1]
        self.assertAlmostEqual(high['LL_adc'],.01*16383/5)
        self.assertAlmostEqual(high['descriptive_half_swing_adc'],.025*16383/5)
        self.assertAlmostEqual(pair['half_swing_high_minus_twice_low_adc'],0.)
        self.assertFalse(pair['pair_qualified']);self.assertFalse(pair['simultaneously_programmed_or_measured'])

    def test_simulator_not_mixed_and_incomplete_pair_not_invented(self):
        result=mvm_transfer([target(0,.045),target(3,.05,source='simulator')])
        self.assertEqual(len(result['states']),1)
        self.assertEqual(result['descriptive_non_simultaneous_pairs'],[])

    def test_wrong_mvm_level_rejected(self):
        t=target(0,.05);t['mvm'][1]['nominal_v']=.5
        with self.assertRaises(ValueError):mvm_transfer([t])

    def test_protocol_reproduction_flags_differ(self):
        self.assertEqual(PROTOCOL_FLAGS['three-cell-native-R-interval-v1'],'--test-cell-ranges')
        self.assertEqual(PROTOCOL_FLAGS['three-cell-native-R-interval-v2-refined'],'--test-cell-ranges-refined')


if __name__=='__main__':unittest.main()
