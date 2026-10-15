import unittest

from report_mapped_sequential_diagnostics import explain


class DiagnosticTests(unittest.TestCase):
    def test_contrast_noise_and_lif_eligibility_remain_separate(self):
        def cell(wl,signal,sd):
            value=dict(signal_adc=signal,noise_adc=sd,quality=True,
                zero=dict(n=5,mean=0.,sd=0.),single=dict(n=5,mean=signal,sd=sd))
            return dict(wl=wl,bl=0,basis=value,selection=value,final=value,
                singleton_supported=True,g_consistent=True,g_fraction_delta=0.,g_fraction_limit=.05)
        coefficient=dict(pair_id=0,logical_index=0,positive_cell=[0,0],negative_cell=[1,0],
            basis=dict(signal_adc=50.,conservative_sd_adc=0.,quality=True),
            selection=dict(signal_adc=50.,conservative_sd_adc=0.,quality=True),
            final_signed_adc=50.,final_sd_bound_adc=5.,final_supported=False)
        trace=dict(pair_id=0,pattern='dense_all',polarity=1,status='inconclusive',numerical_lif_status='pass',
            reference_excited_and_silent=True,all_coefficients_supported=False,mvm_steps_pass=True,
            actual=[[] for _ in range(5)],max_membrane_errors=[0.]*5,exact_spikes_by_repeat=[True]*5)
        source=dict(cells=[cell(0,100.,5.),cell(1,50.,0.)],coefficients=[coefficient],steps=[],lif=[trace],
            singleton_supported=2,mapped_contrasts_supported=0,step_statuses={},lif_statuses={'inconclusive':1})
        result=explain(source)
        self.assertEqual(len(result['unsupported_coefficients']),1)
        self.assertIn('final: signed SDbound=5 > .05×|D|=2.5 ADC',result['unsupported_coefficients'][0]['reasons'])
        self.assertEqual(result['lif'][0]['numerical_status'],'pass');self.assertEqual(result['lif'][0]['saved_status'],'inconclusive')
        self.assertEqual(result['future_candidate_subsets'][0]['positions'],0)
        self.assertFalse(result['thresholds_changed']);self.assertFalse(result['new_remap_qualified'])


if __name__=='__main__':unittest.main()
