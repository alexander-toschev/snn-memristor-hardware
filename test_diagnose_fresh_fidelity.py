import copy
import unittest
import diagnose_fresh_fidelity as d


class Tests(unittest.TestCase):
    def stage(self):
        return dict(raw_target_nrmse=.2,uncertainty_nrms=.2,resolved_sign_error_slots=[],
            measured_sign_unresolved_significant_slots=[],physical_fidelity_pass=True,unresolved_quantization_slots=[0],
            one_adc_weight_resolution=.01,raw_measured_weight=[.1],raw_target_errors=[0.],observed_weight_sd=[.01],engineering_weight_uncertainty=[.02])
    def slots(self):return [dict(tap_index=8,positive_cell=[0,1],negative_cell=[1,2],trained_weight=.1)]
    def test_equal_threshold_pass_and_failure_reasons(self):
        stage=self.stage();self.assertEqual(d.fidelity(stage,self.slots())['reasons'],[])
        stage.update(raw_target_nrmse=.21,uncertainty_nrms=.22,resolved_sign_error_slots=[0],physical_fidelity_pass=False)
        out=d.fidelity(stage,self.slots());self.assertEqual(len(out['reasons']),3);self.assertEqual(out['wrong_sign_taps'],[8])
    def test_unresolved_is_not_wrong_sign(self):
        stage=self.stage();stage.update(measured_sign_unresolved_significant_slots=[0],physical_fidelity_pass=False)
        out=d.fidelity(stage,self.slots());self.assertEqual(out['wrong_sign_taps'],[])
        self.assertEqual(out['significant_sign_unresolved_taps'],[8])
    def test_false_recorded_gate_rejected(self):
        stage=self.stage();stage['physical_fidelity_pass']=False
        with self.assertRaises(ValueError):d.fidelity(stage,self.slots())
    def fixture(self):
        point=dict(conv_sum=0.,bntt_current=0.,lif_before_reset=-.2,lif_after_reset=-.2,lif_spike=0)
        golden={k:[[v] for _ in range(12)] for k,v in point.items() if k!='lif_spike'};golden['lif_spikes']=[[0] for _ in range(12)]
        fixture={'cases':[{'reference_t12x64':golden}]}
        rows=[dict(filter_id=0,case=0,original_test_index=7,repeat=r,actual=[dict(point) for _ in range(12)],
            effective_weight_reference=[dict(point) for _ in range(12)]) for r in range(5)]
        return rows,fixture
    def test_only_mismatches_and_repeat_groups_threshold_inclusive(self):
        rows,fixture=self.fixture();rows[0]['actual'][1].update(lif_before_reset=1.,lif_after_reset=0.,lif_spike=1)
        out=d.mismatch_details(rows,fixture)
        self.assertEqual(out['mismatch_counts'],dict(trained=1,effective=1));self.assertEqual(len(out['mismatch_rows']),1)
        self.assertEqual(out['mismatch_rows'][0]['actual']['threshold_margin'],0.)
        self.assertEqual(out['mismatch_rows'][0]['previous_v_after']['actual'],-.2)
        self.assertEqual(out['repeat_spike_variation_group_count'],1)
    def test_missing_duplicate_repeats_rejected(self):
        rows,fixture=self.fixture()
        with self.assertRaises(ValueError):d.mismatch_details(rows[:-1],fixture)
        with self.assertRaises(ValueError):d.mismatch_details(rows+[copy.deepcopy(rows[0])],fixture)

if __name__=='__main__':unittest.main()
