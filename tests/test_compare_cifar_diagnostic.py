"""Explicitly synthetic unit fixtures. No acquisition or physical measurements."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from compare_cifar_diagnostic import (AUDITOR_SHA,PROTOCOL,FIXTURE,FIXTURE_SHA,FIELDS,EXPECTED_KEYS,
    indexed_summary,comparison,repeat_variation,error_stats,verify_session,local)


def synthetic_summary():
    trajectories=[]
    for f in range(4):
        for c in range(10):
            for r in range(5):
                values=[dict(t=t,conv_sum=float(r),bntt_current=float(r),lif_before_reset=float(r),
                    lif_after_reset=float(r),lif_spike=r%2) for t in range(12)]
                trajectories.append(dict(filter_id=f,case=c,repeat=r,original_test_index=100+c,
                    status='measured',actual=values,effective_weight_reference=copy.deepcopy(values)))
    return dict(synthetic_unittest_not_measurement=True,status='completed',technical_completed=True,
        acquisition_complete=True,decoded_conv_bn_lif_complete=True,filter_tiles_completed=28,
        fitted_alpha_used_for_decoding=False,actual_branch_uses_software_dot_product=False,
        diagnostic_acquisition_not_accuracy_qualification=True,full_network_accuracy_claim=False,
        fixed_decoder_gain_v_per_weight=.03,physical_weight_quality_pass=False,
        total_impacts=123,total_pulses=12,address_coverage={},trajectories=trajectories)


class ComparisonArithmeticTests(unittest.TestCase):
    def test_known_signed_error_and_rmse(self):
        result=error_stats([2.,4.],[1.,2.])
        self.assertEqual(result['mean_error'],1.5)
        self.assertAlmostEqual(result['rmse'],2.5**.5)
        self.assertEqual(result['nrmse'],1.)

    def test_zero_reference_rms_has_null_normalized_error(self):
        result=error_stats([1.,-1.],[0.,0.])
        self.assertEqual(result['rmse'],1.);self.assertIsNone(result['nrmse'])

    def test_metrics_compare_values_not_saved_comparison_claims(self):
        actual={(f,0,0,0):dict.fromkeys(FIELDS,2.)|{'lif_spike':1} for f in range(4)}
        reference={(f,0,0,0):dict.fromkeys(FIELDS,1.)|{'lif_spike':0} for f in range(4)}
        result=comparison(actual,reference)
        self.assertEqual(result['overall']['numeric']['conv_sum']['rmse'],1.)
        self.assertEqual(result['overall']['spike_mismatches'],4)
        self.assertEqual(result['by_filter']['3']['spike_mismatches'],1)

    def test_repeat_sample_sd_not_sem(self):
        actual,_=indexed_summary(synthetic_summary(),list(range(100,110)))
        result=repeat_variation(actual)['overall']
        self.assertEqual(result['groups'],480)
        self.assertAlmostEqual(result['numeric']['conv_sum']['rms_sample_sd'],2.5**.5)
        self.assertEqual(result['numeric']['conv_sum']['maximum_repeat_range'],4.)
        self.assertEqual(result['spike_disagreement_groups'],480)

    def test_duplicate_trajectory_rejected(self):
        summary=synthetic_summary();summary['trajectories'][-1]=summary['trajectories'][0]
        with self.assertRaisesRegex(ValueError,'Duplicate'):indexed_summary(summary,list(range(100,110)))

    def test_missing_timestep_rejected_not_zero_filled(self):
        summary=synthetic_summary();summary['trajectories'][0]['actual'].pop()
        with self.assertRaisesRegex(ValueError,'Missing timestep'):indexed_summary(summary,list(range(100,110)))

    def test_different_test_image_rejected(self):
        summary=synthetic_summary();summary['trajectories'][0]['original_test_index']=999
        with self.assertRaisesRegex(ValueError,'Different CIFAR'):indexed_summary(summary,list(range(100,110)))

    def test_fitted_alpha_and_cpu_fallback_rejected(self):
        for key in ('fitted_alpha_used_for_decoding','actual_branch_uses_software_dot_product'):
            summary=synthetic_summary();summary[key]=True
            with self.assertRaises(ValueError):indexed_summary(summary,list(range(100,110)))

    def test_nonfinite_membrane_rejected(self):
        summary=synthetic_summary();summary['trajectories'][0]['actual'][0]['lif_before_reset']=float('nan')
        with self.assertRaisesRegex(ValueError,'Nonfinite'):indexed_summary(summary,list(range(100,110)))


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        base=Path(__file__).resolve().parents[1]/'.tmp';base.mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=base);self.addCleanup(self.temp.cleanup)
        self.repo=Path(self.temp.name).resolve();(self.repo/'synthetic_session').mkdir()
        self.summary=synthetic_summary();self.write('synthetic_session/summary.json',self.summary)
        self.manifest=dict(synthetic_unittest_not_measurement=True,session_id='synthetic-test-only',source='simulator',
            status='completed',acquisition_complete=True,protocol_id=PROTOCOL,hardware_access=False,
            historical_pilot_imported=False,all_tiles_new_measurements=True,trained_coefficients=300,tiles=28,
            physical_assignments=600,fixed_decoder_gain_v_per_weight=.03,source_sha256={FIXTURE:FIXTURE_SHA},
            result_sha256={'summary.json':self.digest('synthetic_session/summary.json')},invocation={'git':{'commit':'synthetic-test-only'}})
        self.write('synthetic_session/manifest.json',self.manifest)
        self.audit=dict(synthetic_unittest_not_measurement=True,audit_status='passed',errors=[],checks=1,
            auditor_sha256=AUDITOR_SHA,manifest_sha256=self.digest('synthetic_session/manifest.json'),
            session_id='synthetic-test-only',source='simulator',acquisition_complete=True,acquisition_status='completed',
            full_time_outputs=2400,partial_points=16800,physical_weight_quality_pass=False,
            counters=dict(impact_attempts=123,programming_pulse_attempts=12))
        self.write('synthetic_audit.json',self.audit)

    def write(self,name,value):(self.repo/name).write_text(json.dumps(value),encoding='utf-8')
    def digest(self,name):return hashlib.sha256((self.repo/name).read_bytes()).hexdigest()
    def verify(self,source='simulator'):
        return verify_session(self.repo,'synthetic_session','synthetic_audit.json',source,list(range(100,110)))

    def test_failed_physical_quality_does_not_block_valid_comparison(self):
        result=self.verify()
        self.assertFalse(result['provenance']['physical_weight_quality_pass'])
        self.assertEqual(set(result['actual']),EXPECTED_KEYS)

    def test_hardware_cannot_be_fabricated_by_reusing_sim_label(self):
        with self.assertRaisesRegex(ValueError,'Wrong backend'):self.verify('hardware')

    def test_summary_tamper_rejected_even_with_passed_audit(self):
        self.summary['trajectories'][0]['actual'][0]['conv_sum']=999.
        self.write('synthetic_session/summary.json',self.summary)
        with self.assertRaisesRegex(ValueError,'Summary hash'):self.verify()

    def test_stale_audit_binding_rejected(self):
        self.audit['manifest_sha256']='0'*64;self.write('synthetic_audit.json',self.audit)
        with self.assertRaisesRegex(ValueError,'bound to this manifest'):self.verify()

    def test_failed_audit_rejected(self):
        self.audit['audit_status']='failed';self.write('synthetic_audit.json',self.audit)
        with self.assertRaisesRegex(ValueError,'did not pass'):self.verify()

    def test_path_escape_rejected(self):
        with self.assertRaisesRegex(ValueError,'escapes'):local(self.repo,'../outside')


if __name__=='__main__':unittest.main()
