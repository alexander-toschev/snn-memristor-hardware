"""Explicitly synthetic unit fixtures. No acquisition or physical measurements."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from compare_cifar_adaptive import (AUDITOR_SHA,PROTOCOL,FIXTURE,FIXTURE_SHA,FIELDS,EXPECTED_KEYS,
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
        acquisition_complete=True,decoded_conv_bn_lif_complete=True,microtiles_completed=28,
        fitted_alpha_used_for_decoding=False,actual_branch_uses_software_dot_product=False,
        diagnostic_acquisition_not_accuracy_qualification=True,full_network_accuracy_claim=False,
        fixed_decoder_gain_v_per_weight=.03,physical_weight_quality_pass=False,
        new_counters=dict(impact_attempts=123,programming_pulse_attempts=12,programming_decisions=12,programming_skipped_current=0),parent_import=None,address_coverage={},trajectories=trajectories)


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


class AdaptiveProvenanceTests(unittest.TestCase):
    def setUp(self):
        base=Path(__file__).resolve().parent/'.tmp';base.mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=base);self.addCleanup(self.temp.cleanup)
        self.repo=Path(self.temp.name).resolve();(self.repo/'synthetic_session').mkdir()
        self.summary=synthetic_summary()
        counts=self.summary['new_counters']
        tiles=[]
        for fid in range(4):
            for parent in range(7):
                taps=list(range(parent*11,min(75,(parent+1)*11)))
                tiles.append(dict(filter_id=fid,parent_tile_id=parent,microtile_id=0,technical_completed=True,effective_route=[{'tap_index':t} for t in taps]))
        self.manifest=dict(synthetic_unittest_not_measurement=True,session_id='synthetic-only',source='simulator',
            status='completed',acquisition_complete=True,protocol_id=PROTOCOL,hardware_access=False,
            trained_coefficients=300,microtiles_completed=28,new_attempted_microtiles=28,
            fixed_decoder_gain_v_per_weight=.03,source_sha256={FIXTURE:FIXTURE_SHA},new_counters=counts,parent_import=None,
            cumulative_impacts_including_all_parent_attempts=123,cumulative_pulses_including_all_parent_attempts=12,
            result_sha256={},invocation={'git':{'commit':'synthetic-only'}})
        self.audit=dict(synthetic_unittest_not_measurement=True,audit_status='passed',errors=[],checks=1,
            auditor_sha256=AUDITOR_SHA,session_id='synthetic-only',source='simulator',protocol_id=PROTOCOL,
            acquisition_complete=True,acquisition_status='completed',full_time_outputs=2400,partial_points=16800,
            physical_weight_quality_pass=False,new_counters=counts,parent_import_audit=None,new_microtiles=tiles,
            coefficient_coverage=dict(coefficients=300,by_filter={str(f):75 for f in range(4)},measured_tiles=28,exact_300_coverage=True))
        self.save()

    def write(self,name,value):(self.repo/name).write_text(json.dumps(value),encoding='utf-8')
    def digest(self,name):return hashlib.sha256((self.repo/name).read_bytes()).hexdigest()
    def save(self):
        self.write('synthetic_session/summary.json',self.summary)
        self.manifest['result_sha256']['summary.json']=self.digest('synthetic_session/summary.json')
        self.write('synthetic_session/manifest.json',self.manifest)
        self.audit['manifest_sha256']=self.digest('synthetic_session/manifest.json')
        self.write('synthetic_audit.json',self.audit)
    def verify(self):return verify_session(self.repo,'synthetic_session','synthetic_audit.json','simulator',list(range(100,110)))

    def test_complete_quality_false_is_allowed(self):
        result=self.verify();self.assertEqual(len(result['actual']),2400)
        self.assertFalse(result['provenance']['physical_weight_quality_pass'])

    def test_variable_partition_is_allowed(self):
        tile=self.audit['new_microtiles'][0];extra=copy.deepcopy(tile)
        extra['microtile_id']=1;extra['effective_route']=tile['effective_route'][-1:];tile['effective_route']=tile['effective_route'][:-1]
        self.audit['new_microtiles'].insert(1,extra)
        self.summary['microtiles_completed']=29
        self.manifest.update(microtiles_completed=29,new_attempted_microtiles=29)
        self.audit['coefficient_coverage']['measured_tiles']=29;self.audit['partial_points']=17400;self.save()
        self.assertEqual(self.verify()['provenance']['completed_microtiles'],29)

    def test_missing_coefficient_rejected_even_claimed_complete(self):
        self.audit['new_microtiles'][0]['effective_route'].pop();self.save()
        with self.assertRaisesRegex(ValueError,'300 coefficients'):self.verify()

    def test_duplicate_coefficient_rejected(self):
        self.audit['new_microtiles'][0]['effective_route'][0]['tap_index']=1;self.save()
        with self.assertRaisesRegex(ValueError,'300 coefficients'):self.verify()

    def test_partial_acquisition_rejected(self):
        self.manifest['status']='failed';self.save()
        with self.assertRaisesRegex(ValueError,'did not complete'):self.verify()

    def test_summary_tamper_rejected(self):
        self.summary['trajectories'][0]['actual'][0]['conv_sum']=999.
        self.write('synthetic_session/summary.json',self.summary)
        with self.assertRaisesRegex(ValueError,'Summary hash'):self.verify()

    def test_wrong_auditor_rejected(self):
        self.audit['auditor_sha256']='unreviewed';self.save()
        with self.assertRaisesRegex(ValueError,'Unreviewed'):self.verify()

    def test_wrong_backend_gain_rejected(self):
        self.manifest['fixed_decoder_gain_v_per_weight']=1.;self.save()
        with self.assertRaisesRegex(ValueError,'backend gain'):self.verify()

    def add_parent(self):
        # Explicit synthetic parent with 100 attempts but only 80 represented.
        (self.repo/'synthetic_parent').mkdir()
        total=dict(impact_attempts=100,programming_pulse_attempts=20,programming_decisions=20,programming_skipped_current=0)
        selected=dict(impact_attempts=80,programming_pulse_attempts=10,programming_decisions=10,programming_skipped_current=0)
        parent=dict(session_id='synthetic-parent',source='simulator',status='failed',protocol_id='cifar-mode10-feedback-full-fixed-gain-diagnostic-v1',result_sha256={})
        self.write('synthetic_parent/manifest.json',parent)
        audit=dict(audit_status='passed',errors=[],source='simulator',session_id='synthetic-parent',
            auditor_sha256='e7367990b6fc2a00f65e3bb0c4e4e5bf097b9f6b21a00700355edb84e0cbc70f',
            manifest_sha256=self.digest('synthetic_parent/manifest.json'),counters=total)
        self.write('synthetic_parent_audit.json',audit)
        binding=dict(session='synthetic_parent',manifest_sha256=self.digest('synthetic_parent/manifest.json'),audit_path='synthetic_parent_audit.json',
            audit_sha256=self.digest('synthetic_parent_audit.json'),result_sha256={},imported_prefix=['f0_t0','f0_t1','f0_t2'],
            imported_trained_coefficients=42,failed_parent_tile_contributes=False,other_parent_tiles_ignored=True,
            parent_total_impacts=100,parent_total_pulses=20)
        historical=dict(session='synthetic_parent',session_id='synthetic-parent',source='simulator',acquisition_status='failed',
            manifest_sha256=binding['manifest_sha256'],audit_sha256=binding['audit_sha256'],imported_coefficients=42,imported_partial_points=1800,
            imported_tiles=3,unimported_attempts_not_erased=True,all_parent_attempt_counters=total,selected_tile_counters=selected)
        # Four new chunks finish filter0 after its imported three chunks.
        tail=[dict(filter_id=0,parent_tile_id=p,microtile_id=0,technical_completed=True,effective_route=[{'tap_index':t} for t in taps])
              for p,taps in enumerate((range(42,51),range(51,60),range(60,69),range(69,75)),3)]
        self.audit['new_microtiles']=tail+self.audit['new_microtiles'][7:]
        self.manifest.update(parent_import=binding,trained_coefficients=258,new_attempted_microtiles=25,
            cumulative_impacts_including_all_parent_attempts=223,cumulative_pulses_including_all_parent_attempts=32)
        self.summary['parent_import']=copy.deepcopy(binding);self.audit['parent_import_audit']=historical;self.save()

    def test_parent_failed_attempts_retained_separately(self):
        self.add_parent();p=self.verify()['provenance']
        self.assertEqual(p['new_counters']['impact_attempts'],123)
        self.assertEqual(p['parent_import']['selected_tile_counters']['impact_attempts'],80)
        self.assertEqual(p['cumulative_including_all_parent_attempts']['impact_attempts'],223)

    def test_using_only_imported_parent_attempts_in_total_rejected(self):
        self.add_parent();self.manifest['cumulative_impacts_including_all_parent_attempts']=203;self.save()
        with self.assertRaisesRegex(ValueError,'Cumulative impacts'):self.verify()

    def test_parent_manifest_tamper_rejected(self):
        self.add_parent();self.write('synthetic_parent/manifest.json',{'tampered':True})
        with self.assertRaisesRegex(ValueError,'Parent manifest hash'):self.verify()


if __name__=='__main__':unittest.main()
