"""Synthetic-only coverage tests; no acquisition or device imports."""
import copy
import json
from pathlib import Path
import unittest
import hashlib
from unittest.mock import patch
import audit_cifar_feedback_adaptive as adaptive
from audit_cifar_feedback_adaptive import audit_coefficient_coverage,reference_rank,adaptive_matrix_select,adaptive_budget


def synthetic_geometry():
    routing={'filters':[]};results=[]
    for fid in range(4):
        filt=dict(filter_id=fid,tiles=[])
        for parent,start in enumerate(range(0,75,14)):
            slots=[dict(tap_index=t,trained_weight=(t+1)/1000,padding=False,
                positive_cell=[2*fid,t-start],negative_cell=[2*fid+1,t-start]) for t in range(start,min(start+14,75))]
            filt['tiles'].append(dict(tile_id=parent,slots=copy.deepcopy(slots)))
            # Explicitly synthetic adaptive split, including a singleton second microtile.
            for micro,group in enumerate((slots[:-1],slots[-1:])):
                if not group:continue
                results.append(dict(filter_id=fid,parent_tile_id=parent,microtile_id=micro,
                    tile_id=2*parent+micro,technical_completed=True,effective_route=copy.deepcopy(group)))
        routing['filters'].append(filt)
    return routing,results


class AdaptiveCoverageTests(unittest.TestCase):
    def test_variable_groups_still_cover_300_exactly(self):
        routing,results=synthetic_geometry();report=audit_coefficient_coverage(results,routing,True)
        self.assertEqual(report['coefficients'],300);self.assertTrue(report['exact_300_coverage'])
        self.assertEqual(report['by_filter'],{str(f):75 for f in range(4)})

    def test_imported_raw_identity_may_differ_but_wrapper_identity_is_unique(self):
        routing,results=synthetic_geometry();results[2]['tile_id']=1
        self.assertTrue(audit_coefficient_coverage(results,routing,True)['exact_300_coverage'])

    def test_missing_tap_cannot_be_called_complete(self):
        routing,results=synthetic_geometry();results.pop()
        with self.assertRaisesRegex(ValueError,'omits'):audit_coefficient_coverage(results,routing,True)

    def test_duplicate_parent_import_and_new_measurement_rejected(self):
        routing,results=synthetic_geometry();results[1]['effective_route']=[copy.deepcopy(results[0]['effective_route'][0])]
        with self.assertRaisesRegex(ValueError,'twice'):audit_coefficient_coverage(results,routing,True)

    def test_replaced_weight_rejected(self):
        routing,results=synthetic_geometry();results[0]['effective_route'][0]['trained_weight']=999.
        with self.assertRaisesRegex(ValueError,'replaced'):audit_coefficient_coverage(results,routing,True)

    def test_hole_is_not_a_valid_partial_prefix(self):
        routing,results=synthetic_geometry();results[0]['effective_route'].pop(2)
        with self.assertRaisesRegex(ValueError,'ordered prefix'):audit_coefficient_coverage(results,routing,False)

    def test_partial_prefix_is_reported_without_completion(self):
        routing,results=synthetic_geometry();report=audit_coefficient_coverage(results[:1],routing,False)
        self.assertEqual(report['coefficients'],13);self.assertFalse(report['exact_300_coverage'])

    def test_zero_singleton_and_two_dimensional_reference_rank(self):
        self.assertEqual(reference_rank([[0.,0.],[0.,0.]]),0)
        self.assertEqual(reference_rank([[.1,.2],[.2,.4],[0.,0.]]),1)
        self.assertEqual(reference_rank([[.1,0.],[0.,.2],[.1,.2]]),2)

    def test_noisy_measured_rank_cannot_certify_rank_one_reference(self):
        train=[dict(measured_mean_v=x,reference_v=y) for x,y in
               [([.1,.02],[.1,.2]),([.05,.15],[.2,.4]),([.2,.1],[.3,.6])]]
        selection=[dict(measured_repeats_v=[[.1,.02],[.1,.02],[.1,.02]],reference_v=[.1,.2])]
        result=adaptive_matrix_select(train,selection)
        self.assertEqual(result['training_design_rank'],2)
        self.assertEqual(result['reference_design_rank'],1)
        full=[c for c in result['candidates'] if c['name'].startswith('full_')]
        self.assertTrue(full);self.assertTrue(all(not c['eligible'] for c in full))
        self.assertNotIn(result['selected_model'],[c['name'] for c in full])

    def test_reviewed_real_fixture_upper_bounds_from_all_splits(self):
        root=Path(__file__).resolve().parent
        routing=json.loads((root/'weights/candidate_cifar75_routing.json').read_text(encoding='utf-8'))
        fixture=json.loads((root/'weights/prepared_cifar100_conv1.json').read_text(encoding='utf-8'))
        full=adaptive_budget(routing,fixture);resume=adaptive_budget(routing,fixture,True)
        self.assertEqual((full['maximum_impacts'],full['maximum_programming_pulses']),(868466,117180))
        self.assertEqual((resume['maximum_impacts'],resume['maximum_programming_pulses']),(756104,101382))

    def test_third_microtile_rejected(self):
        routing,results=synthetic_geometry();results[0]['microtile_id']=2
        with self.assertRaisesRegex(ValueError,'microtile index'):audit_coefficient_coverage(results,routing,True)

    def test_wrong_original_chunk_rejected(self):
        routing,results=synthetic_geometry();results[0]['parent_tile_id']=1
        with self.assertRaisesRegex(ValueError,'wrong original chunk'):audit_coefficient_coverage(results,routing,True)

    def test_changed_parent_commitment_rejected_before_replay(self):
        from audit_cifar_feedback_adaptive import verify_parent_prefix,CompactChecks
        root=Path(__file__).resolve().parent
        session='results/cifar_feedback_diagnostic/e1e1d0a7-00e2-403b-ac4e-defbf1fe80b8'
        with self.assertRaisesRegex(ValueError,'manifest commitment'):
            verify_parent_prefix(root,session,'unused','0'*64,'0'*64,{},None,None,CompactChecks())


class NonexecutedValidationProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.files={name:b'# synthetic later validator snapshot\n' for name in adaptive.NONEXECUTED_VALIDATION_FILES}
        self.pins={name:hashlib.sha256(data).hexdigest() for name,data in self.files.items()}
        self.names=sorted(self.files)
        self.git=dict(commit=adaptive.VALIDATION_START_COMMIT,dirty_tracked=False,
            untracked=self.names[:],status_porcelain='\n'.join('?? '+name for name in self.names))
        self.blobs={'collector.py':b'import math\n'}
        self.sources={name:hashlib.sha256(data).hexdigest() for name,data in self.blobs.items()}

    def run_check(self):
        ck=adaptive.CompactChecks()
        with patch.dict(adaptive.NONEXECUTED_VALIDATION_FILES,self.pins,clear=True):
            result=adaptive.validation_provenance(self.git,self.sources,self.blobs,self.files,ck)
        return ck,result

    def test_exact_pair_preserves_original_status_and_unknown_launch_contents(self):
        ck,result=self.run_check();self.assertEqual(ck.errors,[])
        self.assertEqual(result['recorded_git_status'],self.git)
        self.assertFalse(result['later_validation_snapshot']['proves_launch_contents'])
        self.assertIn('unknown',result['untracked_validation_contents_at_launch'])

    def test_additional_untracked_python_is_rejected(self):
        self.git['untracked'].append('unknown.py');self.git['status_porcelain']+='\n?? unknown.py'
        ck,_=self.run_check();self.assertTrue(any('unknown.py' in e for e in ck.errors))

    def test_renamed_validator_is_rejected(self):
        self.git['untracked'][0]='renamed.py';self.git['status_porcelain']=self.git['status_porcelain'].replace(self.names[0],'renamed.py')
        with self.assertRaisesRegex(ValueError,'exact reviewed pair'):self.run_check()

    def test_collector_import_makes_exception_invalid(self):
        self.blobs['collector.py']=b'import audit_cifar_feedback_adaptive\n'
        self.sources['collector.py']=hashlib.sha256(self.blobs['collector.py']).hexdigest()
        with self.assertRaisesRegex(ValueError,'imports excluded'):self.run_check()

    def test_source_inventory_cannot_include_validator(self):
        self.sources[self.names[0]]=self.pins[self.names[0]]
        with self.assertRaisesRegex(ValueError,'source inventory'):self.run_check()

    def test_later_snapshot_does_not_allow_mutation(self):
        self.files[self.names[0]]=b'tampered'
        with self.assertRaisesRegex(ValueError,'Later validator'):self.run_check()

    def test_tracked_runtime_dirty_still_rejected(self):
        self.git['dirty_tracked']=True;self.git['status_porcelain']+='\n M collector.py'
        ck,_=self.run_check();self.assertTrue(any('clean_or_exact' in e for e in ck.errors))


if __name__=='__main__':unittest.main()
