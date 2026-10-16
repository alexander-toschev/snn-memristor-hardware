"""Pure draft tests; these do not claim any native B/C acquisition occurred."""
import copy
from contextlib import contextmanager
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parent
SESSION='results/cifar_feedback_remaining_reserves/99631af1-c73e-42ed-abcd-20d4e580f691'
AUDIT='results/cifar_feedback_reviews/remaining_simulator_20261016/audit.json'
spec=importlib.util.spec_from_file_location('bc_compare',Path(__file__).with_name('compare_cifar_remaining.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class ComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a,cls.h=m.pure_helpers(ROOT)
        folder=ROOT/'results/cifar_feedback_reserve/e17464e9-a601-4b7e-adf5-8a1d56de41d2'
        manifest=cls.a.load(folder/'manifest.json')
        cls.wrappers=cls.a.bound_json(folder,manifest,'completed_microtiles.json')
        fixture=cls.a.load(ROOT/cls.h.FIXTURE)
        cls.indices=[c['original_test_index'] for c in fixture['cases']]

    @classmethod
    def verify_actual(cls):
        return m.verify(ROOT,SESSION,AUDIT,'simulator',cls.indices,cls.a,cls.h)

    @contextmanager
    def modified_json(self,suffix,mutate):
        original=self.a.load
        original_bound=self.a.bound_json
        def read(path):
            value=original(path)
            if path.as_posix().endswith(suffix):mutate(value)
            return value
        def bound(directory,manifest,name):
            value=original_bound(directory,manifest,name)
            if (directory/name).as_posix().endswith(suffix):mutate(value)
            return value
        with patch.object(self.a,'load',read),patch.object(self.a,'bound_json',bound):yield

    def test_actual_BC_SIM_exact300_and33_267_provenance(self):
        data=self.verify_actual();p=data['provenance']
        self.assertEqual((len(data['actual']),p['covered_coefficients'],p['new_BC_coefficients']),(2400,300,20))
        self.assertEqual(p['final33_weight_metrics']['coefficients'],33)
        self.assertEqual(p['weight_metrics']['original267']['coefficients'],267)
        self.assertEqual(len(p['ten_reserve_dispositions']),10)

    def test_BC_auditor_pin_rejected(self):
        with patch.object(m,'AUDITOR_SHA','0'*64),self.assertRaisesRegex(ValueError,'Unreviewed/synthetic'):
            self.verify_actual()

    def test_BC_audit_manifest_binding_rejected(self):
        with self.modified_json(AUDIT,lambda d:d.update(manifest_sha256='0'*64)),self.assertRaisesRegex(ValueError,'Audit manifest binding'):
            self.verify_actual()

    def test_BC_ancestry_double_count_rejected(self):
        def change(d):d['cumulative_impacts_including_all_ancestry']+=1
        with self.modified_json(SESSION+'/manifest.json',change),self.assertRaisesRegex(ValueError,'Ancestry double count'):
            self.verify_actual()

    def test_BC_duplicate_import_rejected(self):
        with self.modified_json(SESSION+'/imported_microtiles.json',lambda d:d.__setitem__(1,copy.deepcopy(d[0]))),self.assertRaisesRegex(ValueError,'Duplicate microtile'):
            self.verify_actual()

    def test_BC_actual_summary_missing_gain_is_valid_but_changed_gain_rejected(self):
        with self.modified_json(SESSION+'/summary.json',lambda d:d.update(fixed_decoder_gain_v_per_weight=1)),self.assertRaisesRegex(ValueError,'Summary fixed gain'):
            self.verify_actual()

    def test_actual_A_partition_geometry_is267_13_20(self):
        groups=m.partition(self.wrappers,self.a)
        self.assertEqual({k:len(m.keys(v,self.a)) for k,v in groups.items()},dict(original267=267,reserve_A13=13,reserve_BC20=20))

    def test_partial_import_keeps280_not300(self):
        chunks=[w for w in self.wrappers if (w['filter_id'],w['parent_tile_id']) not in ((2,0),(1,0))]
        groups=m.partition(chunks,self.a)
        self.assertEqual(len(m.keys(chunks,self.a)),280)
        self.assertEqual(groups['reserve_BC20'],[])

    def test_one_block_completion_cannot_mean300(self):
        chunks=[w for w in self.wrappers if (w['filter_id'],w['parent_tile_id'])!=(1,0)]
        self.assertEqual(len(m.keys(chunks,self.a)),291)

    def test_duplicate_import_rejected(self):
        with self.assertRaisesRegex(ValueError,'Duplicate coefficient'):
            m.partition(self.wrappers+[self.wrappers[0]],self.a)

    def test_stage_B_has_three_stage_C_four_reserves(self):
        tile=dict(reset_candidates=[],effective_route=[],pairs=[])
        b=m.dispositions('B',tile,False);c=m.dispositions('C',tile,False)
        self.assertEqual((len(b),len(c)),(3,4))
        self.assertTrue(all(r['status']=='technical_unassessed' and not r['actual_cifar_measured'] for r in b+c))

    def test_quiet_capacity_failure_not_use_or_dead(self):
        tile=dict(reset_candidates=[dict(cell=[3,12],eligible=True)],effective_route=[],pairs=[])
        row=m.dispositions('C',tile,False)[1]
        self.assertEqual(row['cell'],[3,12])
        self.assertEqual(row['status'],'freshly_quiet_route_incomplete')
        self.assertFalse(row['actual_cifar_measured'])

    def test_nonquiet_disposition_explicit(self):
        tile=dict(reset_candidates=[dict(cell=[5,15],eligible=False)],effective_route=[],pairs=[])
        self.assertEqual(m.dispositions('B',tile,False)[2]['status'],'not_freshly_quiet')

    def test_duplicate_candidate_rejected(self):
        c=dict(cell=[5,15],eligible=True)
        with self.assertRaisesRegex(ValueError,'Duplicate RESET'):
            m.dispositions('B',dict(reset_candidates=[c,c],effective_route=[],pairs=[]),False)

    def test_partial_map_labeled_without_final300_claim(self):
        p=dict(source='hardware',acquisition_complete=False,contributing_address_union=self.a.address_sets(self.wrappers))
        svg=m.svg_map(p)
        self.assertIn('PARTIAL measured union',svg)
        self.assertIn('not final conductance',svg)
        self.assertNotIn('complete300 contributing union',svg)

    def test_unreviewed_pure_helper_rejected(self):
        with patch.object(m,'A_HELPER_SHA','0'*64),self.assertRaisesRegex(ValueError,'Reviewed A comparator changed'):
            m.pure_helpers(ROOT)


if __name__=='__main__':unittest.main()
