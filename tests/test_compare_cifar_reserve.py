"""Offline tests: bound actual native SIM plus negative provenance/geometry cases."""
import copy
import importlib.util
from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('reserve_compare', (Path(__file__).resolve().parents[1]/'compare_cifar_reserve.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
SESSION = 'results/cifar_feedback_reserve/55134577-a813-4d1b-a3de-67c81ab285ca'
AUDIT = 'results/cifar_feedback_reviews/reserve_simulator_20261016/audit.json'


class ReserveCompareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT/'.tmp').mkdir(exist_ok=True)
        cls.h = m.helper(ROOT)
        cls.fixture = m.load(ROOT/cls.h.FIXTURE)
        cls.indices = [c['original_test_index'] for c in cls.fixture['cases']]
        cls.valid = cls.verify()

    @classmethod
    def verify(cls):
        return m.verify_session(ROOT, SESSION, AUDIT, 'simulator', cls.indices, cls.h)

    def modified_json(self, suffix, mutate):
        original = m.load
        def reader(path):
            value = original(path)
            if path.as_posix().endswith(suffix):
                mutate(value)
            return value
        return patch.object(m, 'load', reader)

    def test_actual_native_sim_bound_13_plus_287(self):
        p = self.valid['provenance']
        self.assertEqual((p['new_coefficients'], p['retained_historical_coefficients'], len(self.valid['actual'])), (13,287,2400))
        self.assertEqual(p['fresh13_partials']['points'], 600)
        self.assertEqual(p['fresh13_metrics']['coefficients'], 13)
        self.assertEqual(p['all300_metrics']['coefficients'], 300)

    def test_unreviewed_auditor_rejected(self):
        with patch.object(m, 'RESERVE_AUDITOR_SHA', '0'*64), self.assertRaisesRegex(ValueError, 'Unreviewed reserve auditor'):
            self.verify()

    def test_unbound_audit_rejected(self):
        with self.modified_json(AUDIT, lambda a:a.update(manifest_sha256='0'*64)), self.assertRaisesRegex(ValueError,'Audit manifest mismatch'):
            self.verify()

    def test_wrong_gain_rejected(self):
        with self.modified_json(SESSION+'/manifest.json', lambda a:a.update(fixed_decoder_gain_v_per_weight=1.0)), self.assertRaisesRegex(ValueError,'Fixed gain mismatch'):
            self.verify()

    def test_unknown_protocol_bc_not_silently_accepted(self):
        with self.modified_json(SESSION+'/manifest.json', lambda a:a.update(protocol_id='reserve-B')), self.assertRaisesRegex(ValueError,'Unreviewed reserve protocol'):
            self.verify()

    def test_double_counted_ancestry_rejected(self):
        def change(a):
            a['cumulative_impacts_including_all_ancestry'] += 1
        with self.modified_json(SESSION+'/manifest.json', change), self.assertRaisesRegex(ValueError,'Ancestry omitted/double counted'):
            self.verify()

    def test_duplicate_historical_wrapper_rejected(self):
        with self.modified_json(SESSION+'/imported_microtiles.json', lambda a:a.__setitem__(1,copy.deepcopy(a[0]))), self.assertRaisesRegex(ValueError,'Duplicate microtile'):
            self.verify()

    def test_old_excluded_tap_cannot_replace_fresh(self):
        def change(a):
            a[-1]['result']['effective_route'][0]['tap_index']=51
        with self.modified_json(SESSION+'/completed_microtiles.json', change), self.assertRaisesRegex(ValueError,'Fresh13 result mismatch'):
            self.verify()

    def test_partial_report_never_fills_full_comparison(self):
        failed = {'provenance':copy.deepcopy(self.valid['provenance'])}
        failed['provenance'].update(acquisition_complete=False,new_coefficients=0,full_output_points=0,fresh_output_points=0)
        report=m.make_report(self.valid,failed,self.h.trained_reference(self.fixture),self.h)
        self.assertFalse(report['comparison_complete'])
        self.assertIsNone(report['comparisons'])
        self.assertNotIn('fresh13_hardware_vs_simulator',report)
        self.assertIn('excluded old13 are never restored',m.markdown(report))

    def test_coverage_union_can_displace_not_monotonically_grow(self):
        def wrapper(cell):
            return dict(result=dict(reset_candidates=[],pairs=[],effective_route=[dict(positive_cell=list(cell),negative_cell=[7,0])]))
        old=m.address_sets([wrapper((6,1))])['assigned']['cells']
        new=m.address_sets([wrapper((6,2))])['assigned']['cells']
        self.assertEqual(len(new),len(old))
        self.assertNotIn([6,1],new)

    def test_eligible_but_route_failed_not_claimed_measured(self):
        tile=dict(reset_candidates=[dict(cell=[7,15],eligible=True)],effective_route=[],pairs=[])
        row=[r for r in m.dispositions(tile,False) if r['cell']==[7,15]][0]
        self.assertEqual(row['disposition'],'freshly_quiet_without_complete_permitted_route')
        self.assertFalse(row['accurate_weight_claim'])

    def test_duplicate_fresh_partial_rejected(self):
        tile=m.load(ROOT/SESSION/'tiles/f3_p4_reserve/tile.json')
        tile['partials'][1]=copy.deepcopy(tile['partials'][0])
        with self.assertRaisesRegex(ValueError,'Duplicate fresh partial'):
            m.partial_metrics(tile,self.h)

    def test_native_failed_manifest_omits_acquisition_flag(self):
        # A's initial/failure manifest has status=failed and no acquisition_complete.
        manifest={'status':'failed'}
        audit={'acquisition_status':'failed','acquisition_complete':False}
        with tempfile.TemporaryDirectory(dir=ROOT/".tmp") as directory:
            self.assertFalse(m.acquisition_complete(manifest,audit,Path(directory)))
            self.assertNotIn('acquisition_complete',manifest)
            (Path(directory)/'summary.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'Failed route restored'):
                m.acquisition_complete(manifest,audit,Path(directory))

    def test_completed_manifest_requires_explicit_true(self):
        with tempfile.TemporaryDirectory(dir=ROOT/".tmp") as directory:
            with self.assertRaisesRegex(ValueError,'lacks explicit'):
                m.acquisition_complete({'status':'completed'},
                    {'acquisition_status':'completed','acquisition_complete':True},Path(directory))

    def test_failed_manifest_null_or_true_flag_not_missing(self):
        with tempfile.TemporaryDirectory(dir=ROOT/".tmp") as directory:
            for flag in (None,True):
                with self.subTest(flag=flag), self.assertRaisesRegex(ValueError,'Failed manifest acquisition flag differs'):
                    m.acquisition_complete({'status':'failed','acquisition_complete':flag},
                        {'acquisition_status':'failed','acquisition_complete':False},Path(directory))


if __name__=='__main__':
    unittest.main()
