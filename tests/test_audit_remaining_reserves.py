"""Offline independent-auditor tests; synthetic saved logs, no acquisition imports."""
import copy
import json
from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch
import audit_cifar_feedback_remaining_reserves as audit

HERE=Path(__file__).resolve().parents[1]
PUBLIC=HERE if (HERE/'audit_cifar_feedback_reserve.py').exists() else HERE.parent/'hardware-public'
DRAFT=PUBLIC.parent/'reserve_remaining_draft'
audit.initialize(PUBLIC)
ROUTING=audit.load(PUBLIC/'weights/candidate_cifar75_routing.json')
FIXTURE=audit.load(PUBLIC/'weights/prepared_cifar100_conv1.json')
PLAN_PATH=DRAFT/'configs/cifar_feedback_remaining_reserves.json'
PLAN=audit.load(PLAN_PATH) if PLAN_PATH.exists() else None
SYNTHETIC=DRAFT/'synthetic_fixtures'
HAS_RAW_FIXTURES=PLAN is not None and (SYNTHETIC/'success/FIXTURE.json').exists() and (SYNTHETIC/'capacity_failure/FIXTURE.json').exists()
A_SESSION=PUBLIC/'results/cifar_feedback_reserve/55134577-a813-4d1b-a3de-67c81ab285ca'
HAS_A_PARENT=(A_SESSION/'completed_microtiles.json').exists()

class RemainingAuditTests(unittest.TestCase):
    def contract(self,status='completed'):
        names=('protocol.json','board.ini','start_manifest.json','imported_microtiles.json',
            'completed_microtiles.json','summary.json','capacity_failures.json')
        return dict(status=status,result_sha256={n:'unused-test-digest' for n in names},
            maximum_impacts=40378,maximum_programming_pulses=5260,maximum_programming_decisions=5260,
            new_trained_coefficients=20,imported_trained_coefficients=280)

    def test_missing_completed_result_commitments_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            folder=Path(tmp)
            for name in ('completed_microtiles.json','summary.json'):
                with self.subTest(name=name):
                    m=self.contract();del m['result_sha256'][name]
                    with self.assertRaisesRegex(ValueError,'Unbound required file'):
                        audit.verify_manifest_contract(folder,m,audit.CompactChecks())

    def test_partial_root_artifacts_must_be_hash_bound(self):
        for name in ('completed_microtiles.json','capacity_failures.json','summary.json'):
            with self.subTest(name=name),tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
                folder=Path(tmp);(folder/name).write_text('{}')
                m=self.contract('failed');del m['result_sha256'][name]
                with self.assertRaisesRegex(ValueError,'Unbound required file'):
                    audit.verify_manifest_contract(folder,m,audit.CompactChecks())

    def test_each_declared_budget_or_coefficient_count_is_fixed(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            folder=Path(tmp);ck=audit.CompactChecks()
            audit.verify_manifest_contract(folder,self.contract(),ck);self.assertEqual(ck.errors,[])
            for key in ('maximum_impacts','maximum_programming_pulses','maximum_programming_decisions',
                        'new_trained_coefficients','imported_trained_coefficients'):
                with self.subTest(key=key):
                    m=self.contract();m[key]+=1;ck=audit.CompactChecks()
                    audit.verify_manifest_contract(folder,m,ck)
                    self.assertTrue(ck.errors)

    def data(self,stage):
        parent=ROUTING['filters'][audit.SPECS[stage]['fid']]['tiles'][0]
        return parent,audit.block_pools(parent,stage)

    def test_exact_budgets_from_real_fixture(self):
        totals=[0,0]
        for stage in ('B','C'):
            parent,_=self.data(stage);b=audit.block_budget(parent,stage,FIXTURE)
            self.assertEqual(b,dict(maximum_impacts=audit.SPECS[stage]['calls'],maximum_writes=audit.SPECS[stage]['writes']))
            totals[0]+=b['maximum_impacts'];totals[1]+=b['maximum_writes']
        self.assertEqual(totals,[40378,5260])

    def test_all_preferences_reserved_before_free_fill(self):
        for stage in ('B','C'):
            parent,eligible=self.data(stage);d=audit.allocation(parent,stage,eligible)
            for side in ('positive','negative'):
                for tap,cell in audit.SPECS[stage][side].items():
                    self.assertEqual(d['route'][side+'_cell'][tap],cell)
                    self.assertEqual(d['allocation_before_fill'][side+'_cell'][tap],cell)
                self.assertEqual(len(set(map(tuple,d['route'][side+'_cell']))),audit.SPECS[stage]['n'])

    def test_missed_preference_not_global_failure(self):
        parent,eligible=self.data('C');eligible['negative_cell'].remove([3,12])
        d=audit.allocation(parent,'C',eligible)
        self.assertTrue(d['capacity_pass']);self.assertNotIn([3,12],d['route']['negative_cell'])
        self.assertEqual(d['route']['negative_cell'][1],[3,13]);self.assertEqual(d['route']['negative_cell'][2],[3,31])

    def test_insufficient_capacity_no_SET_route(self):
        parent,eligible=self.data('B');eligible['positive_cell']=eligible['positive_cell'][:10]
        d=audit.allocation(parent,'B',eligible)
        self.assertFalse(d['may_issue_SET']);self.assertEqual(d['route'],{'positive_cell':[],'negative_cell':[]})

    def test_duplicate_unknown_reordered_eligibility_reject(self):
        for mode in ('duplicate','unknown','reverse'):
            parent,eligible=self.data('C')
            if mode=='duplicate':eligible['negative_cell'].append(eligible['negative_cell'][0])
            elif mode=='unknown':eligible['negative_cell'].append([3,30])
            else:eligible['negative_cell'].reverse()
            with self.assertRaises(ValueError):audit.allocation(parent,'C',eligible)

    def test_failed_A_counts_new_only_not_cumulative(self):
        parent={'cumulative_impacts_including_all_parent_attempts':1000,'cumulative_pulses_including_all_parent_attempts':100}
        failed={'status':'failed','parent_import':{'session':'base'},'new_counters':{'impact_attempts':23,'programming_pulse_attempts':7},
            'cumulative_impacts_including_all_ancestry':1023,'cumulative_pulses_including_all_ancestry':107}
        totals=audit.historical_totals('base',parent,('failedA',failed))
        self.assertEqual((totals['parent_cumulative_impacts'],totals['parent_cumulative_pulses']),(1023,107))
        with self.assertRaises(ValueError):audit.historical_totals('base',parent,('base',failed))
        failed['parent_import']['session']='other'
        with self.assertRaises(ValueError):audit.historical_totals('base',parent,('failedA',failed))

    def test_actual_session_cannot_pass_without_final_pins(self):
        with patch.object(audit,'FINAL_PINS',{}):
            with self.assertRaisesRegex(ValueError,'not frozen'):
                audit.verify_manifest(PUBLIC,PUBLIC,audit.CompactChecks())

    def test_final_source_bytes_and_actual_parent_bindings(self):
        for name,digest in audit.FINAL_PINS.items():
            self.assertEqual(audit.sha(PUBLIC/name),digest)
        plan=audit.load(PUBLIC/'configs/cifar_feedback_remaining_reserves.json')
        self.assertEqual(plan['protocol_id'],audit.FINAL_PROTOCOL)
        self.assertFalse(plan['draft']);self.assertTrue(plan['native_execution_enabled'])
        for source,binding in plan['parent_bindings'].items():
            self.assertEqual(binding['source'],source);self.assertEqual(binding['choice'],'completed_A')
            self.assertEqual(audit.sha(PUBLIC/binding['session']/'manifest.json'),binding['manifest_sha256'])
            self.assertEqual(audit.sha(PUBLIC/binding['audit_path']),binding['audit_sha256'])

    @unittest.skipUnless(HAS_A_PARENT,'Audited historical A session is not present')
    def test_exact280_import_excludes_all_old_BC(self):
        folder=PUBLIC/'results/cifar_feedback_reserve/55134577-a813-4d1b-a3de-67c81ab285ca'
        original=audit.load(folder/'completed_microtiles.json')
        kept=audit.imported280(original,'parent',audit.sha(folder/'completed_microtiles.json'),ROUTING)
        self.assertEqual(sum(len(w['result']['effective_route']) for w in kept),280)
        self.assertFalse(any((w['filter_id'],w['parent_tile_id']) in ((1,0),(2,0)) for w in kept))
        bad=copy.deepcopy(original);bad.append(copy.deepcopy(original[0]))
        with self.assertRaises(ValueError):audit.imported280(bad,'parent','abc',ROUTING)

    @unittest.skipUnless(HAS_RAW_FIXTURES,'Optional saved synthetic journals are not present in this checkout')
    def test_mutated_route_and_contribution_detected_from_raw(self):
        folder=SYNTHETIC/'success/tiles/f2_p0_reserve_B';original_load=audit.load
        def changed(path):
            value=original_load(path)
            if path==folder/'tile.json':value['partials'][0]['trained_unit_contribution']+=1
            if path==folder/'frozen_route.json':value['slots'][0]['positive_cell']=[4,0]
            return value
        ck=audit.CompactChecks()
        with patch.object(audit,'load',side_effect=changed):
            audit.audit_block(folder,ROUTING['filters'][2],ROUTING['filters'][2]['tiles'][0],FIXTURE,PLAN,'simulator',ck,'SYNTHETIC_success','B')
        self.assertTrue(any('tile/partials' in str(e) for e in ck.errors))
        self.assertTrue(any('routing/frozen_values' in str(e) for e in ck.errors))

    @unittest.skipUnless(HAS_RAW_FIXTURES,'Optional saved synthetic journals are not present in this checkout')
    def test_fabricated_CIFAR_on_failed_B_rejected(self):
        folder=SYNTHETIC/'capacity_failure/tiles/f2_p0_reserve_B';original_load=audit.load
        def changed(path):
            value=original_load(path)
            if path==folder/'tile.json':value['partials']=[{'fake':'CIFAR'}]
            return value
        ck=audit.CompactChecks()
        with patch.object(audit,'load',side_effect=changed):
            audit.audit_block(folder,ROUTING['filters'][2],ROUTING['filters'][2]['tiles'][0],FIXTURE,PLAN,'simulator',ck,'SYNTHETIC_capacity_failure','B')
        self.assertTrue(any('no_partials_on_gate_failure' in str(e) for e in ck.errors))

    def test_bound_ancestor_file_tamper_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            repo=Path(tmp);folder=repo/'parent';folder.mkdir()
            payload=folder/'sample.json';payload.write_text('{}')
            m=dict(source='simulator',session_id='parent-id',status='completed',protocol_id=audit.A.PROTOCOL,
                new_counters={'impact_attempts':1},result_sha256={'sample.json':audit.sha(payload)})
            mp=folder/'manifest.json';mp.write_text(json.dumps(m))
            a=dict(source='simulator',session_id='parent-id',audit_status='passed',errors=[],acquisition_complete=True,
                new_counters=m['new_counters'],full_time_outputs=2400,auditor_sha256=audit.A_SHA,manifest_sha256=audit.sha(mp))
            ap=repo/'audit.json';ap.write_text(json.dumps(a))
            binding=dict(session='parent',audit_path='audit.json',manifest_sha256=audit.sha(mp),audit_sha256=audit.sha(ap))
            ck=audit.CompactChecks();audit.bound_evidence(repo,binding,'simulator',True,ck);self.assertEqual(ck.errors,[])
            payload.write_text('{"tampered":true}')
            ck=audit.CompactChecks();audit.bound_evidence(repo,binding,'simulator',True,ck);self.assertTrue(ck.errors)
            with self.assertRaises(ValueError):audit.bound_evidence(repo,binding,'hardware',True,audit.CompactChecks())

    @unittest.skipUnless(HAS_RAW_FIXTURES and HAS_A_PARENT,'Historical A and saved synthetic fixtures are required')
    def test_synthetic280_plus20_exact_full_coverage(self):
        folder=PUBLIC/'results/cifar_feedback_reserve/55134577-a813-4d1b-a3de-67c81ab285ca'
        original=audit.load(folder/'completed_microtiles.json')
        kept=audit.imported280(original,'parent',audit.sha(folder/'completed_microtiles.json'),ROUTING)
        # Pure arithmetic fixture composition, never saved or claimed as acquisition.
        fresh=audit.load(SYNTHETIC/'success/completed_microtiles.json')
        combined=kept+fresh
        coverage=audit.verify_wrapper_coverage(combined,ROUTING)
        self.assertEqual(coverage['coefficients'],300)
        ck=audit.CompactChecks()
        trajectories,quality=audit.aggregate_microtiles([audit.normalized_result(w) for w in audit.ordered(combined)],FIXTURE,ROUTING,ck)
        self.assertEqual(len(trajectories)*12,2400);self.assertEqual(ck.errors,[])
        with self.assertRaises(ValueError):audit.verify_wrapper_coverage(kept+fresh[:1],ROUTING)

    @unittest.skipUnless(HAS_RAW_FIXTURES,'Optional saved synthetic journals are not present in this checkout')
    def test_wrong_source_marker_rejected(self):
        original_load=audit.load
        def changed(path):
            value=original_load(path)
            if path.name=='FIXTURE.json':value['hardware']=True
            return value
        with patch.object(audit,'load',side_effect=changed):
            with self.assertRaisesRegex(ValueError,'explicitly synthetic'):
                audit.audit_synthetic(SYNTHETIC/'success',FIXTURE,ROUTING,PLAN,audit.CompactChecks())

if __name__=='__main__':unittest.main()
