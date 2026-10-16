"""Synthetic-only reserve gate and immutable replacement tests; no drivers."""
import copy
import json
from pathlib import Path
import unittest
from audit_cifar_feedback_reserve import reserve_pools,reserve_route,reserve_budget,verify_wrapper_coverage,selected_imports


def inputs():
    root=Path(__file__).resolve().parent
    return json.loads((root/'weights/candidate_cifar75_routing.json').read_text(encoding='utf-8')),json.loads((root/'weights/prepared_cifar100_conv1.json').read_text(encoding='utf-8'))


def synthetic_parent(split):
    routing,_=inputs();wrappers=[]
    for f in routing['filters']:
        for t in f['tiles']:
            active=[copy.deepcopy(s) for s in t['slots'] if not s['padding']]
            groups=[active[:12],active[12:]] if split and (f['filter_id'],t['tile_id'])==(3,4) else [active]
            for micro,slots in enumerate(groups):
                wrappers.append(dict(filter_id=f['filter_id'],parent_tile_id=t['tile_id'],microtile_id=micro,
                    origin=dict(session='synthetic',path='synthetic',reused=False),
                    result=dict(filter_id=f['filter_id'],tile_id=2*t['tile_id']+micro,technical_completed=True,effective_route=slots,
                        partials=[dict(case=c,repeat=r,t=step) for c in range(10) for r in range(5) for step in range(12)])))
    return wrappers,routing


class ReserveGateTests(unittest.TestCase):
    def setUp(self):
        self.routing,self.fixture=inputs();self.pool=reserve_pools(self.routing['filters'][3]['tiles'][4])

    def test_budget_derived_from_real_fixture(self):
        b=reserve_budget(self.routing,self.fixture)
        self.assertEqual((b['maximum_impacts'],b['maximum_programming_pulses'],b['maximum_programming_decisions']),(24323,3232,3232))
        self.assertEqual((b['reset_candidate_count'],b['new_trained_coefficients'],b['imported_trained_coefficients']),(33,13,287))

    def test_full_pool_adds_only_reserved_address(self):
        self.assertEqual(len(self.pool['positive_cell']),16);self.assertEqual(len(self.pool['negative_cell']),17)
        self.assertEqual(self.pool['negative_cell'][-1],[7,15]);self.assertEqual(sum(c==[7,15] for c in self.pool['negative_cell']),1)

    def test_exact_gate_boundary_and_reserved_last_tap(self):
        eligible=dict(positive_cell=self.pool['positive_cell'][:13],negative_cell=self.pool['negative_cell'][:12]+[[7,15]])
        r=reserve_route(self.pool,eligible)
        self.assertEqual(len(r['positive_cell']),13);self.assertEqual(r['negative_cell'][-1],[7,15])
        self.assertEqual(list(range(52,65))[-1],64)

    def test_reserve_not_quiet_cannot_fall_back_to_extra_old_negative(self):
        eligible=copy.deepcopy(self.pool);eligible['negative_cell'].remove([7,15])
        self.assertIsNone(reserve_route(self.pool,eligible))

    def test_insufficient_original_negative_cannot_count_reserve_twice(self):
        eligible=dict(positive_cell=self.pool['positive_cell'],negative_cell=self.pool['negative_cell'][:11]+[[7,15]])
        self.assertIsNone(reserve_route(self.pool,eligible))

    def test_insufficient_positive_rejects_before_programming(self):
        eligible=copy.deepcopy(self.pool);eligible['positive_cell']=eligible['positive_cell'][:12]
        self.assertIsNone(reserve_route(self.pool,eligible))

    def test_fresh_quiet_priority_and_fallback(self):
        r=reserve_route(self.pool,copy.deepcopy(self.pool));self.assertEqual(r['positive_cell'][:2],[[6,2],[6,11]])
        eligible=copy.deepcopy(self.pool);eligible['positive_cell'].remove([6,2]);r=reserve_route(self.pool,eligible)
        self.assertEqual(r['positive_cell'][0],[6,11]);self.assertNotIn([6,2],r['positive_cell'])
        self.assertEqual(len({tuple(x) for x in r['positive_cell']}),13)

    def test_duplicate_or_foreign_eligible_cells_rejected(self):
        for extra in ([7,15],[7,16]):
            eligible=copy.deepcopy(self.pool);eligible['negative_cell'].append(extra)
            with self.assertRaises(ValueError):reserve_route(self.pool,eligible)

    def test_reordered_candidates_cannot_change_assignment(self):
        eligible=copy.deepcopy(self.pool);eligible['positive_cell'].reverse()
        with self.assertRaisesRegex(ValueError,'order changed'):reserve_route(self.pool,eligible)


class ReserveReplacementTests(unittest.TestCase):
    def test_both_hardware_12_plus_1_and_sim13_removed_exactly(self):
        for split in (False,True):
            original,routing=synthetic_parent(split);snapshot=copy.deepcopy(original)
            chunks=selected_imports(original,'synthetic-parent','synthetic-wrapper-hash',routing)
            report=verify_wrapper_coverage(chunks,routing,excluded=True)
            self.assertEqual(report['coefficients'],287);self.assertEqual(report['measured_microtiles'],27)
            self.assertFalse(any((c['filter_id'],c['parent_tile_id'])==(3,4) for c in chunks))
            self.assertEqual(original,snapshot)
            self.assertEqual(chunks[0]['origin']['ancestry_original_origin'],original[0]['origin'])
            for c in chunks:
                self.assertEqual(c['result'],original[c['origin']['wrapper_index']]['result'])

    def test_retained_old_target_is_not_an_imported287(self):
        original,routing=synthetic_parent(True)
        with self.assertRaisesRegex(ValueError,'coverage'):verify_wrapper_coverage(original,routing,excluded=True)

    def test_missing_or_duplicate_tap_rejected(self):
        original,routing=synthetic_parent(True);original[-1]['result']['effective_route'].pop()
        with self.assertRaisesRegex(ValueError,'coverage'):selected_imports(original,'p','h',routing)

    def test_imported_weight_mutation_rejected(self):
        original,routing=synthetic_parent(False);original[0]['result']['effective_route'][0]['trained_weight']+=.1
        with self.assertRaisesRegex(ValueError,'tap/weight'):selected_imports(original,'p','h',routing)

    def test_missing_imported_timestep_not_filled(self):
        original,routing=synthetic_parent(False);original[0]['result']['partials'].pop()
        with self.assertRaisesRegex(ValueError,'partial ledger'):selected_imports(original,'p','h',routing)


if __name__=='__main__':unittest.main()
