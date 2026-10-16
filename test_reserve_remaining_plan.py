"""Pure offline selection tests; no backend dependencies."""
import copy
import unittest
from reserve_remaining_plan import BLOCKS,PLAN,candidate_pools,select_route,require_parent_bindings


class RemainingPlanTests(unittest.TestCase):
    def test_counts_scope_and_all_preferred_slots(self):
        self.assertEqual(sum(b['maximum_impacts'] for b in BLOCKS),40378)
        self.assertEqual(sum(b['maximum_writes'] for b in BLOCKS),5260)
        self.assertEqual(sum(len(b['taps']) for b in BLOCKS),20)
        self.assertEqual([sum(map(len,candidate_pools(b).values())) for b in BLOCKS],[29,27])
        for block in BLOCKS:
            pools=candidate_pools(block);result=select_route(block,pools)
            self.assertTrue(result['capacity_pass'])
            for side in ('positive','negative'):
                for tap,cell in block['preferred_'+side].items():
                    self.assertEqual(result['route'][side+'_cell'][block['taps'].index(tap)],cell)
            self.assertFalse(any(d['actual_cifar_assigned'] for d in result['preferred_disposition']))

    def test_unquiet_preferred_falls_back_without_lowering_threshold(self):
        for block in BLOCKS:
            eligible=candidate_pools(block)
            for side in ('positive','negative'):
                for cell in block['preferred_'+side].values():eligible[side+'_cell'].remove(cell)
            result=select_route(block,eligible)
            self.assertTrue(result['capacity_pass'])
            self.assertTrue(all(d['status']=='screened_not_quiet' for d in result['preferred_disposition']))
            self.assertTrue(all(len(cells)==len(block['taps']) for cells in result['route'].values()))

    def test_earlier_fallback_does_not_steal_later_preferred(self):
        block=BLOCKS[0];eligible=candidate_pools(block);eligible['negative_cell'].remove([5,6])
        result=select_route(block,eligible)
        self.assertNotEqual(result['route']['negative_cell'][0],[5,15])
        self.assertEqual(result['route']['negative_cell'][1],[5,15])
        self.assertEqual(len(set(map(tuple,result['route']['negative_cell']))),11)

    def test_capacity_abort_is_not_cell_failure_or_SET_permission(self):
        block=BLOCKS[1];eligible=candidate_pools(block)
        eligible['positive_cell']=eligible['positive_cell'][:8]
        result=select_route(block,eligible)
        self.assertFalse(result['may_issue_SET']);self.assertEqual(result['route']['negative_cell'],[])
        quiet=[d for d in result['preferred_disposition'] if d['quiet']]
        self.assertTrue(quiet);self.assertTrue(all(d['status']=='quiet_but_route_capacity_abort' for d in quiet))

    def test_no_unbound_parent_or_automatic_fallback(self):
        with self.assertRaises(ValueError):require_parent_bindings(PLAN)
        plan=copy.deepcopy(PLAN);plan.update(draft=False,native_execution_enabled=True)
        with self.assertRaises(ValueError):require_parent_bindings(plan)
        self.assertFalse(plan['automatic_parent_fallback']);self.assertFalse(plan['automatic_retry'])

    def test_unknown_and_duplicate_evidence_rejected(self):
        block=BLOCKS[0];eligible=candidate_pools(block);eligible['negative_cell'].append([5,16])
        with self.assertRaises(ValueError):select_route(block,eligible)
        eligible=candidate_pools(block);eligible['positive_cell'].append(eligible['positive_cell'][0])
        with self.assertRaises(ValueError):select_route(block,eligible)


if __name__=='__main__':unittest.main()
