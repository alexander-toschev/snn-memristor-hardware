"""Synthetic-only coverage/capacity classifications; no acquisition imports."""
import copy
import unittest
from analyze_cifar_adaptive_capacity import analyze,original_pools,category


def fixture(split=False):
    routing={'filters':[]};wrappers=[]
    for f in range(4):
        parents=[]
        for pid,start in enumerate(range(0,75,14)):
            slots=[dict(tap_index=t,padding=False,positive_cell=[2*f,t-start],negative_cell=[2*f+1,t-start]) for t in range(start,min(start+14,75))]
            parent=dict(tile_id=pid,slots=slots);parents.append(parent);pending=slots[:]
            passes=2 if split and f==0 and pid==0 else 1
            for micro in range(passes):
                pools=original_pools(parent);candidates=[]
                for side,cells in pools.items():
                    for i,cell in enumerate(cells):
                        quiet=not(split and f==0 and pid==0 and micro==0 and side=='positive_cell' and i>=13)
                        candidates.append(dict(side=side,cell=cell,reset_crossed=quiet,eligible=quiet,
                            mvm_confirmation=[{'voltage_v':0.}]*5,mode10_quiet_limit_v=.00062,reset_stop='synthetic'))
                k=min(len(pending),sum(c['eligible'] for c in candidates if c['side']=='positive_cell'),sum(c['eligible'] for c in candidates if c['side']=='negative_cell'))
                active=copy.deepcopy(pending[:k])
                for side in pools:
                    cells=[c['cell'] for c in candidates if c['side']==side and c['eligible']]
                    for i,s in enumerate(active):s[side]=cells[i]
                wrappers.append(dict(filter_id=f,parent_tile_id=pid,microtile_id=micro,
                    origin=dict(reused=False,new_measurement=True,session='synthetic',path='synthetic',sha256='synthetic'),
                    result=dict(technical_completed=True,pending_taps_before=[s['tap_index'] for s in pending],effective_route=active,reset_candidates=candidates,pairs=[])))
                pending=pending[k:]
        routing['filters'].append(dict(filter_id=f,tiles=parents))
    return wrappers,routing


class CapacityTests(unittest.TestCase):
    def test_union_is_not_sum_of_repeated_probes(self):
        w,r=fixture();a=analyze(w,r);self.assertEqual(len(a['addresses']),256)
        row=a['rows'][0];self.assertGreater(row['probe_events'],row['reset_probed'])
        self.assertEqual(a['split_parent_chunks'],0)

    def test_split_13_plus_1_keeps_full_original_pool(self):
        w,r=fixture(True);a=analyze(w,r)
        self.assertEqual(a['split_parent_chunks'],1)
        self.assertEqual([x['k_used'] for x in a['microtiles'][:2]],[13,1])
        self.assertEqual(a['microtiles'][1]['quiet_positive'],17)

    def test_improper_assignment_more_than_quiet_capacity_rejected(self):
        w,r=fixture(True);w[0]['result']['effective_route'].append(copy.deepcopy(r['filters'][0]['tiles'][0]['slots'][-1]))
        with self.assertRaisesRegex(ValueError,'capacity assignment'):analyze(w,r)

    def test_quiet_flag_cannot_override_mean_limit(self):
        w,r=fixture();w[0]['result']['reset_candidates'][0]['mvm_confirmation']=[{'voltage_v':.001}]*5
        with self.assertRaisesRegex(ValueError,'numeric criterion'):analyze(w,r)

    def test_category_does_not_hide_failure_history_in_flags(self):
        row=dict(SET_attempts=1,assignments=1,quiet_confirmations=1,reset_probes=2,quiet_failures=1)
        self.assertEqual(category(row),'SET_attempted');self.assertEqual(row['quiet_failures'],1)

    def test_missing_prefix_cannot_be_a_300_coefficient_map(self):
        w,r=fixture();w.pop()
        with self.assertRaisesRegex(ValueError,'exact 300'):analyze(w,r)


if __name__=='__main__':unittest.main()
