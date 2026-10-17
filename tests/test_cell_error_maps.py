"""Compact CSV aggregation only; no hardware or large raw fixtures."""
import copy
import csv
from pathlib import Path
import unittest
from tools.build_cell_error_maps import INPUT, aggregate, address, load

ROOT = Path(__file__).resolve().parents[1]


class CellMapTests(unittest.TestCase):
    def test_actual_300_weights_256_addresses_and_confounded_roles(self):
        maps, _ = load(ROOT)
        for rows in maps:
            self.assertEqual(len(rows), 256)
            self.assertEqual(sum(r['active_assignments'] for r in rows), 300)
            self.assertEqual(sum(r['paired_role_events'] for r in rows), 300)
        r=maps[1][3*32+15]
        self.assertEqual((r['wl'],r['bl'],r['exhausted'],r['hit'],r['actual_SET_events']),(3,15,8,1,9))
        self.assertEqual(r['stop_map_category'],'exhausted_present')
        self.assertEqual(len([x for x in r['references'] if x['role']=='planned_SET_leg']),9)
        self.assertTrue(all(not r['references'] and r['excluded_by_protocol'] for r in maps[1] if r['bl']==8))
        self.assertEqual(sum(x['tap']==8 and x['role']=='planned_SET_leg' for r in maps[1] for x in r['references']),4)

    def test_duplicate_or_missing_physical_and_logical_records_rejected(self):
        folder=ROOT/INPUT
        with (folder/'SESSION2_CELLS.csv').open() as f:cells=list(csv.DictReader(f))
        with (folder/'SESSION2_WEIGHTS.csv').open() as f:weights=list(csv.DictReader(f))
        bad=copy.deepcopy(cells);bad[0]=bad[1]
        with self.assertRaises(ValueError):aggregate(bad,weights)
        bad=copy.deepcopy(weights);bad[0]=bad[1]
        with self.assertRaises(ValueError):aggregate(cells,bad)
        with self.assertRaises(ValueError):aggregate(cells,weights[:-1])

    def test_address_mapping_rejects_out_of_range_and_excluded_route(self):
        self.assertEqual(address('[3,15]'),(3,15))
        for s in ('[8,0]','[0,32]','[-1,0]','[3.0,15]'):
            with self.assertRaises(ValueError):address(s)
        folder=ROOT/INPUT
        with (folder/'SESSION2_CELLS.csv').open() as f:cells=list(csv.DictReader(f))
        with (folder/'SESSION2_WEIGHTS.csv').open() as f:weights=list(csv.DictReader(f))
        weights[0]['positive_cell']='[0,8]'
        with self.assertRaisesRegex(ValueError,'Excluded BL8'):aggregate(cells,weights,True)


if __name__=='__main__':unittest.main()
