import copy
import hashlib
import json
import unittest
from pathlib import Path
from repair_aggregate import aggregate, V2, EXPECTED, sha

class RepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records=[json.loads(s) for s in (V2/'patches.jsonl').read_text().splitlines()]

    def test_frozen_hashes_and_counts(self):
        for name,digest in EXPECTED.items():self.assertEqual(sha(V2/name),digest)
        self.assertEqual(len(self.records),7011)
        self.assertEqual(len([r for r in self.records if r['noise_flag']==1 and r['split'] in ('train','val')]),6912)

    def test_saved_offline_analyzer_agrees(self):
        patches,summary=aggregate(self.records)
        saved=json.loads((V2/'full75_summary_offline.json').read_text())
        offline=json.loads((V2/'full75_patches_offline.json').read_text())
        self.assertEqual(len(patches),2304)
        for repaired,reference in zip(patches,offline):
            self.assertEqual((repaired['replicate'],repaired['split'],repaired['position'],repaired['top'],repaired['left'],repaired['time'],repaired['read']),
                             (reference['replicate'],reference['split'],reference['split_position'],reference['top'],reference['left'],reference['t'],reference['read']))
            for field in ('ideal','quantized_ideal','programmed_readback','quantized_programmed','sim_mvm'):
                self.assertAlmostEqual(repaired[field],reference[field],places=12)
        self.assertTrue(summary['all_full75_gates_passed'])
        for rep,actual in summary['replicates'].items():
            expected=saved['per_replicate'][rep]
            self.assertEqual(actual['validation_gate_passed'],expected['val_raw_gate_passed'])
            self.assertEqual(actual['val_count'],expected['sim_mvm']['n'])
            self.assertAlmostEqual(actual['ideal_sd'],expected['sim_mvm']['ideal_sd'],places=12)
            self.assertAlmostEqual(actual['train_affine_slope'],expected['train_only_surrogate']['slope'],places=12)
            self.assertAlmostEqual(actual['train_affine_intercept'],expected['train_only_surrogate']['intercept'],places=12)
            self.assertAlmostEqual(actual['surrogate_val_rmse'],expected['train_only_surrogate']['val']['rmse'],places=12)
            for field in ('sim_mvm','quantized_ideal','quantized_programmed','programmed_readback'):
                self.assertAlmostEqual(actual[field]['rmse'],expected[field]['rmse'],places=12)
                self.assertAlmostEqual(actual[field]['corr'],expected[field]['pearson_r'],places=12)

    def test_missing_tile_rejected(self):
        records=self.records[:1]+self.records[2:]
        # Remove a noisy train row, rather than an ungrouped deterministic control.
        records=[r for i,r in enumerate(self.records) if i!=next(i for i,x in enumerate(self.records) if x['noise_flag']==1 and x['split']=='train')]
        with self.assertRaisesRegex(ValueError,'incomplete three-tile'):aggregate(records)

    def test_duplicate_tile_rejected(self):
        noisy=next(x for x in self.records if x['noise_flag']==1 and x['split']=='train')
        with self.assertRaisesRegex(ValueError,'duplicate patch tile'):aggregate(self.records+[copy.deepcopy(noisy)])

if __name__=='__main__': unittest.main()
