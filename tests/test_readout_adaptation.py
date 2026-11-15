"""Offline synthetic tests: leakage boundaries, epoch ties and evidence integrity."""
from __future__ import annotations
from pathlib import Path
import tempfile
import unittest
import numpy as np
from numpy.typing import NDArray
import torch
from experiment_common.runtime import use_local_dependencies
use_local_dependencies(prefer_environment=True)
from experiment_common.durable import sha
from experiment_common.readout import evaluate
from experiment_common.readout_adaptation import (
    adapt,earliest_best,fit_scaler,stratified_split,verified_file,verify_checkpoint,
)


class AdaptationTests(unittest.TestCase):
    def test_exact_reproducible_stratification(self) -> None:
        truth = np.asarray([i for i in range(10) for _ in range(991+2*i)],dtype=np.int64)
        self.assertEqual(len(truth),10000)
        fit,valid = stratified_split(truth,2000)
        fit2,valid2 = stratified_split(truth,2000)
        self.assertEqual((len(fit),len(valid)),(8000,2000))
        self.assertEqual(len(np.intersect1d(fit,valid)),0)
        self.assertEqual(len(np.unique(np.concatenate([fit,valid]))),10000)
        np.testing.assert_array_equal(fit,fit2)
        np.testing.assert_array_equal(valid,valid2)
        for label in range(10):
            self.assertLessEqual(abs(int(np.count_nonzero(truth[valid]==label))-.2*np.count_nonzero(truth==label)),1)

    def test_scaler_cannot_see_validation_outliers(self) -> None:
        counts = np.asarray([[0,1],[1,2],[2,3],[3,4]],dtype=np.uint8)
        fit = np.asarray([0,1],dtype=np.int64)
        original = fit_scaler(counts,fit)
        counts[2:] = 255
        changed = fit_scaler(counts,fit)
        self.assertTrue(torch.equal(original.mean,changed.mean))
        self.assertTrue(torch.equal(original.sd,changed.sd))

    def test_earliest_tie_and_not_last_epoch(self) -> None:
        self.assertEqual(earliest_best([5,9,9,8]),2)
        self.assertEqual(earliest_best([3,3,3]),1)

    def test_complete_synthetic_fit_refit_checkpoint_before_labels(self) -> None:
        torch.set_num_threads(4)
        rng = np.random.default_rng(42)
        counts = rng.integers(0,5,size=(200,12),dtype=np.uint8)
        truth = (np.arange(200)%10).astype(np.int64)
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            model,selected = adapt(counts,truth,out,lambda _: None,validation_count=40,max_epochs=3)
            self.assertIn(selected,(1,2,3))
            verify_checkpoint(out)
            accesses: list[str] = []
            def reader(split: str,count: int) -> NDArray[np.int64]:
                verify_checkpoint(out)
                self.assertTrue((out/'selection.json').is_file())
                accesses.append(split)
                return truth[:count]
            metrics = evaluate(model,counts[:20],out,label_reader=reader)
            self.assertEqual(accesses,['test'])
            self.assertEqual(metrics['test_images'],20)
            with (out/'readout.pt').open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaises(ValueError):
                verify_checkpoint(out)

    def test_evidence_hash_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            path = out/'counts.bin'
            path.write_bytes(b'original')
            manifest: dict[str,object] = {'result_sha256':{'counts.bin':sha(path)}}
            self.assertEqual(verified_file(out,'counts.bin',manifest),path)
            path.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                verified_file(out,'counts.bin',manifest)
            with self.assertRaises(ValueError):
                verified_file(out,'../counts.bin',manifest)


if __name__ == '__main__':
    unittest.main()
