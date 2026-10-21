"""Extraction equivalence; no native simulator, hardware or training run."""
from __future__ import annotations
import gzip
import json
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from experiment_common import mnist_inputs as common_inputs
from experiment_common.native_memriboard import Collector as CommonNative
from experiment_common.native_output import safe
from experiment_common.durable import Writer
from experiment_common.mnist_stdp import DigitalStep
from experiment_common.stdp_collector import Collector
from experiments.mnist_archive64_20261020 import inputs as historical_inputs
from experiments.mnist_baseline_20261019.reference_source.encoders import PoissonEncoder as ReferenceEncoder
from experiments.mnist_archive64_20261020.native import Collector as HistoricalNative
from experiments.mnist_online_stdp64_20261021 import logic, collector
from experiments.cifar_resumable_20261019 import durable
from tests.test_mnist_online_stdp64 import FakeConnector, FakeCV, FakeWriter


class SharedExtractionTests(unittest.TestCase):
    def test_official_inputs_poisson_and_masks_equal_archived_implementation(self):
        torch.set_num_threads(2)
        for split in ('train', 'test'):
            pixels = common_inputs.images(split, 4)
            self.assertTrue(np.array_equal(pixels, historical_inputs.images(split, 4)))
            self.assertTrue(np.array_equal(common_inputs.labels(split, 4), historical_inputs.labels(split, 4)))
            actual = common_inputs.encode(pixels)
            expected = ReferenceEncoder(T=100,rate_scale=.0015,base_seed=123,deterministic=True,out_format='TBNCHW')(
                torch.from_numpy(pixels.astype(np.float32)/255.))
            self.assertTrue(torch.equal(actual, expected))
            self.assertTrue(np.array_equal(common_inputs.patch_masks(actual), historical_inputs.patch_masks(expected)))
            self.assertTrue(torch.equal(common_inputs.encode(pixels[:1]), actual[:, :1]))

    def test_old_protocol_paths_are_real_aliases_not_duplicate_collectors(self):
        self.assertIs(logic.DigitalStep, DigitalStep)
        self.assertIs(collector.Collector, Collector)
        self.assertIs(durable.Writer, Writer)
        self.assertTrue(callable(durable.simulated_process_loss))

    def test_native_command_response_and_guards_equal_original(self):
        old_connector = FakeConnector()
        new_connector = FakeConnector()
        old_writer, new_writer = FakeWriter(), FakeWriter()
        old = HistoricalNative(old_connector, FakeCV(), old_writer, 1000, lambda _: None)
        new = CommonNative(new_connector, FakeCV(), new_writer, 1000, lambda _: None)
        self.assertEqual(old.read_g(2, 4, phase='test'), new.read_g(2, 4, phase='test'))
        self.assertEqual(asdict(old.query(3, (1<<25)-1, 0, 0)), asdict(new.query(3, (1<<25)-1, 0, 0)))
        self.assertEqual(old_connector.commands, new_connector.commands)
        self.assertEqual(len(new_connector.commands),29)
        self.assertEqual(new_connector.commands[-1],dict(mode_flag=10,wl=3,vol=[.05]*25+[0.]*7,id=0))
        self.assertEqual([row['bl'] for row in new_connector.commands[3:-1]],list(range(25)))
        def without_clock(rows):
            return [{k: v for k, v in row.items() if k != 'monotonic_s'} for row in rows]
        self.assertEqual(without_clock(old_writer.rows), without_clock(new_writer.rows))

    def test_json_safe_preserves_original_numpy_scalar_treatment(self):
        self.assertEqual(safe(np.float64(1.25)), repr(np.float64(1.25)))
        self.assertEqual(safe(1.25), 1.25)
        self.assertEqual(safe(float('inf')), 'inf')

    def test_durable_records_roundtrip_and_final_index(self):
        root = Path('.tmp')
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as folder:
            path = Path(folder)/'raw'
            writer = Writer(str(path), limit=128, minimum_free_bytes=0)
            records = [dict(event='request', id=1), dict(event='result', values=[1., 2, 3]), dict(event='complete')]
            for record in records:
                writer.write(record)
            writer.finish()
            index = json.loads((path/'index.json').read_text(encoding='utf-8'))
            reconstructed = []
            for part in index['parts']:
                with gzip.open(path/part['path'], 'rt', encoding='utf-8') as stream:
                    reconstructed.extend(json.loads(line) for line in stream)
            self.assertEqual(reconstructed, records)
            self.assertEqual(index['records'], 3)
            self.assertTrue(index['stream_finalized'])
            with self.assertRaisesRegex(ValueError, 'already finalized'):
                writer.write(dict(event='forbidden'))


if __name__ == '__main__':
    unittest.main()
