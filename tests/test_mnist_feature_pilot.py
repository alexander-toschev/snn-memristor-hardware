"""No neural execution: input framing, source pins and default-closed pilot gate."""
import gzip
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from experiments.mnist_feature_pilot_20261019 import experiment as e


class PilotTests(unittest.TestCase):
    def test_train_prefix_and_crc_without_test_data(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'train.gz'
            payload=bytes(range(10));p.write_bytes(gzip.compress(struct.pack('>II',2049,10)+payload))
            self.assertEqual(e.idx_prefix(p,False,10,4),payload[:4])
            p.write_bytes(gzip.compress(struct.pack('>II',2049,10)+payload[:-1]))
            with self.assertRaises(ValueError):e.idx_prefix(p,False,10,4)

    def test_shape_and_label_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'input.gz'
            p.write_bytes(gzip.compress(struct.pack('>IIII',2051,2,27,28)+bytes(2*27*28)))
            with self.assertRaises(ValueError):e.idx_prefix(p,True,2,1)
            p.write_bytes(gzip.compress(struct.pack('>II',2049,2)+bytes([255,1])))
            with self.assertRaises(ValueError):e.idx_prefix(p,False,2,1)

    def test_source_pins_and_broken_hook_reference_present(self):
        pins=e.verify_sources()
        self.assertEqual(pins[e.CHECKPOINT.relative_to(e.ROOT).as_posix()],e.CHECKPOINT_SHA)
        self.assertFalse(any('t10k-' in name for name in pins))

    def test_pilot_closed_before_dependencies_or_network(self):
        with patch.object(e,'APPROVED_SPEC_SHA256',frozenset()):
            with self.assertRaisesRegex(ValueError,'review required'):e.main(['--run-pilot'])
        with self.assertRaisesRegex(ValueError,'review required'):e.main(['--run-pilot','--device','cpu'])
        self.assertEqual(e.SPEC['train_ids'],list(range(128)))
        self.assertFalse(e.SPEC['readout_training']);self.assertFalse(e.SPEC['board_access'])

    def test_frozen_tensor_evidence_detects_mutation_and_learning(self):
        import torch
        layer=SimpleNamespace(learning=False,training=False,thresh=torch.tensor(.38))
        connection=SimpleNamespace(w=torch.tensor([.2,.4]),b=torch.zeros(1))
        net=SimpleNamespace(connections={('Input','Conv1'):connection},layers={'Conv1':layer},learning=False,training=False)
        initial=e.state_hashes(net)
        with torch.no_grad():connection.w.add_(.01)
        self.assertNotEqual(e.state_hashes(net),initial)
        layer.learning=True
        with self.assertRaisesRegex(ValueError,'learning enabled'):e.state_hashes(net)
        layer.learning=False;net.connections[('Conv1','Extra')]=connection
        with self.assertRaisesRegex(ValueError,'additional connection'):e.state_hashes(net)


if __name__=='__main__':unittest.main()
