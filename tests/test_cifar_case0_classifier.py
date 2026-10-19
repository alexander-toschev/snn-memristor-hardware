"""Offline bridge tests. No native Connector/simulator or training."""
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.cifar_full_case0_20261019 import classifier as c
import numpy as np
import torch


class ClassifierBridgeTests(unittest.TestCase):
    def test_relative_paths_only(self):
        for name in ('../escape', 'C:/escape', '/escape', '\\escape'):
            with self.assertRaises(ValueError):
                c.local(ROOT, name)

    def test_exact_three_tile_sum(self):
        parts = np.zeros(c.SHAPE, dtype='<f4')
        parts[..., 0], parts[..., 1], parts[..., 2] = 1., 2., -4.
        payload = parts.tobytes()
        conv = c.decode_payload(payload, hashlib.sha256(payload).hexdigest())
        self.assertEqual(tuple(conv.shape), c.SHAPE[:-1])
        self.assertTrue(torch.equal(conv, -torch.ones(c.SHAPE[:-1])))

    def test_missing_wronghash_nonfinite_rejected(self):
        good = np.zeros(c.SHAPE, dtype='<f4')
        payload = good.tobytes()
        for data, digest in ((payload[:-4], hashlib.sha256(payload[:-4]).hexdigest()),
                             (payload, '0' * 64)):
            with self.assertRaises(ValueError):
                c.decode_payload(data, digest)
        good[0, 0, 0, 0, 0] = np.nan
        payload = good.tobytes()
        with self.assertRaises(ValueError):
            c.decode_payload(payload, hashlib.sha256(payload).hexdigest())

    def test_smoke_or_failed_session_rejected_before_import(self):
        import json
        with tempfile.TemporaryDirectory(dir=ROOT / '.tmp') as directory:
            path = Path(directory)
            (path / 'start_manifest.json').write_text('{}')
            for status, scope in (('completed', 'smoke'), ('failed', 'full')):
                (path / 'manifest.json').write_text(json.dumps(dict(status=status, scope=scope)))
                with self.assertRaisesRegex(ValueError, 'completed full'):
                    c.verify_session(path)

    def test_checkpoint_cpu_forward_and_weight_binding(self):
        from experiments.cifar_full_case0_20261019 import collector
        torch.set_num_threads(2)
        ref = c.digital()
        state, model, case, image = ref.assets(ROOT, 0)
        flat, weights = collector.load_assets()
        self.assertEqual(flat, image.flatten().tolist())
        self.assertTrue(torch.equal(torch.from_numpy(weights.copy()), state['conv_layers.0.weight'].reshape(64, 75)))
        _, result, validation = c.cpu_control(state, model, image)
        self.assertEqual(validation['direct_forward_max_abs'], 0.)
        self.assertEqual(validation['tiled_forward_max_abs']['spikes'], 0.)
        self.assertEqual(tuple(result[2].shape), (1, 10))

    def test_complete_auditor_payloads_consumed_without_cpu_replacement(self):
        # Explicit software-only boundary fixture; this does not claim that a
        # physical trace was collected or independently audited in this test.
        from experiments.cifar_full_case0_20261019 import audit_native
        torch.set_num_threads(2)
        ref = c.digital()
        state, model, case, image = ref.assets(ROOT, 0)
        parts = ref.cpu_tiles(ref.encode(image), state).numpy().astype('<f4')
        payloads = {'raw': parts.tobytes(), 'calibrated': np.zeros(c.SHAPE, dtype='<f4').tobytes()}
        report = dict(complete_full_tensor=True, scope='full', source='simulator',
                      observed_coefficients=4800, covered_tensor_elements=12*64*28*28*3,
                      native_impacts=42, programming_pulses=7,
                      payload_sha256={k: hashlib.sha256(v).hexdigest() for k, v in payloads.items()})
        manifest = dict(source='simulator', impacts=42, pulses=7, protocol_sha256='test')
        with tempfile.TemporaryDirectory(dir=ROOT / '.tmp') as directory:
            session = Path(directory)
            (session / 'manifest.json').write_text('{}')
            with mock.patch.object(c, 'verify_session', return_value=(manifest, 'test', iter(()))), \
                    mock.patch.object(audit_native, 'audit', return_value=(payloads, report)):
                result = c.classify_session(session, state, model, case, image)
        self.assertEqual(result['conv1_errors']['raw']['relative_l2'], 0.)
        self.assertAlmostEqual(result['conv1_errors']['calibrated']['relative_l2'], 1.)
        self.assertEqual(result['predictions']['raw'], result['predictions']['cpu_reference'])


if __name__ == '__main__':
    unittest.main()
