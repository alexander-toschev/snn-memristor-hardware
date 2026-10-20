"""Focused file-only portable launcher/replay contract test; no model or board."""
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'.deps')]
from experiments.mnist_conductance_20261019 import portable,core
from tools import audit_mnist_conductance as original_replay,audit_mnist_conductance_portable as portable_replay

class PortableTest(unittest.TestCase):
    def test_unchanged_protocol_fixture_and_numeric_replay(self):
        config,sources,fixture,evidence=portable.preflight()
        self.assertEqual(portable.protocol_sha(),'a9cc77abce4cdde7e62a4bb8924d3eaf8b795509207dff3f262b4bbb56597e95')
        self.assertEqual(core.SPEC,original_replay.expected_protocol())
        self.assertIs(portable_replay.numeric.Replay,original_replay.Replay)
        self.assertIs(portable_replay.numeric.durable_records,original_replay.durable_records)
        self.assertIs(portable_replay.numeric.replay_diagnostics,original_replay.replay_diagnostics)
        self.assertEqual(config.get('board','board_type'),'rp5_fpga_python')
        self.assertFalse(any('/MemriCORE/' in p for p in sources))
        self.assertEqual(len(fixture['images']),176);self.assertFalse(fixture['labels_included'])
        original_replay.verify_fixture(fixture)
        self.assertEqual(core.SPEC['maximum_impacts'],172288);self.assertEqual(core.SPEC['maximum_pulses'],2560)
        self.assertEqual(core.SPEC['learning_wl'],[0,1,2,3]);self.assertEqual(core.SPEC['control_wl'],[4,5,6,7])

if __name__=='__main__':unittest.main()
