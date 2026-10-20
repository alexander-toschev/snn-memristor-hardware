"""No physical I/O: launcher file preflight and saved native-data replay."""
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'.deps')]
from experiments.conductance_stdp_hardware_20261019 import portable
from tools import audit_conductance_stdp_portable as audit

class PortableTests(unittest.TestCase):
    def test_public_preflight_same_protocol_without_private_sources(self):
        config,sources,evidence=portable.preflight()
        self.assertEqual(config.get('board','board_type'),'rp5_fpga_python')
        self.assertFalse(any('/MemriCORE/' in name for name in sources))
        self.assertEqual(portable.protocol_sha(),'6d86180c0a73311753bfa420b66e8f6727d779106025ad23ac9d8cc6a3b414c5')
        self.assertEqual(portable.core.BL,(6,8,9,14));self.assertEqual(portable.core.WL,(5,7))

    def test_numeric_tail_exact_copy(self):
        old=(ROOT/'tools/audit_conductance_stdp_local_transfer.py').read_text()
        new=Path(audit.__file__).read_text()
        self.assertEqual(old.split('# BEGIN FROZEN NUMERIC TAIL\n')[1].split('# END FROZEN NUMERIC TAIL')[0],
                         new.split('# BEGIN UNCHANGED NUMERIC REPLAY\n')[1].split('# END UNCHANGED NUMERIC REPLAY')[0])

    def test_missing_private_not_attested_public_mismatch_rejected(self):
        proof=audit.recorded_sources({'source_sha256':{'vendor/MemriBoard/MemriCORE/absent.py':'0'*64}})
        self.assertEqual(proof['private_runtime_status'],'private_runtime_not_independently_reverified')
        with self.assertRaises((AssertionError,ValueError)):
            audit.recorded_sources({'source_sha256':{'read_all.py':'0'*64}})

    def test_existing_native_sim_numeric_replay(self):
        report=audit.audit(ROOT/'results/conductance_stdp_calibrated_20261019/d27da87e-9928-4303-971e-215afbc63158')
        self.assertEqual(report['status'],'pass');self.assertEqual(report['impacts'],6616)
        self.assertEqual(report['training_pulses'],32);self.assertEqual(report['selected_gain'],800)
        self.assertFalse(report['hardware_io_executed'])

if __name__=='__main__':unittest.main()
