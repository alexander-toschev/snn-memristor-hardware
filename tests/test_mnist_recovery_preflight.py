"""Offline artifact-format validation, no model or hardware dependencies."""
import gzip
from pathlib import Path
import struct
import tempfile
import unittest
from experiments.mnist_recovery_20261019 import experiment as recovery


class PreflightTests(unittest.TestCase):
    def setUp(self):
        (recovery.ROOT/'.tmp').mkdir(exist_ok=True)
        temporary=tempfile.TemporaryDirectory(dir=recovery.ROOT/'.tmp');self.addCleanup(temporary.cleanup)
        self.output=Path(temporary.name)
    def test_idx_checks_size_and_label_range(self):
        path=self.output/'labels.gz'
        with gzip.open(path,'wb') as stream:stream.write(struct.pack('>II',2049,3)+bytes([0,3,9]))
        self.assertEqual(recovery.check_idx(path,3,False)['samples'],3)
        with gzip.open(path,'wb') as stream:stream.write(struct.pack('>II',2049,3)+bytes([0,3,10]))
        with self.assertRaisesRegex(ValueError,'outside'):recovery.check_idx(path,3,False)
        with gzip.open(path,'wb') as stream:stream.write(struct.pack('>II',2049,3)+bytes([0,3]))
        with self.assertRaisesRegex(ValueError,'length'):recovery.check_idx(path,3,False)
    def test_paths_do_not_escape_repository(self):
        for bad in ('../outside.json','D:/outside.json','/outside.json',''):
            with self.assertRaises(ValueError):recovery.local_path(bad)
    def test_archived_hook_warning_is_static_and_detected(self):
        report=recovery.source_semantics(recovery.local_path(recovery.ARCHIVE+'/reference_source/csnn_mnist_net.py'))
        self.assertTrue(report['historical_hook_warning'])
        self.assertEqual(report['connection_load_lines'],[253,258])


if __name__=='__main__':unittest.main()
