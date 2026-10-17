"""Offline boundaries and historical-source policy; no experiment imports."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from audit_cell_range_repeat import Checks,verify_public_source
from interval_audit_core import protocol_profile,stop_band,expected_band,inside


class RefinedAuditTests(unittest.TestCase):
    def test_protocol_counts_and_budget(self):
        for protocol,count,maximum in [('three-cell-native-R-interval-v1',27,6120),('three-cell-native-R-interval-v2-refined',24,5991)]:
            p=protocol_profile(protocol)
            self.assertEqual(3*sum(map(len,p['targets'].values())),count)
            self.assertEqual(3*sum(551+43*len(qs) for qs in p['targets'].values()),maximum)
            self.assertEqual(p['maximum_impacts'],maximum)

    def test_inner_hit_does_not_accept_outer_only(self):
        p=protocol_profile('three-cell-native-R-interval-v2-refined')
        inner,tol=stop_band(.04,p);outer=expected_band(.04)
        self.assertEqual(tol,.0075)
        for r in (inner['r_min'],inner['r_max']):self.assertTrue(inside(r,inner))
        r=330/(.04*1.009)
        self.assertTrue(inside(r,outer));self.assertFalse(inside(r,inner))
        self.assertLess(r,inner['r_min'])  # explicit overshoot, never fabricated hit
        r=330/(.04*.991)
        self.assertTrue(inside(r,outer));self.assertFalse(inside(r,inner))
        self.assertGreater(r,inner['r_max'])  # continue SET, not overshoot

    def test_control_bands_unchanged_and_unknown_protocol_rejected(self):
        p=protocol_profile('three-cell-native-R-interval-v2-refined')
        for q in (.045,.05):self.assertEqual(stop_band(q,p),(expected_band(q),.01))
        with self.assertRaises(ValueError):protocol_profile('unknown')

    def check_source(self,name,local,blob,digest):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            root=Path(tmp);p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(local)
            ck=Checks();difference=verify_public_source(root,name,digest,blob,ck)
            return ck,difference

    def test_historical_runtime_git_is_not_current_runtime(self):
        blob=b'old code';digest=hashlib.sha256(blob).hexdigest()
        ck,d=self.check_source('experiment.py',b'new code',blob,digest)
        self.assertFalse(ck.errors);self.assertEqual(d['recorded_sha256'],digest)
        self.assertNotEqual(d['local_sha256'],digest);self.assertFalse(d['imported_by_audit'])

    def test_vendor_local_mismatch_rejected_even_with_valid_git(self):
        blob=b'old code';digest=hashlib.sha256(blob).hexdigest()
        ck,d=self.check_source('vendor/MemriBoard/manager/__init__.py',b'changed',blob,digest)
        self.assertTrue(ck.errors);self.assertTrue(d['imported_by_audit'])

    def test_git_mismatch_rejected_even_with_matching_local(self):
        blob=b'old code';digest=hashlib.sha256(blob).hexdigest()
        ck,_=self.check_source('configs/cell_range_repeat.json',blob,b'wrong Git',digest)
        self.assertTrue(ck.errors)


if __name__=='__main__':unittest.main()
