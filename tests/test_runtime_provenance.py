"""Offline provenance versions only: no Connector, model, network or Git writes."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import runtime_provenance as p
from tools import audit_cifar_interrupted_prefix as audit
from tools import classify_cifar_continuation as composer


def candidate_versions():
    value=json.loads((p.ROOT/p.REGISTRATION).read_text())
    value['registration']='approved_after_review' # synthetic test registration only
    return value


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.registration=patch.object(p,'versions',return_value=candidate_versions())
        self.registration.start();self.addCleanup(self.registration.stop)

    def test_missing_private_files_need_no_receipt_or_file_reads(self):
        with patch.object(Path,'read_bytes',side_effect=AssertionError('private file read')):
            result=p.private_recorded(dict(p.PRIVATE))
        self.assertEqual(result['status'],p.STATUS)
        self.assertFalse(result['private_bytes_git_verified'])
        self.assertFalse(result['fresh_remote_check_performed'])
        wrong=dict(p.PRIVATE);wrong[next(iter(wrong))]='0'*64
        with self.assertRaises(ValueError):p.private_recorded(wrong)

    def test_historical_git_source_exact_known_version(self):
        name='experiments/cifar_resumable_20261019/contract.py'
        pin=candidate_versions()['historical_sources'][name]
        self.assertEqual(p.verify_source(name,pin),'historical_git_object_bytes')
        with self.assertRaises(ValueError):p.verify_source(name,'0'*64)
        with self.assertRaises(ValueError):p.verify_source('unknown.py',pin)
        with self.assertRaises(ValueError):p.verify_source('../outside',pin)

    def test_git_errors_or_wrong_historical_bytes_fail(self):
        name='experiments/cifar_resumable_20261019/contract.py';pin=candidate_versions()['historical_sources'][name]
        class Result:returncode=0;stdout=b'changed'
        with patch.object(p.subprocess,'run',return_value=Result()):
            with self.assertRaises(ValueError):p.verify_source(name,pin)

    def test_hardware_ini_public_sources_still_strict(self):
        ini,pins=p.hardware_runtime()
        self.assertIn(b'rp5_fpga_python',ini)
        self.assertTrue(set(p.PRIVATE)<=set(pins))
        with patch.object(p,'verify_source',side_effect=ValueError('changed public source')):
            with self.assertRaises(ValueError):p.hardware_runtime()

    def test_current_batch_historical_plan_keeps_exact_range(self):
        from experiments.cifar_resumable_20261019 import contract
        path=p.ROOT/'reports/2026-10-20/cifar-hardware-batch-13-18-review/HW_BATCH_19_25_PLAN.json'
        before=p.sha(path);value=json.loads(path.read_text())
        self.assertEqual(contract.validate_plan(value,'hardware'),(19,25))
        self.assertEqual(p.sha(path),before)
        bad=copy.deepcopy(value);bad['accepted_stage_ids']=list(range(18))
        with self.assertRaises(ValueError):contract.validate_plan(bad,'hardware')

    def test_exact_old_and_new_audit_inventories(self):
        v=candidate_versions();names=('tools/audit_cifar_interrupted_prefix.py','tools/cifar_prefix_replay.py')
        old=dict(provenance={},audit_source_sha256={n:v['historical_sources'][n] for n in names})
        p.audit_sources(old)
        new=copy.deepcopy(old);new['provenance']['provenance_generation']='v2';new['audit_source_sha256']={n:v['current'][n] for n in (*names,'tools/runtime_provenance.py')}
        new['audit_source_sha256'][p.REGISTRATION]=p.sha(p.ROOT/p.REGISTRATION)
        p.audit_sources(new)
        bad=copy.deepcopy(new);del bad['provenance']['provenance_generation']
        with self.assertRaisesRegex(ValueError,'requires explicit v2'):p.audit_sources(bad)
        bad=copy.deepcopy(old);bad['provenance']['provenance_generation']='v2'
        with self.assertRaisesRegex(ValueError,'Historical producer'):p.audit_sources(bad)
        for name in new['audit_source_sha256']:
            bad=copy.deepcopy(new);del bad['audit_source_sha256'][name]
            with self.subTest(name=name),self.assertRaises(ValueError):p.audit_sources(bad)
        bad=copy.deepcopy(old);bad['audit_source_sha256']['tools/runtime_provenance.py']=v['current']['tools/runtime_provenance.py']
        with self.assertRaises(ValueError):p.audit_sources(bad)

    def test_real_completed_stage_source_check_without_receipt(self):
        index=json.loads((p.ROOT/'reports/2026-10-20/cifar-generic-batch-review/REGRESSION_BATCH_REVIEW.json').read_text())
        # Select a saved, completed stage through the fixed reviewed report, never live discovery.
        def find(value):
            if isinstance(value,dict):
                for k,v in value.items():
                    if k=='acquisition_session' and '/cifar_resumable_20261019/' in str(v):return v
                    got=find(v)
                    if got:return got
            elif isinstance(value,list):
                for row in value:
                    got=find(row)
                    if got:return got
        name=find(index)
        self.assertIsNotNone(name)
        session=p.ROOT/name
        start,proof=audit.verify_sources(session,stage_session=True)
        self.assertEqual(proof['status'],p.STATUS)
        self.assertEqual(proof['provenance_generation'],'v2')
        self.assertEqual(proof['public_source_verification']['experiments/cifar_resumable_20261019/contract.py'],'historical_git_object_bytes')
        self.assertEqual(composer.verify_stage_grant(session,start)['sha256'],start['execution_grant_sha256'])



if __name__=='__main__':unittest.main()
