"""Offline source-receipt mutations and unchanged replay of saved native SIM; no board."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from tools import audit_conductance_stdp_local_transfer as a


class TransferReceiptTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='stdp-transfer-test-',dir=a.ROOT/'.tmp')
        self.session=Path(self.tmp.name);self.manifest=dict(source='hardware',finished_utc='2026-10-20T09:00:00+00:00',source_sha256={k:v[0] for k,v in a.PRIVATE.items()},launcher_sha256='cef6db7a176d2264c12be077852d56a0dc62bc7e696f351bc49ebfeb328ae5ed',launcher_spec_sha256='ef9d61315c7a248732c0f56a877e92776db92746a8d2e4a24f02a2b1979635e9',qualifying_sim_audit_sha256='19ebfdbb29e4196a7780521743684f309939917310ce2543d7be508353b5ae53')
        for name,data in [('manifest.json',self.manifest),('start_manifest.json',self.manifest),('protocol.json',{'synthetic':True})]:
            (self.session/name).write_text(json.dumps(data),encoding='utf-8')
        (self.session/'board.ini').write_bytes(b'synthetic INI; never used as real data')
        self.evidence_path=self.session/'observed.json';self.receipt_path=self.session/'receipt.json'
        self.receipt=dict(schema='stdp-private-source-operator-receipt-v1',binding=a.receipt_binding(self.session,self.manifest),verification='remote_operator_verified',operator='root',local_private_bytes_available=False,cryptographic_remote_attestation=False,host='synthetic-host',git_head='a'*40,observed_utc='2026-10-20T10:00:00+00:00',records=[dict(path=k,observed_sha256=v[0],expected_sha256=v[0],size_bytes=v[1],resolved_path='/synthetic/root/'+k) for k,v in a.PRIVATE.items()],stdout_evidence_path=self.evidence_path.relative_to(a.ROOT).as_posix(),limitations=['SYNTHETIC TEST ONLY; not hardware evidence'])
        self.save()

    def tearDown(self):self.tmp.cleanup()

    def save(self):
        evidence=dict(evidence_type='operator_transcription_of_visible_terminal_stdout',identity={k:self.receipt[k] for k in ('observed_utc','host','git_head')},binding=self.receipt['binding'],records=self.receipt['records'])
        self.evidence_path.write_text(json.dumps(evidence),encoding='utf-8')
        self.receipt['stdout_evidence_sha256']=a.sha(self.evidence_path)
        self.receipt_path.write_text(json.dumps(self.receipt),encoding='utf-8')

    def verify(self):return a.verify_receipt(self.receipt_path,self.session,self.manifest)

    def test_exact_static_replay_identity(self):
        self.assertEqual(a.verify_replay_identity()['numeric_tail_lf_sha256'],a.TAIL_SHA)

    def test_exact_synthetic_receipt_accepted_with_limits(self):
        result=self.verify();self.assertFalse(result['strict_all_source_local_bytes_pass']);self.assertFalse(result['cryptographic_remote_attestation'])

    def test_rehashed_binding_and_private_source_mutations_rejected(self):
        original=copy.deepcopy(self.receipt)
        for field in ('acquisition_session','start_manifest_sha256','manifest_sha256','source_inventory_sha256','board_ini_sha256','launcher_sha256','hardware_source_freeze_sha256','qualifying_sim_audit_sha256'):
            self.receipt=copy.deepcopy(original);self.receipt['binding'][field]='wrong';self.save()
            with self.subTest(field=field),self.assertRaises(ValueError):self.verify()
        for key,value in [('observed_sha256','0'*64),('expected_sha256','0'*64),('size_bytes',1),('resolved_path','/wrong/path')]:
            self.receipt=copy.deepcopy(original);self.receipt['records'][0][key]=value;self.save()
            with self.subTest(field=key),self.assertRaises(ValueError):self.verify()
        self.receipt=copy.deepcopy(original);self.receipt['observed_utc']='2026-10-19T00:00:00+00:00';self.save()
        with self.assertRaises(ValueError):self.verify()
        self.receipt=copy.deepcopy(original);self.receipt['records'].append(dict(self.receipt['records'][0],path='private/extra.py'));self.save()
        with self.assertRaises(ValueError):self.verify()

    def test_ini_start_source_and_observation_tampering_rejected(self):
        for name in ('board.ini','start_manifest.json','manifest.json'):
            path=self.session/name;saved=path.read_bytes();path.write_bytes(saved+b' ')
            with self.subTest(name=name),self.assertRaises(ValueError):self.verify()
            path.write_bytes(saved)
        self.evidence_path.write_text('{}')
        with self.assertRaises(ValueError):self.verify()
        self.save();wrong=dict(self.manifest,source='simulator')
        with self.assertRaises(ValueError):a.verify_receipt(self.receipt_path,self.session,wrong)


class SavedSimulatorReplayTests(unittest.TestCase):
    def test_saved_native_sim_identical_numeric_results(self):
        session=a.ROOT/'results/conductance_stdp_calibrated_20261019/d27da87e-9928-4303-971e-215afbc63158'
        result=a.audit(session)
        previous=json.loads((a.REVIEW/'NATIVE_SIM_INDEPENDENT_AUDIT.json').read_text())
        current=json.loads(json.dumps(result['native_replay']))
        self.assertEqual({k:v for k,v in current.items() if k!='auditor_sha256'}, {k:v for k,v in previous.items() if k!='auditor_sha256'})
        self.assertFalse(result['hardware_io_executed'])
        self.assertFalse(result['old_strict_local_full_audit_pass_claimed'])


if __name__=='__main__':unittest.main()
