"""Offline fake Connector/bytes only; no native SIM or hardware acquisition."""
import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest

import test_full_case0_native as fixtures
from tools import cifar_prefix_replay as replay
from tools import audit_cifar_interrupted_prefix as wrapper


class PrefixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.NativeTests.setUpClass()
        cls.rows=fixtures.NativeTests.trace.rows
        cls.image=fixtures.NativeTests.image;cls.weights=fixtures.NativeTests.weights;cls.pin=fixtures.NativeTests.pin
        cls.end=next(i+1 for i,r in enumerate(cls.rows) if r['event']=='stage_complete')
        cls.prefix=cls.rows[:cls.end]

    def call(self,rows,**kwargs):
        return replay.audit_range(iter(rows),self.image,self.weights,self.pin,stop_stage=3,**kwargs)

    def test_original_complete_smoke_and_literal_prefix(self):
        payload,report=self.call(self.rows)
        self.assertEqual(sorted(payload),[0,1,2])
        self.assertFalse(report['complete_full_tensor'])
        for stage in report['accepted_stages']:
            self.assertEqual(stage['shape'],[12,4,1,1])
            self.assertEqual(len(payload[stage['stage']]['raw']),12*4*4)
        prefix,partial=self.call(self.prefix,allow_prefix_eof=True)
        self.assertEqual(prefix[0],payload[0])
        self.assertEqual(partial['stop_stage_exclusive'],1)
        self.assertFalse(partial['terminal_session_event_verified'])
        self.assertFalse(partial['frozen_full_auditor_pass_claimed'])

    def test_incomplete_or_forged_stage_rejected(self):
        variants=[]
        variants.append(self.prefix[:-1])
        missing=copy.deepcopy(self.prefix)
        missing.pop(next(i for i,r in enumerate(missing) if r['event']=='result'))
        variants.append(missing)
        nofinal=[r for r in self.prefix if not(r['event']=='vector_complete' and r.get('phase')=='final_basis')]
        variants.append(nofinal)
        for event in ('basis_complete','heldout_complete','stage_complete'):
            duplicate=copy.deepcopy(self.prefix)
            i=next(i for i,r in enumerate(duplicate) if r['event']==event)
            duplicate.insert(i,copy.deepcopy(duplicate[i]));variants.append(duplicate)
        with self.assertRaises(ValueError):self.call(self.prefix)
        for rows in variants:
            with self.subTest(length=len(rows)),self.assertRaises((ValueError,StopIteration,KeyError)):
                self.call(rows,allow_prefix_eof=True)

    def test_stage_segment_must_bind_declared_range(self):
        rows=copy.deepcopy(self.prefix)
        rows[0].update(acquisition_kind='stage_segment',stage_range=[0,1])
        # Real record emission is tested in the resumable collector suite. This
        # fixture tests only range semantics, not native evidence production.
        rows.append(dict(event='session_complete',source='simulator',stage=0,status='completed',
                         impact_attempts=max(r.get('task_id',0) for r in rows),
                         programming_pulses=sum(r['event']=='request' and r['command_si']['mode_flag']==7
                                                and bool(r['command_si']['vol']) for r in rows)))
        payload,report=replay.audit_range(rows,self.image,self.weights,self.pin,stop_stage=1)
        self.assertEqual(list(payload),[0]);self.assertTrue(report['terminal_session_event_verified'])
        with self.assertRaises(ValueError):self.call(rows)

    def test_crc_closed_parts_only_and_contiguous_names(self):
        base=Path(__file__).resolve().parents[1]/'.tmp'
        base.mkdir(exist_ok=True)
        first=b''.join(json.dumps(v).encode()+b'\n' for v in self.prefix)
        tail=b''.join(json.dumps(v).encode()+b'\n' for v in self.rows[self.end:self.end+20])
        with tempfile.TemporaryDirectory(dir=base) as tmp:
            raw=Path(tmp)
            (raw/'part-000000.jsonl.gz').write_bytes(gzip.compress(first))
            (raw/'part-000001.jsonl.gz').write_bytes(gzip.compress(tail)[:-8])
            inv=wrapper.inventory(raw)
            self.assertEqual(inv['admitted_parts'],['part-000000.jsonl.gz'])
            self.assertEqual(inv['prefix_records'],self.end)
            self.assertFalse(inv['parts'][1]['admitted'])
            self.assertEqual(list(wrapper.prefix_records(raw,inv)),self.prefix)
            # A CRC-bad first part cannot contribute even valid-looking lines.
            packed=bytearray((raw/'part-000000.jsonl.gz').read_bytes());packed[-8]^=1
            (raw/'part-000000.jsonl.gz').write_bytes(packed)
            bad=wrapper.inventory(raw)
            self.assertEqual(bad['prefix_records'],0)
            self.assertFalse(any(p['admitted'] for p in bad['parts']))
        with tempfile.TemporaryDirectory(dir=base) as tmp:
            raw=Path(tmp)
            (raw/'part-000001.jsonl.gz').write_bytes(gzip.compress(first))
            self.assertEqual(wrapper.inventory(raw)['prefix_records'],0)

    def test_changed_inventory_bytes_rejected(self):
        base=Path(__file__).resolve().parents[1]/'.tmp'
        with tempfile.TemporaryDirectory(dir=base) as tmp:
            raw=Path(tmp);path=raw/'part-000000.jsonl.gz'
            path.write_bytes(gzip.compress(b''.join(json.dumps(v).encode()+b'\n' for v in self.prefix)))
            inv=wrapper.inventory(raw)
            path.write_bytes(gzip.compress(b'{}\n'))
            with self.assertRaises(ValueError):list(wrapper.prefix_records(raw,inv))

    def test_portable_policy_only_excluded_condition(self):
        freeze=next(r for r in self.prefix if r['event']=='calibration_frozen')
        basis=next(r for r in self.prefix if r['event']=='basis_complete')['basis']
        local=copy.deepcopy(freeze['matrix']);recorded=copy.deepcopy(local)
        self.assertLess(local['fit_rank'],8)
        # Direct policy fixture: equal columns imply a clearly separated rank1.
        fit=[dict(mask=[i%32],centered_v=[.006]*8) for i in range(192)]
        local['condition']=1e19;recorded['condition']=2e19
        proof=replay.compare_calibration(local,recorded,basis,fit,True)
        self.assertEqual(proof['strict_numeric_replay'],'FAIL_condition_only')
        with self.assertRaises(ValueError):replay.compare_calibration(local,recorded,basis,fit,False)
        changed=copy.deepcopy(recorded);changed['matrix'][0][0]+=.001
        with self.assertRaises(ValueError):replay.compare_calibration(local,changed,basis,fit,True)
        for condition in (1e5,float('inf'),float('nan')):
            changed=copy.deepcopy(recorded);changed['condition']=condition
            with self.assertRaises(ValueError):replay.compare_calibration(local,changed,basis,fit,True)

    def test_portable_rank_margin_and_candidate_restriction(self):
        import numpy as np
        freeze=next(r for r in self.prefix if r['event']=='calibration_frozen')
        local=copy.deepcopy(freeze['matrix']);recorded=copy.deepcopy(local)
        local['condition']=1e19;recorded['condition']=2e19
        x=np.zeros((192,8));x[0,0]=1.;x[1,1]=192*np.finfo(float).eps*2
        fit=[dict(mask=[i%32],centered_v=row.tolist()) for i,row in enumerate(x)]
        basis=next(r for r in self.prefix if r['event']=='basis_complete')['basis']
        with self.assertRaisesRegex(ValueError,'rank margin'):
            replay.compare_calibration(local,recorded,basis,fit,True)
        for target in (local,recorded):target['candidates'].append(dict(name='full'))
        with self.assertRaisesRegex(ValueError,'full-matrix'):
            replay.compare_calibration(local,recorded,basis,fit,True)

    def test_private_receipt_exact_allowlist_and_evidence_binding(self):
        root=wrapper.ROOT
        receipt_path=root/'reports/2026-10-19/cifar-full-case0-review/PRIVATE_SOURCE_OPERATOR_RECEIPT.json'
        receipt=json.loads(receipt_path.read_text())
        session=root/receipt['acquisition_session']
        pins={name:pair[0] for name,pair in wrapper.PRIVATE.items()}
        result=wrapper.verify_private_receipt(receipt_path,session,pins)
        self.assertFalse(result['strict_all_source_local_bytes_pass'])
        with self.assertRaises(ValueError):wrapper.verify_private_receipt(receipt_path,session.with_name('another'),pins)
        variants=[]
        changed=copy.deepcopy(receipt);changed['records'][0]['observed_sha256']='0'*64;variants.append(changed)
        changed=copy.deepcopy(receipt);changed['records'][0]['path']='other.py';variants.append(changed)
        changed=copy.deepcopy(receipt);changed['host']='other';variants.append(changed)
        changed=copy.deepcopy(receipt);changed['stdout_evidence_sha256']='0'*64;variants.append(changed)
        with tempfile.TemporaryDirectory(dir=root/'.tmp') as tmp:
            altered=Path(tmp)/'receipt.json'
            for changed in variants:
                altered.write_text(json.dumps(changed))
                with self.subTest(changed=changed['host']),self.assertRaises(ValueError):
                    wrapper.verify_private_receipt(altered,session,pins)


if __name__=='__main__':unittest.main()
