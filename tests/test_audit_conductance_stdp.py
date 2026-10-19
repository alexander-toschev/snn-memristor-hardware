"""Offline corruption checks against recorded native SIM; original files unchanged."""
import copy
import gzip
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from tools.audit_conductance_stdp import ROOT, audit, digest

SESSION = ROOT/'results/conductance_stdp_20261019/2ac0818a-5861-418f-b3a9-e0a2b7f11b78'


def negative_checks():
    read_bytes,read_text=Path.read_bytes,Path.read_text
    base={p:p.read_bytes() for p in SESSION.rglob('*') if p.is_file()}
    outcomes=[]
    cases={
        'raw_hash_corruption':'Result bytes mismatch',
        'rehashed_lif_corruption':'Numeric replay mismatch',
        'rehashed_timing_corruption':'Input timing mismatch',
        'rehashed_guard_corruption':'Pulse current exceeded',
        'rehashed_bracket_duplicate_reference':'Bracket task IDs must be distinct',
        'rehashed_top_bracket_reuse':'Duplicate top-level bracket',
        'rehashed_checkpoint_read_other_cell':'Checkpoint read cell/mode differs',
        'rehashed_nested_bracket_other_cell':'Checkpoint MVM cell differs',
    }
    for case,reason in cases.items():
        files=copy.copy(base)
        if case=='raw_hash_corruption':
            files[SESSION/'raw/raw.jsonl.gz']+=b'bad'
        else:
            rows=[json.loads(x) for x in gzip.decompress(files[SESSION/'raw/raw.jsonl.gz']).splitlines()]
            if case=='rehashed_lif_corruption':
                next(r for r in rows if r['event']=='lif_step')['u_after']+=.1
            elif case=='rehashed_timing_corruption':
                next(r for r in rows if r['event']=='lif_step')['pre']=[1,0,0,0]
            elif case=='rehashed_guard_corruption':
                next(r for r in rows if r['event']=='result' and r['phase']=='pulse_guard')['native_result'][0]=1.
            elif case=='rehashed_bracket_duplicate_reference':
                r=next(r for r in rows if r['event']=='bracket');r['after']=copy.deepcopy(r['before'])
            elif case=='rehashed_top_bracket_reuse':
                indices=[i for i,r in enumerate(rows) if r['event']=='bracket']
                rows[indices[1]]=copy.deepcopy(rows[indices[0]])
            elif case=='rehashed_checkpoint_read_other_cell':
                # Preserve repetition/contiguity and valid R/ADC but bind to another BL.
                checkpoints=[r for r in rows if r['event']=='checkpoint']
                for name in ('reads','q','native_inverse_r'):
                    checkpoints[0][name]=copy.deepcopy(checkpoints[1][name])
            elif case=='rehashed_nested_bracket_other_cell':
                checkpoints=[r for r in rows if r['event']=='checkpoint']
                checkpoints[0]['responses']=copy.deepcopy(checkpoints[1]['responses'])
            data=('\n'.join(json.dumps(x) for x in rows)+'\n').encode();packed=gzip.compress(data,mtime=0)
            index=json.loads(files[SESSION/'raw/index.json'])
            index.update(lines=len(rows),expanded_bytes=len(data),expanded_sha256=digest(data),
                         compressed_bytes=len(packed),compressed_sha256=digest(packed))
            files[SESSION/'raw/raw.jsonl.gz']=packed;files[SESSION/'raw/index.json']=json.dumps(index).encode()
            manifest=json.loads(files[SESSION/'manifest.json'])
            for name in ('raw/raw.jsonl.gz','raw/index.json'):
                manifest['result_sha256'][name]=digest(files[SESSION/name])
            files[SESSION/'manifest.json']=json.dumps(manifest).encode()
        def rb(p):return files[p] if p in files else read_bytes(p)
        def rt(p,*args,**kwargs):return files[p].decode() if p in files else read_text(p,*args,**kwargs)
        with patch.object(Path,'read_bytes',rb),patch.object(Path,'read_text',rt):
            try:audit(SESSION)
            except ValueError as exc:
                if reason not in str(exc):raise AssertionError((case,'Wrong rejection',str(exc))) from exc
                outcomes.append(dict(case=case,rejected=True,reason=str(exc)))
            else:raise AssertionError('Corruption accepted: '+case)
    return outcomes


@unittest.skipUnless((SESSION/'manifest.json').is_file(), 'Recorded native SIM dataset required')
class ActualEvidenceTests(unittest.TestCase):
    def test_actual_native_evidence(self):
        result=audit(SESSION)
        self.assertEqual(result['status'],'pass')
        self.assertEqual(result['training_pulses'],0)

    def test_eight_corruptions_are_rejected(self):
        self.assertEqual(len(negative_checks()),8)


if __name__=='__main__':unittest.main()
