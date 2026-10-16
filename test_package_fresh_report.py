"""Offline packager binding/bounds tests, no backend."""
import hashlib,json,tempfile,unittest
from pathlib import Path
import package_fresh_report as p

class Tests(unittest.TestCase):
    def test_relative_paths_cannot_escape(self):
        with self.assertRaises(ValueError):p.local(Path.cwd(),'../elsewhere')
        with self.assertRaises(ValueError):p.local(Path.cwd(),str(Path.cwd().resolve()))
    def test_missing_science_hash_rejects_before_any_output(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root=Path(tmp).resolve();source=root/'science';source.mkdir();(source/'SCIENCE.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'SCIENCE SHA'):p.package(root,'science','0'*64,'output')
            self.assertFalse((root/'output').exists())
    def test_source_destination_overlap_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root=Path(tmp).resolve();(root/'science').mkdir()
            for out in ('science','science/nested'):
                with self.assertRaises(ValueError):p.package(root,'science','0'*64,out)
    def test_manifest_and_audit_binding_and_mutation(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root=Path(tmp).resolve();d=root/'session';d.mkdir();auditor=next(iter(p.AUDITORS))
            m=dict(protocol_id=p.AUDITORS[auditor],source='simulator',session_id='test',status='completed',
                   invocation={'git':{'commit':'source'}},source_sha256={})
            (d/'manifest.json').write_text(json.dumps(m));mh=p.sha(d/'manifest.json')
            a=dict(auditor_sha256=auditor,audit_status='passed',errors=[],manifest_sha256=mh,session_id='test',
                   source='simulator',acquisition_complete=True,checks=10)
            (root/'audit.json').write_text(json.dumps(a))
            row=dict(session='session',audit='audit.json',manifest_sha256=mh,audit_sha256=p.sha(root/'audit.json'),
                auditor_sha256=auditor,session_id='test',source='simulator',source_commit='source',complete=True,
                weights=[{}]*300,cells=[{}]*256,full_time_outputs=2400)
            self.assertEqual(p.bind_session(root,row)['audit_checks'],10)
            m['status']='failed';a['acquisition_complete']=False
            (d/'manifest.json').write_text(json.dumps(m));row['manifest_sha256']=p.sha(d/'manifest.json')
            a['manifest_sha256']=row['manifest_sha256'];(root/'audit.json').write_text(json.dumps(a))
            row.update(audit_sha256=p.sha(root/'audit.json'),complete=False,weights=[{}]*7,full_time_outputs=0)
            partial=p.bind_session(root,row)
            self.assertFalse(partial['acquisition_complete'])
            self.assertEqual(partial['observed_weight_records'],7)
            self.assertEqual(partial['acquisition_status'],'failed')
            row['source']='hardware'
            with self.assertRaisesRegex(ValueError,'Backend'):p.bind_session(root,row)
            row['source']='simulator';(d/'manifest.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'Manifest SHA'):p.bind_session(root,row)
    def test_csv_nulls_and_compact_fields(self):
        r=p.expected_weight_rows([dict(filter=0,tap=8,cell=[0,1],empty=None,leg_evidence={'large':'omitted'})])
        self.assertEqual(r,[dict(filter='0',tap='8',cell='[0, 1]',empty='')])

if __name__=='__main__':unittest.main()
