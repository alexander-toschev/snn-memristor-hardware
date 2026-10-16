import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import compare_fresh_traces as c


class Tests(unittest.TestCase):
    def write(self,root,source,alter=False):
        directory=root/source;directory.mkdir();stream=io.StringIO(newline='')
        writer=csv.writer(stream,lineterminator='\n');writer.writerow(c.FIELDS)
        for f in range(4):
            for case in range(10):
                for repeat in range(5):
                    for t in range(12):
                        mismatch=alter and f==0 and case==0 and t==0
                        writer.writerow([f,case,case+100,repeat,t,1. if mismatch else 0.,0.,1. if mismatch else -.5,
                                         0. if mismatch else -.5,int(mismatch)])
        data=stream.getvalue().encode();(directory/'ACTUAL_TRACE.csv').write_bytes(data)
        auditor=next(iter(c.AUDITORS));metadata=dict(schema='audited-fresh-actual-trace-v2',generator_sha256=c.EXPORTER_SHA,
            analyzer_sha256=c.ANALYZER_SHA,auditor_sha256=auditor,protocol_id=c.AUDITORS[auditor],
            acquisition_complete=True,inventory_verified=True,actual_only=True,source=source,
            decoder_gain_v_per_weight=.25 if source=='hardware' else .03,units=c.UNITS,
            session_id=source,source_commit='a'*40,manifest_sha256='1'*64,audit_sha256='2'*64,
            fixture_sha256='3'*64,trained_weights_sha256='4'*64,planned_route_sha256='5'*64,summary_sha256='6'*64,
            planned_route_canonical_sha256='7'*64,
            route_canonicalization='parsed JSON; sort_keys=True; separators=(comma,colon); ensure_ascii=False; allow_nan=False; UTF-8; no trailing newline',
            csv=dict(path='ACTUAL_TRACE.csv',rows=2400,columns=c.FIELDS,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
        (directory/'TRACE_PROVENANCE.json').write_text(json.dumps(metadata));return directory,metadata
    def test_exact_arithmetic_repeat_vs_unique_and_direction(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as name:
            root=Path(name).resolve();self.write(root,'hardware',True);self.write(root,'simulator')
            result=c.compare(c.read_trace(root,'simulator'),c.read_trace(root,'hardware'))
            m=result['total'];self.assertEqual(m['spike_mismatches'],5);self.assertEqual(m['unique_logical_spike_mismatches'],1)
            self.assertAlmostEqual(m['conv']['rmse'],(5/2400)**.5);self.assertEqual(m['conv']['max_abs'],1)
            self.assertAlmostEqual(m['conv']['mean_difference'],5/2400)
            self.assertEqual(result['difference_direction'],'hardware minus simulator')
    def test_hash_mismatch_and_refreshed_wrongorder_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as name:
            root=Path(name).resolve();directory,m=self.write(root,'hardware');path=directory/'ACTUAL_TRACE.csv'
            data=path.read_bytes();lines=data.splitlines(keepends=True);lines[1],lines[2]=lines[2],lines[1];path.write_bytes(b''.join(lines))
            with self.assertRaisesRegex(ValueError,'SHA'):c.read_trace(root,'hardware')
            m['csv']['sha256']=c.sha(path);(directory/'TRACE_PROVENANCE.json').write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,'order'):c.read_trace(root,'hardware')
    def test_route_fixture_and_gain_mismatch_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as name:
            root=Path(name).resolve();self.write(root,'hardware');self.write(root,'simulator')
            a,b=c.read_trace(root,'hardware'),c.read_trace(root,'simulator')
            b['metadata']['planned_route_sha256']='8'*64
            self.assertFalse(c.compare(a,b)['route_raw_bytes_equal'])
            for key in ('planned_route_canonical_sha256','fixture_sha256','trained_weights_sha256'):
                original=b['metadata'][key];b['metadata'][key]='bad'
                with self.assertRaisesRegex(ValueError,'shared contract'):c.compare(a,b)
                b['metadata'][key]=original
            path=root/'simulator'/'TRACE_PROVENANCE.json';m=json.loads(path.read_text());m['decoder_gain_v_per_weight']=.25;path.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,'gain'):c.read_trace(root,'simulator')
    def test_v1_export_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as name:
            root=Path(name).resolve();directory,m=self.write(root,'hardware')
            m['schema']='audited-fresh-actual-trace-v1'
            (directory/'TRACE_PROVENANCE.json').write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,'v2 re-export'):c.read_trace(root,'hardware')
    def test_nan_and_missing_rows_rejected_after_hash_refresh(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as name:
            root=Path(name).resolve();directory,m=self.write(root,'hardware');path=directory/'ACTUAL_TRACE.csv';original=path.read_text()
            for data,message in [(original.replace('0.0,0.0,-0.5','nan,0.0,-0.5',1),'Nonfinite'),
                                 ('\n'.join(original.splitlines()[:-1])+'\n','Missing')]:
                path.write_text(data,newline='');m['csv'].update(bytes=path.stat().st_size,sha256=c.sha(path))
                (directory/'TRACE_PROVENANCE.json').write_text(json.dumps(m))
                with self.assertRaisesRegex(ValueError,message):c.read_trace(root,'hardware')
    def test_output_and_overlap_failclosed(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as name:
            root=Path(name).resolve();self.write(root,'hardware');self.write(root,'simulator')
            with self.assertRaises(ValueError):c.run(root,'hardware','simulator','hardware/new')
            result=c.run(root,'hardware','simulator','comparison')
            self.assertEqual(result['timepoints'],2400);self.assertEqual(result['spike_mismatches'],0)
            with self.assertRaises(ValueError):c.run(root,'hardware','simulator','comparison')

if __name__=='__main__':unittest.main()
