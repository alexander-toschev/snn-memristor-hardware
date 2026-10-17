"""Offline tests: fake Connector only, no native simulator or physical board."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import audit_preparation as audit
import all256_mvm as runtime
from tests.test_calibrated_mvm import FakeConnector, converters


class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'.tmp')
        cls.folder=Path(cls.temp.name)
        worker=runtime.Acquisition(FakeConnector(),converters,cls.folder,'audit-fake',runtime.SPEC,'hardware',runtime.schedule(runtime.SPEC))
        with contextlib.redirect_stdout(io.StringIO()):result=worker.run()
        assert result['status']=='completed'
        (cls.folder/'protocol.json').write_bytes((audit.ROOT/runtime.CONFIG).read_bytes())
        (cls.folder/'schedule.json').write_text(json.dumps(runtime.schedule(runtime.SPEC)),encoding='utf-8')
        (cls.folder/'board.ini').write_text('[board]\ndac_bit=12\nvol_ref_dac=5\nadc_bit=14\nvol_ref_adc=5\ngain=11\nres_load=3000\nres_switches=10\nsum_gain=330\n',encoding='utf-8')
        cls.manifest=dict(source='hardware',session_id='audit-fake',status='completed',source_sha256=audit.SOURCE_PINS,
            schedule_sha256=audit.SCHEDULE_SHA,impact_attempts=worker.sequence,
            result_sha256={n:audit.sha(cls.folder/n) for n in ('protocol.json','schedule.json','board.ini','raw.jsonl','observations.json')})
        (cls.folder/'manifest.json').write_text(json.dumps(cls.manifest),encoding='utf-8')
        cls.analysis=dict(C=[[float(i==j) for j in range(8)] for i in range(8)],d=[0.]*8,zero=[0.]*8,
            basis_H=[[50.]*32 for _ in range(8)],selection_by_wl=[dict(passed=False) for _ in range(8)],
            selected_model='identity',lif=[dict(status='inconclusive')])

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    @contextlib.contextmanager
    def replace_and_rehash(self,name,content):
        path=self.folder/name;original=path.read_bytes();m=(self.folder/'manifest.json').read_bytes()
        try:
            path.write_bytes(content);manifest=json.loads(m);manifest['result_sha256'][name]=audit.sha(path)
            (self.folder/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8');yield
        finally:path.write_bytes(original);(self.folder/'manifest.json').write_bytes(m)

    def test_independent_raw_guards_and_exact_count(self):
        result,obs=audit.audit_raw(self.folder,'hardware')
        self.assertTrue(result['passed']);self.assertTrue(result['guards_recomputed'])
        self.assertEqual((result['paired_impacts'],result['complete_vectors'],result['skipped_vectors']),(13736,1525,605))
        self.assertEqual(len(obs['readbacks']),1536)

    def test_modified_guard_detected_even_with_updated_file_hash(self):
        lines=(self.folder/'raw.jsonl').read_text().splitlines()
        index=next(i for i,line in enumerate(lines) if json.loads(line)['event']=='vector_guard')
        row=json.loads(lines[index]);row['predictions'][0]['mode9_model_current_a']=1.;lines[index]=json.dumps(row)
        with self.replace_and_rehash('raw.jsonl',('\n'.join(lines)+'\n').encode()):
            with self.assertRaisesRegex(ValueError,'Independent guard mismatch'):audit.audit_raw(self.folder,'hardware')

    def test_modified_compact_detected_even_with_updated_file_hash(self):
        obs=audit.load(self.folder/'observations.json');obs['vectors'][0]['adc'][0]=1
        with self.replace_and_rehash('observations.json',json.dumps(obs).encode()):
            with self.assertRaisesRegex(ValueError,'Raw/compact mismatch'):audit.audit_raw(self.folder,'hardware')

    def test_single_evidence_separate_from_failed_row_selection(self):
        result=audit.scientific_review(audit.load(self.folder/'observations.json'),self.analysis)
        self.assertEqual(result['raw_single_supported'],256)
        self.assertEqual(result['single_and_observed_mixed_pass'],256)
        self.assertTrue(all(c['row_selection_claim'] is False for c in result['cells']))
        self.assertEqual(result['heldout_metrics']['test_rows']['both_pass'],256)
        self.assertEqual(result['heldout_metrics']['test_signed']['both_pass'],128)
        self.assertFalse(result['full_cifar_qualified'])

    def test_mean_and_noise_are_separate_and_signed_mean_cancellation(self):
        H=[[20.]*32 for _ in range(8)];C=[[float(i==j) for j in range(8)] for i in range(8)]
        data={(0,1):[[40.+(noise if w==0 else 0.) for w in range(8)] for noise in (-10,-5,0,5,10)]}
        result=audit.heldout_metrics(data,H,[0.]*8,C,[0.]*8,signed=True)
        self.assertEqual(result['mean_pass'],4);self.assertEqual(result['repeatability_pass'],3)

    def test_conversion_matches_native_without_import_in_auditor(self):
        for adc in (20,200,1000,2500):
            r,g=audit.convert_read(adc)
            self.assertEqual(r,converters.convert_adc_to_res(11,3000,.1,14,5,10,adc));self.assertEqual(g,1/r)
        with self.assertRaises(ValueError):audit.convert_read(0)
        with self.assertRaises(ValueError):audit.convert_read(19)

    def test_paths_and_html_no_false_global_qualification(self):
        for name in ('../escape','C:/outside','/outside'):
            with self.assertRaises(ValueError):audit.local(name)
        raw,obs=audit.audit_raw(self.folder,'hardware');science=audit.scientific_review(obs,self.analysis)
        page=audit.render(dict(sessions=dict(hardware=dict(raw_audit=raw,science=science))))
        self.assertEqual(page.count('class="cell '),256)
        self.assertIn('Ни один цвет не означает программируемость',page)
        self.assertIn('Выбор/fit C не повторялся',page)


if __name__=='__main__':unittest.main()
