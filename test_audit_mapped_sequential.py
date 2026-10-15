"""Offline adversarial fixtures for independent mapped-sequential auditing."""
import contextlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from test_mapped_sequential_mvm import FakeConnector, converters, run
import audit_mapped_sequential as audit


class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(dir=audit.ROOT);cls.base=Path(cls.temp.name)/'base';cls.base.mkdir()
        worker=run.SequentialRun(FakeConnector(),converters,cls.base,'audit-fake',run.SPEC,'hardware',run.schedule(run.SPEC))
        with contextlib.redirect_stdout(io.StringIO()): result=worker.run()
        if result['status']!='completed':raise AssertionError(result)
        (cls.base/'protocol.json').write_bytes((audit.ROOT/'configs/mapped_sequential_mvm.json').read_bytes())
        (cls.base/'schedule.json').write_text(json.dumps(run.schedule(run.SPEC)),encoding='utf-8')
        (cls.base/'board.ini').write_text('[board]\ndac_bit=12\nvol_ref_dac=5\nadc_bit=14\nvol_ref_adc=5\ngain=11\nres_load=3000\nres_switches=10\nsum_gain=330\n',encoding='utf-8')
        manifest=dict(status='completed',source='hardware',session_id='audit-fake',impact_attempts=10706,
                      source_sha256=audit.PINS,schedule_sha256=audit.canonical(run.schedule(run.SPEC)))
        cls.write_manifest(cls.base,manifest)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    @staticmethod
    def write_manifest(folder,manifest=None):
        manifest=manifest or audit.load(folder/'manifest.json')
        manifest['result_sha256']={p.name:audit.sha(p) for p in folder.iterdir() if p.is_file() and p.name!='manifest.json'}
        (folder/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')

    def setUp(self):
        self.folder=Path(self.temp.name)/self._testMethodName;shutil.copytree(self.base,self.folder)

    def edit_json(self,name,fn):
        path=self.folder/name;data=audit.load(path);fn(data);path.write_text(json.dumps(data),encoding='utf-8');self.write_manifest(self.folder)

    def edit_raw(self,fn):
        path=self.folder/'raw.jsonl';rows=[json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]
        fn(rows);path.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8');self.write_manifest(self.folder)

    def test_complete_fixture_and_independent_schedule(self):
        self.assertEqual(audit.expected_schedule(run.SPEC),run.schedule(run.SPEC))
        result=audit.inspect(self.folder);self.assertTrue(result['passed']);self.assertEqual(result['impacts'],10706)

    def test_rehashed_compact_centered_value_rejected(self):
        self.edit_json('observations.json',lambda data:data['blocks'][0].update(centered_adc=999999))
        with self.assertRaisesRegex(ValueError,'bracket'):audit.inspect(self.folder)

    def test_rehashed_guard_rejected(self):
        def change(rows):
            next(r for r in rows if r['event']=='acquisition_guard')['decisions'][0]['mode9_model_current_a']=123.
        self.edit_raw(change)
        with self.assertRaisesRegex(ValueError,'guard'):audit.inspect(self.folder)

    def test_reference_commitment_must_precede_evaluation(self):
        def change(rows):
            index=next(i for i,r in enumerate(rows) if r['event']=='reference_frozen');event=rows.pop(index)
            rows.insert(index+4,event)
        self.edit_raw(change)
        with self.assertRaisesRegex(ValueError,'frozen|timing'):audit.inspect(self.folder)

    def test_rehashed_lif_substitution_rejected(self):
        self.edit_json('analysis.json',lambda data:data['lif'][0]['actual'][0][0].update(current=999))
        with self.assertRaisesRegex(ValueError,'LIF trajectories'):audit.inspect(self.folder)

    def test_rehashed_gate_substitution_rejected(self):
        self.edit_json('analysis.json',lambda data:data['coefficients'][0].update(final_supported=not data['coefficients'][0]['final_supported']))
        with self.assertRaisesRegex(ValueError,'Contrast gates'):audit.inspect(self.folder)

    def test_portable_paths_reject_escape(self):
        for value in ('../secret','D:/secret','/secret'):
            with self.assertRaises(ValueError):audit.local(value)

    def test_guarded_complete_session_retains_inconclusive(self):
        folder=Path(self.temp.name)/'guarded';folder.mkdir()
        worker=run.SequentialRun(FakeConnector(high_basis=True),converters,folder,'audit-fake',run.SPEC,'hardware',run.schedule(run.SPEC))
        with contextlib.redirect_stdout(io.StringIO()):result=worker.run()
        self.assertEqual(result['status'],'completed');self.assertGreater(result['skipped_commands'],0)
        for name in ('protocol.json','schedule.json','board.ini'):shutil.copyfile(self.base/name,folder/name)
        manifest=audit.load(self.base/'manifest.json');manifest['impact_attempts']=result['impact_attempts'];self.write_manifest(folder,manifest)
        report=audit.inspect(folder)
        self.assertEqual(report['mapped_contrasts_supported'],0)
        self.assertEqual(report['lif_statuses'],{'inconclusive':24})


if __name__=='__main__':unittest.main()
