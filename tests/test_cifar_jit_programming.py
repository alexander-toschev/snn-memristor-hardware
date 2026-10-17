"""Offline only. Actual native converters, FakeConnector and a synthetic clock."""
import contextlib
from datetime import datetime,timedelta,timezone
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cifar_jit_programming as j
from tests.test_cifar_feedback import FakeConnector,converters


class Clock:
    def __init__(self):self.value=datetime(2026,10,16,17,tzinfo=timezone.utc);self.elapsed=0.
    def now(self):return self.value+timedelta(seconds=self.elapsed)
    def mono(self):return self.elapsed
    def sleep(self,seconds):self.elapsed+=seconds


def raw(folder):
    index=json.loads((folder/'raw_parts.json').read_text())
    return [json.loads(line) for p in index['parts'] for line in (folder/p['path']).read_text().splitlines()]


class JITTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.plan,cls.route=j.load_inputs()

    def runner(self,folder,fake,clock=None):
        clock=clock or Clock()
        return j.JITRun(fake,converters,folder,'OFFLINE_FAKE',self.plan,'simulator',clock=clock.now,monotonic=clock.mono,sleep=clock.sleep)

    def test_exact_frozen_route_budget_and_no_new_training_targets(self):
        self.assertEqual([s['tap_index'] for s in self.route],list(range(9)))
        self.assertEqual(self.route[5]['negative_cell'],[3,8])
        self.assertEqual(18*(2*244+8+32+811+4+32+9*32),29934)
        self.assertEqual(18*(2*55+109),3942)
        self.assertEqual(self.plan['maximum_impacts'],29934)
        self.assertEqual(self.plan['maximum_programming_pulses'],3942)
        with patch.object(j,'sha256',return_value='corrupt'):
            with self.assertRaises(ValueError):j.load_inputs()

    def test_actual_fake_two_cycles_maps_sentinels_and_retention(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            folder=Path(tmp);fake=FakeConnector(reset_step=100,set_step=1);clock=Clock()
            run=self.runner(folder,fake,clock);result=run.collect(self.route)
            self.assertEqual(result['status'],'completed');self.assertEqual(len(result['decisions']),18)
            self.assertEqual([(d['cycle'],d['pair_index']) for d in result['decisions']],[(c,i) for c in range(2) for i in range(9)])
            self.assertEqual(len((folder/'disturbance_maps.jsonl').read_text().splitlines()),18)
            self.assertEqual(len((folder/'sentinel.jsonl').read_text().splitlines()),54)
            maps=[json.loads(line) for line in (folder/'disturbance_maps.jsonl').read_text().splitlines()]
            self.assertTrue(all(len(m['observed'])==9 and all(len(p['samples'])==5 for p in m['observed']) for m in maps))
            self.assertEqual(clock.elapsed,18.)
            events=raw(folder);requests=[r for r in events if r['event']=='request']
            self.assertEqual(len(requests),len(fake.calls));self.assertLessEqual(len(requests),29934)
            self.assertTrue(all(r['command_si']['mode_flag'] in (7,10) for r in requests))
            self.assertFalse(any(r['phase']=='cifar' for r in requests))
            for d in result['decisions']:
                key=(d['cycle'],d['pair_index'])
                same=[r for r in requests if (r.get('cycle'),r.get('pair_index'))==key]
                gate=max(r['task_id'] for r in same if r['phase']=='fresh_pair_gate')
                sets=[r['task_id'] for r in same if r['phase']=='pair_set']
                self.assertTrue(all(i>gate for i in sets))
                self.assertTrue(all(d['retention']['elapsed_monotonic_s']>=1 for d in result['decisions']))

    def test_second_reset_invalidates_first_quiet_then_no_SET_but_next_pair(self):
        p=tuple(self.route[0]['positive_cell']);n=tuple(self.route[0]['negative_cell'])
        class Disturb(FakeConnector):
            def impact(self,task):
                disturb=task['mode_flag']==7 and task['vol'] and task['sign']==0 and (task['wl'],task['bl'])==n
                result=super().impact(task)
                if disturb:self.states[p]=10
                return result
        fake=Disturb(reset_step=100,set_step=1);fake.states[n]=200
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            folder=Path(tmp);result=self.runner(folder,fake).collect(self.route)
            first=result['decisions'][0]
            self.assertTrue(first['positive_reset']['eligible']);self.assertTrue(first['negative_reset']['eligible'])
            self.assertFalse(first['fresh_pair_gate']['both_legs_quiet'])
            self.assertEqual(first['outcome'],'fresh_quiet_gate_failed_no_SET')
            self.assertNotIn('program',first);self.assertEqual(result['decisions'][1]['status'],'completed')
            self.assertFalse(result['programming_fidelity_pass'])
            self.assertFalse(any(r['event']=='request' and r.get('cycle')==0 and r.get('pair_index')==0 and r['phase']=='pair_set' for r in raw(folder)))

    def test_deadline_stops_before_next_command_and_preserves_raw(self):
        clock=Clock()
        class Expire(FakeConnector):
            def impact(self,task):
                result=super().impact(task);clock.value=datetime.fromisoformat(j.SPEC['cutoff_utc']);return result
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            folder=Path(tmp);fake=Expire();run=self.runner(folder,fake,clock)
            with self.assertRaises(j.DeadlineReached):run.collect(self.route)
            self.assertEqual(len(fake.calls),1);events=raw(folder)
            self.assertEqual(sum(r['event']=='deadline_stop' for r in events),1)
            self.assertEqual(sum(r['event']=='request' for r in events),1)
            summary=json.loads((folder/'summary.json').read_text());self.assertFalse(summary['technical_completed'])

    def test_current_guard_is_global_without_impact_and_transport_is_not_negative_gate(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            fake=FakeConnector();run=self.runner(Path(tmp),fake);run.last_mode7[(2,3)]=(1.,0)
            try:
                with self.assertRaises(j.CurrentGuardStop):run.impact(2,3,7,.3,'guard_test',sign=1)
                self.assertEqual(len(fake.calls),0);self.assertEqual(run.skipped,1)
            finally:run.log.close()
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            fake=FakeConnector(fail=True);run=self.runner(Path(tmp),fake)
            with self.assertRaises(ConnectionError):run.collect(self.route)
            self.assertEqual(len(fake.calls),1)

    def test_per_decision_guard_and_short_retention_do_not_issue_commands(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            fake=FakeConnector();clock=Clock();run=self.runner(Path(tmp),fake,clock)
            try:
                run.pair_start_counts=(0,0);run.sequence=1663
                with self.assertRaisesRegex(ValueError,'Per-pair'):run.impact(2,3,7,0.,'limit')
                self.assertFalse(fake.calls)
                run.pair_start_counts=None;run.monotonic_cutoff=.5
                with self.assertRaises(j.DeadlineReached):run.retention(self.route[0],{'cycle':0,'pair_index':0})
                self.assertEqual(clock.elapsed,0.);self.assertFalse(fake.calls)
            finally:run.log.close()

    def test_main_closes_connector_and_saves_transport_failure(self):
        class Managed(FakeConnector):
            cb_type='real'
            def __init__(self):super().__init__(fail=True);self.closed=0
            def open_port(self):return True
            def close_port(self):self.closed+=1;return True
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp,contextlib.redirect_stdout(io.StringIO()):
            folder=Path(tmp);fake=Managed()
            with patch.object(j,'ROOT',folder),patch.object(j,'local_path',side_effect=lambda n:folder/n), \
                 patch.object(j,'load_inputs',return_value=(self.plan,self.route)), \
                 patch.object(j,'check_sources',return_value=(None,b'[board]\n',{})), \
                 patch.object(j,'sha256',return_value='synthetic_test_only'), \
                 patch.object(j,'launch_provenance',return_value={'synthetic_test_only':True}), \
                 patch.object(j,'utc_now',return_value=Clock().now()), \
                 patch.object(j,'native_connector',return_value=fake):
                self.assertEqual(j.main(['--run-hardware']),2)
            self.assertEqual(fake.closed,1);self.assertEqual(len(fake.calls),1)
            manifest=json.loads(next((folder/'results').rglob('manifest.json')).read_text())
            self.assertEqual(manifest['status'],'failed');self.assertEqual(manifest['impact_attempts'],1)
            self.assertIn('raw_parts.json',manifest['result_sha256']);self.assertFalse((folder/j.LOCK).exists())

    def test_frontdoor_no_default_IO_and_expired_open_rejected(self):
        import experiment
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            with patch.object(j,'main',return_value=7) as called:
                for mode in ('--run-simulator','--run-hardware','--dry-run'):
                    self.assertEqual(experiment.main(['--test-cifar-jit-programming',mode]),7)
                    called.assert_called_with([mode])
                for extra in ([],['--dry-run','--phase','phase1'],['--run-hardware','--run-simulator']):
                    with self.assertRaises(SystemExit):experiment.main(['--test-cifar-jit-programming']+extra)
            with patch.object(j,'native_connector',side_effect=AssertionError('No backend')),patch.object(j,'check_sources',return_value=(None,b'',{})):
                self.assertEqual(j.main([]),0)
                self.assertEqual(j.main(['--dry-run']),0)
                with patch.object(j,'utc_now',return_value=datetime.fromisoformat(j.SPEC['cutoff_utc'])):
                    with self.assertRaises(j.DeadlineReached):j.main(['--run-hardware'])


if __name__=='__main__':unittest.main()
