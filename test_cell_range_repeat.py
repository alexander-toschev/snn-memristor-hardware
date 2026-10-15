"""Offline bounded interval tests with native converters and a fake public Connector."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from manager.service import converters
from manager.terminate import EqualIn
import cell_range_repeat as interval
import experiment

ROOT=Path(__file__).resolve().parent


class FakeConnector:
    cb_type='real'
    def __init__(self,overshoot=False,reset_miss=False,set_miss=False,bad_r=False,fail_set=False):
        self.commands=[];self.q={};self.closed=0
        self.overshoot,self.reset_miss,self.set_miss,self.bad_r,self.fail_set=overshoot,reset_miss,set_miss,bad_r,fail_set
    def open_port(self):return True
    def close_port(self):self.closed+=1;return True
    def impact(self,task):
        self.commands.append(copy.deepcopy(task));assert task['mode_flag'] in (7,10)
        if task['mode_flag']==10:
            assert task['id']==0 and len(task['vol'])==32
            for i,v in enumerate(task['vol']):task['vol'][i]=converters.convert_volt_to_dac(12,5,v)
            return 1e-9,0,sum(task['vol'])
        key=task['wl'],task['bl'];self.q.setdefault(key,.02)
        if task['vol']:
            if task['sign']==0:self.q[key]=.02 if self.reset_miss else .004
            else:
                if self.fail_set:raise TimeoutError('SET response lost')
                self.q[key]=.051 if self.overshoot else .004 if self.set_miss else (.03 if key[0]==5 else .04)+round((task['vol']-.3)/.01)*.005
        adc=converters.convert_res_to_adc(11,3000,.3,14,5,10,330/self.q[key])
        r=converters.convert_adc_to_res(11,3000,.3,14,5,10,adc)
        return r+1 if self.bad_r else r,task['id'],adc
    def __getattr__(self,name):raise AssertionError('Forbidden Connector attribute: '+name)


class IntervalTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory(dir=ROOT);self.addCleanup(tmp.cleanup);self.output=Path(tmp.name)
        self.plan=json.loads(interval.local_path(interval.BASE_CONFIG).read_text())
        self.plan.update(json.loads((ROOT/'configs/cell_range_repeat.json').read_text()))
    def run_fake(self,fake,delay=True):
        clock=[0.0]
        def wait(seconds):
            if delay:clock[0]+=seconds
        worker=interval.IntervalRun(fake,converters,self.output,'offline',self.plan,'hardware',EqualIn,wait)
        with patch.object(interval.time,'monotonic',side_effect=lambda:clock[0]),contextlib.redirect_stdout(io.StringIO()):
            result=worker.run()
        raw=[json.loads(l) for l in (self.output/'raw.jsonl').read_text().splitlines()]
        return result,raw
    def test_full_three_cells_three_cycles_and_native_intervals(self):
        fake=FakeConnector();result,raw=self.run_fake(fake)
        self.assertEqual(result['status'],'completed');self.assertEqual(len(result['cycles']),9)
        targets=[t for c in result['cycles'] for t in c['targets']]
        self.assertEqual(len(targets),27);self.assertTrue(all(t['interval_and_mvm_pass'] for t in targets))
        self.assertTrue(all(t['hit_origin']=='programming_response' for t in targets))
        self.assertEqual(result['impact_attempts'],1278);self.assertEqual(result['programming_pulse_attempts'],36)
        self.assertEqual(fake.closed,1)
        self.assertEqual(len([r for r in raw if r['event']=='delay']),27)
        self.assertTrue(all(t['delay']['elapsed_s']==1 for t in targets))
        for t in targets:
            low,high=t['native_r_interval_ohm']
            self.assertTrue(EqualIn([low,high])([low]));self.assertTrue(EqualIn([low,high])([high]))
            self.assertAlmostEqual(330/low,t['target_q']*1.01)
            self.assertEqual(len(t['mvm']),30)
            self.assertEqual([len(t[p]['task_ids']) for p in ('pre','post','delayed')],[5,5,3])
    def test_overshoot_is_miss_not_crossing_hit(self):
        result,raw=self.run_fake(FakeConnector(overshoot=True))
        self.assertEqual(result['status'],'completed')
        self.assertTrue(all(t['status']=='overshoot' and not t['immediate_hit'] for c in result['cycles'] for t in c['targets']))
        self.assertFalse(any(r.get('phase')=='interval_mvm' for r in raw))
    def test_set_ramp_is_shared_271_per_cycle_not_per_target(self):
        result,raw=self.run_fake(FakeConnector(set_miss=True))
        self.assertEqual(result['status'],'completed')
        self.assertEqual(result['programming_pulse_attempts'],9*271+3)  # Later cycles already start below RESET endpoint.
        for c in result['cycles']:
            sets=[r for r in raw if r['event']=='request' and r['cycle_id']==c['cycle_id'] and r['phase']=='interval_set']
            self.assertEqual([r['step'] for r in sets],list(range(271)))
            self.assertTrue(all(t['status']=='set_ramp_exhausted' for t in c['targets']))
    def test_reset_miss_skips_set_but_retains_all_targets_and_final(self):
        result,raw=self.run_fake(FakeConnector(reset_miss=True))
        self.assertEqual(result['programming_pulse_attempts'],9*271)
        self.assertTrue(all(t['status']=='reset_unconfirmed' for c in result['cycles'] for t in c['targets']))
        self.assertTrue(all(len(c['final']['task_ids'])==3 for c in result['cycles']))
        self.assertFalse(any(r.get('phase')=='interval_set' for r in raw))
    def test_native_r_mismatch_stops_before_using_guard_or_interval(self):
        fake=FakeConnector(bad_r=True);result,raw=self.run_fake(fake)
        self.assertEqual(result['status'],'failed');self.assertEqual(len(fake.commands),1)
        self.assertEqual(raw[-1]['status'],'error');self.assertIn('differs',raw[-1]['error'])
        self.assertEqual(fake.closed,1)
    def test_lost_response_persists_partial_target_and_actual_attempt(self):
        result,raw=self.run_fake(FakeConnector(fail_set=True))
        self.assertEqual(result['status'],'failed');self.assertEqual(result['programming_pulse_attempts'],2)
        self.assertEqual(result['cycles'][0]['targets'][0]['status'],'failed')
        self.assertEqual(raw[-1]['status'],'error')
        self.assertTrue((self.output/'targets.jsonl').exists())
    def test_short_delay_does_not_pass_interval_screen(self):
        result,raw=self.run_fake(FakeConnector(),delay=False)
        targets=[t for c in result['cycles'] for t in c['targets']]
        self.assertTrue(all(t['all_readbacks_inside'] and t['mvm_screen']['passes'] for t in targets))
        self.assertFalse(any(t['interval_and_mvm_pass'] for t in targets))
    def test_dispatch_explicit_mode_and_conflicts(self):
        with patch.object(interval,'main',return_value=0) as run:
            for mode in ('--dry-run','--run-simulator','--run-hardware'):
                experiment.main([mode,'--test-cell-ranges']);run.assert_called_with([mode])
            run.reset_mock()
            for args in (['--test-cell-ranges'],['--run-hardware','--test-cell-ranges','--all'],['--dry-run','--test-cell-ranges','--map-cells']):
                with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):experiment.main(args)
            run.assert_not_called()
        with patch.object(interval,'native_connector') as connector,contextlib.redirect_stdout(io.StringIO()):
            interval.main([]);connector.assert_not_called()
    def test_unknown_operational_override_fails_before_connector(self):
        spec=json.loads((ROOT/'configs/cell_range_repeat.json').read_text())
        interval.validate(spec)
        spec['programming_guard']={'sign0_limit_a':999}
        path=self.output/'bad.json';path.write_text(json.dumps(spec))
        original=interval.local_path
        with patch.object(interval,'local_path',side_effect=lambda name:path if name==interval.CONFIG else original(name)), \
             patch.object(interval,'native_connector') as connector,patch.object(interval,'check_sources') as sources:
            with self.assertRaisesRegex(ValueError,'overrides'):interval.main(['--run-hardware'])
            connector.assert_not_called();sources.assert_not_called()


if __name__=='__main__':unittest.main()
