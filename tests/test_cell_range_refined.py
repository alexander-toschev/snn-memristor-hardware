"""Offline refined protocol tests: public fake Connector, native converters, no hardware."""
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
from tests.test_cell_range_repeat import FakeConnector
import cell_range_repeat as baseline
import cell_range_refined as refined
import experiment

ROOT=Path(__file__).resolve().parents[1]

class RefinedFake(FakeConnector):
    def __init__(self,overshoot_inner=False,outer_read=False,set_miss=False):
        super().__init__();self.overshoot_inner=overshoot_inner;self.outer_read=outer_read;self.set_miss=set_miss
    def impact(self,task):
        if task['mode_flag']==10:return super().impact(task)
        self.commands.append(copy.deepcopy(task));key=task['wl'],task['bl'];self.q.setdefault(key,.02)
        if task['vol']:
            if task['sign']==0:self.q[key]=.004
            else:
                step=round((task['vol']-.3)/.01)
                self.q[key]=.004 if self.set_miss else (.031+step*.001 if key[0]==5 else .04+step*.005)
                if self.overshoot_inner and key[0]!=5 and step==0:self.q[key]=.04035
        q=self.q[key]
        if self.outer_read and not task['vol'] and abs(q-.04)<1e-8:q=.04035
        adc=converters.convert_res_to_adc(11,3000,.3,14,5,10,330/q)
        return converters.convert_adc_to_res(11,3000,.3,14,5,10,adc),task['id'],adc

class RefinedTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory(dir=ROOT);self.addCleanup(tmp.cleanup);self.output=Path(tmp.name)
        self.spec=json.loads((ROOT/refined.CONFIG).read_text())
        self.plan=json.loads(refined.local_path(refined.BASE_CONFIG).read_text());self.plan.update(self.spec)
    def run_fake(self,fake):
        clock=[0.]
        def wait(s):clock[0]+=s
        worker=refined.RefinedRun(fake,converters,self.output,'offline',self.plan,'hardware',EqualIn,wait)
        with patch.object(baseline.time,'monotonic',side_effect=lambda:clock[0]),contextlib.redirect_stdout(io.StringIO()):result=worker.run()
        raw=[json.loads(x) for x in (self.output/'raw.jsonl').read_text().splitlines()]
        return result,raw
    def test_full_24_hits_outer_readback_and_counts(self):
        fake=RefinedFake();r,raw=self.run_fake(fake)
        self.assertEqual(r['status'],'completed');self.assertEqual(len(r['cycles']),9)
        ts=[t for c in r['cycles'] for t in c['targets']]
        self.assertEqual(len(ts),24);self.assertTrue(all(t['interval_and_mvm_pass'] for t in ts))
        self.assertEqual(r['impact_attempts'],1146);self.assertEqual(r['programming_pulse_attempts'],33)
        self.assertEqual(fake.closed,1);self.assertEqual(len([x for x in raw if x['event']=='delay']),24)
        for t in ts:
            self.assertAlmostEqual(330/t['native_r_interval_ohm'][0],t['target_q']*1.01)
            self.assertEqual(t['stop_q_relative_tolerance'],.01 if t['target_q'] in (.045,.05) else .0075)
            self.assertTrue(EqualIn(t['stop_native_r_interval_ohm'])([t['evidence_native_r_ohm']]))
        self.assertEqual(self.spec['maximum_impacts'],9*(1+271+5+271+3)+24*43)
    def test_inside_outer_but_above_inner_is_overshoot_not_hit(self):
        r,raw=self.run_fake(RefinedFake(overshoot_inner=True))
        for c in r['cycles'][:6]:
            t=c['targets'][0]
            self.assertEqual(t['status'],'overshoot');self.assertFalse(t['immediate_hit']);self.assertNotIn('mvm',t)
            self.assertLess(t['evidence_q'],.0404);self.assertGreater(t['evidence_q'],.0403)
            self.assertIn('stop_native_r_interval_ohm',t)
    def test_readbacks_qualify_against_outer_not_inner(self):
        r,raw=self.run_fake(RefinedFake(outer_read=True))
        for c in r['cycles'][:6]:
            t=c['targets'][0];self.assertTrue(t['interval_and_mvm_pass'])
            self.assertFalse(EqualIn(t['stop_native_r_interval_ohm'])([t['pre']['native_r_ohm'][0]]))
    def test_shared_271_set_budget_and_miss_fields(self):
        r,raw=self.run_fake(RefinedFake(set_miss=True))
        for c in r['cycles']:
            steps=[x['step'] for x in raw if x['event']=='request' and x['cycle_id']==c['cycle_id'] and x['phase']=='interval_set']
            self.assertEqual(steps,list(range(271)))
            self.assertTrue(all(t['status']=='set_ramp_exhausted' and 'stop_q_relative_tolerance' in t for t in c['targets']))
    def test_dispatch_default_no_io_and_conflicts(self):
        with patch.object(refined,'main',return_value=0) as run:
            for mode in ('--dry-run','--run-simulator','--run-hardware'):
                experiment.main([mode,'--test-cell-ranges-refined']);run.assert_called_with([mode])
            run.reset_mock()
            for args in (['--test-cell-ranges-refined'],['--dry-run','--test-cell-ranges-refined','--test-cell-ranges'],['--run-hardware','--test-cell-ranges-refined','--all']):
                with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):experiment.main(args)
            run.assert_not_called()
        with patch.object(refined,'native_connector') as connect,contextlib.redirect_stdout(io.StringIO()):
            refined.main([]);connect.assert_not_called()
    def test_config_cannot_override_guard_or_widen_qualification(self):
        refined.validate(self.spec)
        for key,value in [('programming_guard',{}),('q_relative_tolerance',.02),('stop_q_relative_tolerance',.01)]:
            spec=copy.deepcopy(self.spec);spec[key]=value
            with self.assertRaisesRegex(ValueError,'overrides'):refined.validate(spec)
    def test_all_stop_endpoints_inclusive_native_equal_in(self):
        for q in (.031,.032,.04,.045,.05):
            b=refined.target_fields(q)['stop_native_r_interval_ohm']
            self.assertTrue(EqualIn(b)([b[0]]));self.assertTrue(EqualIn(b)([b[1]]))

if __name__=='__main__':unittest.main()
