"""Explicit FakeConnector fixtures plus independent replay/mutation tests."""
import copy
from datetime import datetime,timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import audit_fresh_conv1 as a
import fresh_conv1 as c  # Fixture generation ONLY, never imported by the auditor.
from test_cifar_feedback import FakeConnector,converters
from test_cifar_memristor import small_inputs

ROOT=Path(__file__).resolve().parent
a.initialize(ROOT)

class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory(dir=Path(__file__).parent);cls.base=Path(cls.tmp.name)
        cls.plan,cls.route,cls.fullfixture=c.load_inputs();cls.f,cls.t,cls.fixture=small_inputs()
        for s in cls.t['slots']:s['candidate_pairs']=[{k:s[k] for k in a.SIDES}]
        cls.cache={}
        for source in ('simulator','hardware','quiet_failure'):
            folder=cls.base/source;folder.mkdir();kind='simulator' if source=='quiet_failure' else source
            fake=FakeConnector(initial=10,reset_step=0,set_step=1) if source=='quiet_failure' else FakeConnector(reset_step=100,set_step=1)
            fake.cb_type='real' if kind=='hardware' else 'simulator'
            run=c.FreshRun(fake,converters,folder,'EXPLICIT_AUDIT_FAKE',cls.plan,kind,
                clock=lambda:datetime(2026,10,16,17,tzinfo=timezone.utc),monotonic=lambda:0.)
            try:run.collect_tile(cls.f,cls.t,cls.fixture)
            except c.QuietRouteExhausted:
                if source!='quiet_failure':raise
            ck=a.A.CompactChecks();events,counts=a.A.read_raw(folder/'raw.jsonl',kind,ck,'EXPLICIT_AUDIT_FAKE')
            if ck.errors:raise AssertionError(ck.errors[:5])
            cls.cache[source]=(events,counts)
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def replay(self,source='hardware',cache=False):
        ck=a.A.CompactChecks();kind='simulator' if source=='quiet_failure' else source
        if cache:
            with patch.object(a.A,'read_raw',return_value=copy.deepcopy(self.cache[source])):
                result=a.audit_tile(self.base/source,self.f,self.t,self.fixture,self.plan,kind,'EXPLICIT_AUDIT_FAKE',ck)
        else:result=a.audit_tile(self.base/source,self.f,self.t,self.fixture,self.plan,kind,'EXPLICIT_AUDIT_FAKE',ck)
        return result,ck

    def test_hardware_gain025_full_tail_raw_replay(self):
        result,ck=self.replay();self.assertEqual(ck.errors,[]);self.assertEqual(len(result['partials']),600)
        self.assertEqual(result['calibration']['decoder_gain_v_per_weight'],.25)

    def test_SIMgain003_full_tail_raw_replay(self):
        result,ck=self.replay('simulator');self.assertEqual(ck.errors,[])
        self.assertEqual(result['calibration']['decoder_gain_v_per_weight'],.03)

    def test_quiet_exhaustion_no_fabricated_partials(self):
        result,ck=self.replay('quiet_failure');self.assertEqual(ck.errors,[])
        self.assertFalse(result['technical_completed']);self.assertEqual(result['partials'],[])

    def mutate_ledger(self,name,fn):
        original=a.A.ledger
        def changed(p):
            rows=original(p)
            if p.name==name:fn(rows)
            return rows
        with patch.object(a.A,'ledger',side_effect=changed):
            result,ck=self.replay(cache=True)
        self.assertTrue(ck.errors)

    def test_reset_index_restart_rejected(self):self.mutate_ledger('reset_confirmations.jsonl',lambda x:x[0].update(next_step=55))
    def test_forged_joint_gate_rejected(self):self.mutate_ledger('joint_gates.jsonl',lambda x:x[0].update(both_legs_quiet=False))
    def test_wrong_SET_leg_rejected(self):self.mutate_ledger('pair_programming.jsonl',lambda x:x[0].update(active_set_side='negative_cell'))
    def test_omitted_CIFAR_contribution_rejected(self):self.mutate_ledger('partials.jsonl',lambda x:x.pop())
    def test_replaced_actual_with_target_rejected(self):self.mutate_ledger('partials.jsonl',lambda x:x[0].update(trained_unit_contribution=123))

    def test_initial_opposite_residual_does_not_write_inactive_leg(self):
        with tempfile.TemporaryDirectory(dir=self.base) as tmp:
            folder=Path(tmp);slot=copy.deepcopy(self.t['slots'][0]);fake=FakeConnector(initial=0,set_step=1)
            fake.states[tuple(slot['positive_cell'])]=500
            fake.cb_type='real'
            run=c.FreshRun(fake,converters,folder,'EXPLICIT_AUDIT_FAKE',self.plan,'hardware',
                clock=lambda:datetime(2026,10,16,17,tzinfo=timezone.utc),monotonic=lambda:0.)
            expected=run.program_pair(slot,{'pair_tap':0});run.log.close()
            ck=a.A.CompactChecks();events,counts=a.A.read_raw(folder/'raw.jsonl','hardware',ck,'EXPLICIT_AUDIT_FAKE')
            r=a.A.Replay(events,ck);ledgers={'set_feedback':[],'pair_programming':[]}
            result=a.replay_program(r,slot,{'pair_tap':0},.25,ledgers);ck.equal(expected,result,'initial_overshoot_result')
            self.assertEqual(ck.errors,[]);self.assertEqual(result['stop'],'initial_overshoot_no_SET')
            self.assertEqual(counts['programming_pulse_attempts'],0);self.assertEqual(r.position,len(events))

    def test_actual_frozen_route_bounds_and_weights(self):
        evidence=json.loads((ROOT/'weights/fresh_conv1_capacity.json').read_text())
        original=json.loads((ROOT/'weights/candidate_cifar75_routing.json').read_text())
        ck=a.A.CompactChecks();r=a.validate_route(self.route,original,evidence,ck);self.assertEqual(ck.errors,[])
        self.assertEqual(r['trained_coefficients'],300);self.assertEqual(r['candidate_pair_attempts'],537)
        self.assertEqual(a.bounds(self.route,self.fullfixture)['maximum_impacts'],1702644)
        self.assertEqual(a.bounds(self.route,self.fullfixture)['maximum_programming_pulses'],91770)
        bad=copy.deepcopy(self.route);bad['filters'][0]['tiles'][0]['slots'][0]['trained_weight']+=.1
        ck=a.A.CompactChecks();a.validate_route(bad,original,evidence,ck);self.assertTrue(ck.errors)

    def test_order_includes_fallback_BL3_dependency(self):
        slots=[dict(tap_index=0,trained_weight=.1,candidate_pairs=[dict(positive_cell=[2,0],negative_cell=[3,8])]),
               dict(tap_index=1,trained_weight=-.1,candidate_pairs=[dict(positive_cell=[2,1],negative_cell=[3,2]),dict(positive_cell=[2,3],negative_cell=[3,3])])]
        self.assertEqual([x['tap_index'] for x in a.order(slots)],[1,0])

    def test_gain_not_fitted_perweight(self):
        cal=a.fixed_cal([.1,-.1],[[.020,-.010]]*5,.25)
        self.assertEqual(cal['effective_weights'],[.08,-.04]);self.assertGreater(cal['weight_rmse'],0)
        self.assertEqual(cal['decoder_gain_v_per_weight'],.25)

    def replay_reset_fixture(self,fake,runner=c.FreshRun):
        with tempfile.TemporaryDirectory(dir=self.base) as tmp:
            folder=Path(tmp);slot=copy.deepcopy(self.t['slots'][0]);meta={'pair_tap':slot['tap_index']}
            run=runner(fake,converters,folder,'EXPLICIT_AUDIT_FAKE',self.plan,'simulator',
                clock=lambda:datetime(2026,10,16,17,tzinfo=timezone.utc),monotonic=lambda:0.)
            expected=run.reset_pair(slot,meta);run.log.close()
            ck=a.A.CompactChecks();events,counts=a.A.read_raw(folder/'raw.jsonl','simulator',ck,'EXPLICIT_AUDIT_FAKE')
            r=a.A.Replay(events,ck);ledgers={k:[] for k in ('reset_confirmations','reset_feedback','joint_gates','reset_pairs')}
            actual=a.replay_reset_pair(r,slot,meta,ledgers);ck.equal(expected,actual,'reset/result')
            for name,rows in ledgers.items():ck.equal(a.A.ledger(folder/(name+'.jsonl')),rows,'reset/ledger/'+name)
            self.assertEqual(ck.errors,[]);self.assertEqual(r.position,len(events))
            return actual,ledgers

    def test_raw_failed_confirmation_consumes_remaining_reset_index(self):
        class DisturbConfirmation(c.FreshRun):
            def leg_triple(self,cell,phase,**meta):
                if phase=='jit_reset_confirm' and meta['confirmation']==0:self.connector.states[tuple(cell)]=10
                return super().leg_triple(cell,phase,**meta)
        out,ledgers=self.replay_reset_fixture(FakeConnector(reset_step=100),DisturbConfirmation)
        self.assertTrue(out['eligible'])
        self.assertEqual([x['step'] for x in ledgers['reset_feedback']],[0,0])
        self.assertEqual([x['eligible'] for x in ledgers['reset_confirmations']],[False,True,False,True])

    def test_raw_joint_repair_continues_step_without_restart(self):
        slot=self.t['slots'][0];p=tuple(slot['positive_cell']);n=tuple(slot['negative_cell'])
        class Disturb(FakeConnector):
            def impact(self,task):
                event=task['mode_flag']==7 and task['vol'] and task['sign']==0 and (task['wl'],task['bl'])==n
                result=super().impact(task)
                if event:self.states[p]=10
                return result
        fake=Disturb(reset_step=10);fake.states[p]=30;fake.states[n]=10
        out,ledgers=self.replay_reset_fixture(fake)
        self.assertTrue(out['eligible']);self.assertEqual(len(out['joint_gates']),2)
        self.assertEqual([x['step'] for x in ledgers['reset_feedback'] if x['cell']==list(p)],[0,1,2,3])

    def test_raw_joint_repair_cap_keeps_failure(self):
        slot=self.t['slots'][0];p=tuple(slot['positive_cell']);n=tuple(slot['negative_cell'])
        class Disturb(FakeConnector):
            def impact(self,task):
                event=task['mode_flag']==7 and task['vol'] and task['sign']==0
                key=(task['wl'],task.get('bl'));result=super().impact(task)
                if event:self.states[n if key==p else p]=10
                return result
        fake=Disturb(reset_step=100);fake.states[p]=10
        out,ledgers=self.replay_reset_fixture(fake)
        self.assertFalse(out['eligible']);self.assertEqual(len(out['joint_gates']),3)

if __name__=='__main__':unittest.main()
