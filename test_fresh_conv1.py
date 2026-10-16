"""Offline state-machine tests; FakeConnector only, pinned native converters."""
import contextlib
import copy
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import fresh_conv1 as c
from test_cifar_feedback import FakeConnector, converters


class FreshTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.plan, cls.route, cls.fixture = c.load_inputs()

    @contextlib.contextmanager
    def run_with(self, fake=None, runner=c.FreshRun):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            fake = fake or FakeConnector(reset_step=100, set_step=1)
            run = runner(fake, converters, Path(tmp), 'EXPLICIT_OFFLINE_FAKE', self.plan, 'simulator',
                         clock=lambda:datetime(2026,10,16,17,tzinfo=timezone.utc), monotonic=lambda:0.)
            try: yield run, fake, Path(tmp)
            finally: run.log.close()

    def slot(self): return copy.deepcopy(self.route['filters'][0]['tiles'][0]['slots'][0])

    def test_route_all300_exact_weights_capacity_disjoint_and_counts(self):
        evidence = json.loads(c.EVIDENCE.read_text(encoding='utf8'))
        bycell = {tuple(x['cell']):x for x in evidence['cells']}
        original = json.loads((c.ROOT/'weights/candidate_cifar75_routing.json').read_text())
        for f, old in zip(self.route['filters'], original['filters']):
            originals = {s['tap_index']:s['trained_weight'] for t in old['tiles'] for s in c.slots(t)}
            self.assertEqual([s['tap_index'] for t in f['tiles'] for s in c.slots(t)], list(range(75)))
            for tile in f['tiles']:
                addresses = []
                for slot in c.slots(tile):
                    self.assertEqual(slot['trained_weight'], originals[slot['tap_index']])
                    self.assertLessEqual(len(slot['candidate_pairs']), 2)
                    for candidate in slot['candidate_pairs']:
                        addresses.extend(tuple(candidate[side]) for side in c.SIDES)
                        side = c.SIDES[0] if slot['trained_weight']>=0 else c.SIDES[1]
                        cell = bycell[tuple(candidate[side])]
                        self.assertTrue(cell['active_candidate_allowed'])
                        self.assertGreaterEqual(cell['screening_amplitude_v'], .25*abs(slot['trained_weight']))
                self.assertEqual(len(addresses), len(set(addresses)))
        bound = c.budget(c.tiles(self.route), self.fixture)
        self.assertEqual((bound['maximum_impacts'], bound['maximum_programming_pulses']), (1702644,91770))
        self.assertEqual((c.RESET_LEG_BOUND,c.QUIET_ATTEMPT_BOUND), (1120,2336))

    def test_confirmation_failure_continues_remaining_ramp_without_restart(self):
        class DisturbConfirmation(c.FreshRun):
            def leg_triple(self, cell, phase, **meta):
                if phase=='jit_reset_confirm' and meta['confirmation']==0:
                    self.connector.states[tuple(cell)]=10
                return super().leg_triple(cell, phase, **meta)
        with self.run_with(runner=DisturbConfirmation) as (run, fake, _):
            state=run.start_leg([0,0],{})
            self.assertTrue(run.advance_leg(state,{}))
            self.assertEqual(state['next_step'],1);self.assertEqual(state['confirmations'],2)
            self.assertEqual([t['vol'] for t in fake.calls if t['mode_flag']==7 and t['vol']], [.3])

    def test_all55_failed_confirmations_bounded(self):
        class AlwaysFail(c.FreshRun):
            def leg_triple(self, cell, phase, **meta):
                if phase=='jit_reset_confirm': self.connector.states[tuple(cell)]=10
                return super().leg_triple(cell, phase, **meta)
        with self.run_with(runner=AlwaysFail) as (run, _, _):
            state=run.start_leg([0,0],{})
            self.assertFalse(run.advance_leg(state,{}))
            self.assertEqual(state['next_step'],55);self.assertEqual(state['confirmations'],56)
            self.assertEqual(run.sequence,c.RESET_LEG_BOUND);self.assertEqual(run.pulses,55)

    def test_last_reset_level_can_confirm_without_extra_pulse(self):
        with self.run_with(FakeConnector(initial=57,reset_step=1)) as (run,_,_):
            state=run.start_leg([0,0],{})
            self.assertTrue(run.advance_leg(state,{}))
            self.assertEqual(state['next_step'],55);self.assertEqual(run.pulses,55)

    def test_second_leg_disturbs_first_joint_repair_uses_next_level(self):
        slot=self.slot();p=tuple(slot['positive_cell']);n=tuple(slot['negative_cell'])
        class Disturb(FakeConnector):
            def impact(self,task):
                event=(task['mode_flag']==7 and task['vol'] and task['sign']==0 and (task['wl'],task['bl'])==n)
                result=super().impact(task)
                if event:self.states[p]=10
                return result
        fake=Disturb(reset_step=10,set_step=1);fake.states[p]=30;fake.states[n]=10
        with self.run_with(fake) as (run, _, _):
            out=run.reset_pair(slot,{'pair_tap':slot['tap_index']})
            self.assertTrue(out['eligible']);self.assertEqual(len(out['joint_gates']),2)
            self.assertFalse(out['joint_gates'][0]['both_legs_quiet'])
            volts=[t['vol'] for t in fake.calls if t['mode_flag']==7 and t['vol'] and (t['wl'],t['bl'])==p]
            self.assertEqual(volts,[.3,.35,.4,.45])

    def test_repair_two_round_cap_preserves_failure(self):
        slot=self.slot();p=tuple(slot['positive_cell']);n=tuple(slot['negative_cell'])
        class CrossDisturb(FakeConnector):
            def impact(self,task):
                event=task['mode_flag']==7 and task['vol'] and task['sign']==0
                key=(task['wl'],task.get('bl'));result=super().impact(task)
                if event:self.states[n if key==p else p]=10
                return result
        fake=CrossDisturb(reset_step=100);fake.states[p]=10
        with self.run_with(fake) as (run, _, _):
            out=run.reset_pair(slot,{'pair_tap':slot['tap_index']})
            self.assertFalse(out['eligible']);self.assertEqual(len(out['joint_gates']),3)
            self.assertLessEqual(run.sequence,c.QUIET_ATTEMPT_BOUND)

    def test_opposite_residual_never_activates_unqualified_leg(self):
        slot=self.slot();slot['trained_weight']=.001
        fake=FakeConnector();fake.states[tuple(slot['positive_cell'])]=2
        with self.run_with(fake) as (run, _, _):
            out=run.program_pair(slot,{'pair_tap':slot['tap_index']})
            self.assertEqual(out['stop'],'initial_overshoot_no_SET')
            self.assertEqual(out['programming_pulse_attempts'],0)
            self.assertEqual(len(out['post_feedback']),5)
            self.assertFalse(any(t['mode_flag']==7 and t['vol'] for t in fake.calls))

    def test_fallback_quiet_failure_then_actual_full_measurement_tail(self):
        f=copy.deepcopy(self.route['filters'][0]);tile=copy.deepcopy(f['tiles'][0]);tile['slots']=tile['slots'][:1]
        slot=tile['slots'][0];self.assertEqual(len(slot['candidate_pairs']),2)
        fake=FakeConnector(reset_step=0,set_step=1)
        fake.states[tuple(slot['positive_cell'])]=10
        with self.run_with(fake) as (run, _, folder):
            out=run.collect_tile(f,tile,self.fixture)
            self.assertTrue(out['technical_completed']);self.assertEqual(len(out['reset_attempts']),2)
            self.assertFalse(out['reset_attempts'][0]['eligible']);self.assertTrue(out['reset_attempts'][1]['eligible'])
            self.assertEqual(out['effective_route'][0]['positive_cell'],slot['candidate_pairs'][1]['positive_cell'])
            self.assertEqual(len(out['partials']),600)
            self.assertTrue(all(p['trained_unit_contribution'] is not None for p in out['partials']))
            self.assertNotIn('full_ridge',out['matrix']['selected_model'])
            self.assertTrue((folder/'raw_parts.json').is_file())

    def test_exhausted_route_keeps_raw_and_never_substitutes_old_CIFAR(self):
        f=self.route['filters'][0];tile=copy.deepcopy(f['tiles'][0]);tile['slots']=tile['slots'][:1]
        with self.run_with(FakeConnector(initial=10,reset_step=0)) as (run, _, folder):
            with self.assertRaises(c.QuietRouteExhausted):run.collect_tile(f,tile,self.fixture)
            out=json.loads((folder/'tile.json').read_text())
            self.assertEqual(out['partials'],[]);self.assertFalse(out['technical_completed'])
            self.assertEqual(len(out['reset_attempts']),2);self.assertEqual(out['pairs'],[])

    def test_deadline_current_and_transport_stop_before_further_commands(self):
        with self.run_with() as (run,fake,_):
            run.clock=lambda:run.cutoff
            with self.assertRaises(c.DeadlineReached):run.read([0,0],1,'expired')
            self.assertEqual(fake.calls,[])
        with self.run_with() as (run,fake,_):
            run.last_mode7[(0,0)]=(1.,0)
            with self.assertRaises(c.CurrentGuardStop):run.impact(0,0,7,.3,'guard',sign=1)
            self.assertEqual(fake.calls,[])
        with self.run_with(FakeConnector(fail=True)) as (run,fake,_):
            with self.assertRaises(ConnectionError):run.reset_pair(self.slot(),{'pair_tap':0})
            self.assertEqual(len(fake.calls),1)

    def test_default_help_and_draft_native_flags_do_not_open_backend(self):
        with patch.object(c,'native_connector',side_effect=AssertionError('No I/O')):
            with contextlib.redirect_stdout(io.StringIO()):self.assertEqual(c.main([]),0)
            with patch.object(c,'load_inputs',return_value=({**self.plan,'review_pending':True},self.route,self.fixture)):
                for flag in ('--run-hardware','--run-simulator'):
                    with self.assertRaisesRegex(ValueError,'Draft protocol'):c.main([flag])

    def test_dispatcher_explicit_modes_rejects_extra_options_and_no_default_IO(self):
        import experiment
        with patch.object(c,'main',return_value=73) as entry:
            for mode in ('--dry-run','--run-simulator','--run-hardware'):
                self.assertEqual(experiment.main(['--test-cifar-fresh-conv1',mode]),73)
                entry.assert_called_with([mode])
            with contextlib.redirect_stdout(io.StringIO()):self.assertEqual(experiment.main([]),0)
            for extra in ([],['--dry-run','--all'],['--dry-run','--cifar-stage','pilot'],
                          ['--dry-run','--cifar-resume-session','old'],['--dry-run','--run-hardware'],
                          ['--dry-run','--map-cells']):
                with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                    experiment.main(['--test-cifar-fresh-conv1']+extra)

    def test_global_budget_rejects_before_connector(self):
        with self.run_with() as (run,fake,_):
            run.global_budget=dict(impacts=0,pulses=0,maximum_impacts=1,maximum_programming_pulses=1)
            run.read([0,0],1,'one')
            with self.assertRaisesRegex(ValueError,'Global impact'):run.read([0,0],1,'two')
            self.assertEqual(len(fake.calls),1);self.assertEqual(run.global_budget['impacts'],1)

    def test_launcher_transport_failure_closes_and_preserves_incomplete_manifest(self):
        class FailingPort(FakeConnector):
            cb_type='real'
            def __init__(self):super().__init__(fail=True);self.closed=0
            def open_port(self):return True
            def close_port(self):self.closed+=1;return True
        fake=FailingPort();plan={**self.plan,'review_pending':False}
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root=Path(tmp)
            def paths(name):
                if name==c.LOCK:return root/'test.lock'
                if name=='results/cifar_fresh_conv1':return root
                return c.ROOT/name
            with patch.object(c,'load_inputs',return_value=(plan,self.route,self.fixture)), \
                 patch.object(c,'check_sources',return_value=({},b'fake-test-only',{})), \
                 patch.object(c,'native_connector',return_value=fake),patch.object(c,'local_path',side_effect=paths), \
                 patch.object(c,'utc_now',return_value=datetime(2026,10,16,17,tzinfo=timezone.utc)), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(c.main(['--run-hardware']),2)
            self.assertEqual(fake.closed,1);self.assertEqual(len(fake.calls),1)
            manifests=list(root.glob('*/manifest.json'));self.assertEqual(len(manifests),1)
            manifest=json.loads(manifests[0].read_text());self.assertEqual(manifest['status'],'failed')
            self.assertEqual(manifest['total_impacts'],1);self.assertEqual(manifest['completed_microtiles'],0)
            self.assertFalse((manifests[0].parent/'summary.json').exists())

    def test_program_order_accounts_for_fallback_but_logical_taps_stay_fixed(self):
        a=self.slot();b=copy.deepcopy(a);a['tap_index']=0;b['tap_index']=1;b['trained_weight']=-.01
        a['candidate_pairs']=[{'positive_cell':[2,0],'negative_cell':[3,8]}]
        b['candidate_pairs']=[{'positive_cell':[2,1],'negative_cell':[3,9]},
                              {'positive_cell':[2,2],'negative_cell':[3,3]}]
        self.assertEqual([s['tap_index'] for s in c.programming_order([a,b])],[1,0])
        self.assertEqual([a['tap_index'],b['tap_index']],[0,1])


if __name__=='__main__':unittest.main()
