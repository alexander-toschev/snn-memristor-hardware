"""No-BL8 source/route/front-door regressions; FakeConnector only."""
import contextlib
import copy
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import fresh_conv1_no_bl8 as c
from test_cifar_feedback import FakeConnector, converters


class NoBL8Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.plan, cls.route, cls.fixture = c.load_inputs()

    @contextlib.contextmanager
    def runner(self, fake=None):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            fake = fake or FakeConnector(reset_step=100, set_step=1)
            run = c.NoBL8Run(fake, converters, Path(tmp), 'EXPLICIT_OFFLINE_FAKE_NO_BL8', self.plan,
                'simulator', clock=lambda:datetime(2026,10,16,17,tzinfo=timezone.utc), monotonic=lambda:0.)
            try: yield run, fake, Path(tmp)
            finally: run.log.close()

    def test_all300_weights_and_logical_tap8_preserved_no_physical_bl8(self):
        original=json.loads((c.ROOT/'weights/candidate_cifar75_routing.json').read_text())
        evidence=json.loads(c.EVIDENCE.read_text())
        cells={tuple(x['cell']):x for x in evidence['cells']}
        for f, old in zip(self.route['filters'], original['filters']):
            weights={s['tap_index']:s['trained_weight'] for t in old['tiles'] for s in c.slots(t)}
            self.assertEqual([s['tap_index'] for t in f['tiles'] for s in c.slots(t)],list(range(75)))
            for tile in f['tiles']:
                used=[];self.assertLessEqual(len(c.slots(tile)),7)
                for s in c.slots(tile):
                    self.assertEqual(s['trained_weight'],weights[s['tap_index']])
                    for pair in s['candidate_pairs']:
                        for side in c.SIDES:
                            self.assertNotEqual(pair[side][1],8);used.append(tuple(pair[side]))
                        active=pair[c.SIDES[0] if s['trained_weight']>=0 else c.SIDES[1]]
                        self.assertTrue(cells[tuple(active)]['active_candidate_allowed'])
                        self.assertGreaterEqual(cells[tuple(active)]['screening_amplitude_v'],.25*abs(s['trained_weight']))
                self.assertEqual(len(used),len(set(used)))
        b=c.budget(c.tiles(self.route),self.fixture)
        self.assertEqual((b['tiles'],b['trained_coefficients'],b['candidate_pair_attempts'],b['maximum_impacts'],b['maximum_programming_pulses']),
                         (44,300,530,1686292,91000))
        self.assertEqual(len(self.route['primary_distinct_addresses']),108)
        self.assertEqual(self.plan['target_gain_v_per_weight'],{'hardware':.25,'simulator':.03})

    def test_old_loader_historical_verification_is_mandatory(self):
        with patch.object(c.inherited,'load_inputs',side_effect=ValueError('Historical capacity source changed')):
            with self.assertRaisesRegex(ValueError,'Historical capacity'):c.load_inputs()

    def test_changed_route_or_config_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            path=Path(tmp)/'changed.json'
            route=copy.deepcopy(self.route)
            route['filters'][0]['tiles'][0]['slots'][0]['candidate_pairs'][0]['positive_cell'][1]=8
            path.write_text(json.dumps(route))
            with patch.object(c,'ROUTE',path),self.assertRaisesRegex(ValueError,'Frozen no-BL8 route'):c.load_inputs()
            path.write_text(json.dumps({**self.plan,'maximum_impacts':1686293}))
            with patch.object(c,'CONFIG',path),self.assertRaisesRegex(ValueError,'Protocol differs'):c.load_inputs()

    def test_excluded_address_rejected_before_any_fake_command(self):
        with self.runner() as (run,fake,_):
            with self.assertRaisesRegex(ValueError,'BL8'):run.impact(0,8,7,0,'bad-read')
            with self.assertRaisesRegex(ValueError,'BL8'):run.impact(0,8,7,.3,'bad-write')
            vector=[0.]*32;vector[8]=.1
            with self.assertRaisesRegex(ValueError,'BL8'):run.impact(0,0,10,vector,'bad-vector')
            self.assertEqual(fake.calls,[]);self.assertEqual(run.sequence,0)

    def test_fake_full_tile_new600_partials_fixed_gain_and_frozen_matrix(self):
        f=copy.deepcopy(self.route['filters'][0]);tile=f['tiles'][0]
        with self.runner() as (run,fake,folder):
            result=run.collect_tile(f,tile,self.fixture)
            self.assertTrue(result['technical_completed'])
            self.assertEqual(len(result['effective_route']),7)
            self.assertEqual(len(result['partials']),600)
            self.assertTrue(all(p['trained_unit_contribution'] is not None for p in result['partials']))
            self.assertTrue((folder/'raw_parts.json').is_file())
            for task in fake.calls:
                if task['mode_flag']==7:self.assertNotEqual(task['bl'],8)
                else:self.assertEqual(task['vol'][8],0.)
            self.assertTrue((folder/'frozen_matrix.json').is_file())
            self.assertLessEqual(run.sequence,c.budget([(f,tile)],self.fixture)['maximum_impacts'])

    def test_default_help_and_pending_review_cannot_open_backend(self):
        with patch.object(c,'native_connector',side_effect=AssertionError('No backend')):
            with contextlib.redirect_stdout(io.StringIO()):self.assertEqual(c.main([]),0)
            with patch.object(c,'load_inputs',return_value=({**self.plan,'review_pending':True},self.route,self.fixture)):
                for flag in ('--run-hardware','--run-simulator'):
                    with self.assertRaisesRegex(ValueError,'Draft protocol'):c.main([flag])

    def test_dispatcher_new_and_historical_are_distinct_and_explicit(self):
        import experiment
        with patch.object(c,'main',return_value=17) as new,patch.object(c.inherited,'main',return_value=19) as old:
            for mode in ('--dry-run','--run-simulator','--run-hardware'):
                self.assertEqual(experiment.main(['--test-cifar-fresh-conv1-no-bl8',mode]),17);new.assert_called_with([mode])
                self.assertEqual(experiment.main(['--test-cifar-fresh-conv1',mode]),19);old.assert_called_with([mode])
            for extra in ([],['--dry-run','--all'],['--dry-run','--cifar-stage','pilot'],['--dry-run','--cifar-resume-session','old'],
                          ['--dry-run','--test-cifar-fresh-conv1'],['--dry-run','--run-hardware']):
                with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                    experiment.main(['--test-cifar-fresh-conv1-no-bl8']+extra)

    def test_expired_cutoff_before_backend(self):
        with patch.object(c,'load_inputs',return_value=({**self.plan,'review_pending':False},self.route,self.fixture)), \
             patch.object(c,'check_sources',return_value=({},b'fake-test-only',{})), \
             patch.object(c,'utc_now',return_value=datetime(2026,10,16,19,tzinfo=timezone.utc)), \
             patch.object(c,'native_connector',side_effect=AssertionError('No backend')):
            with self.assertRaises(c.DeadlineReached):c.main(['--run-hardware'])

    def test_transport_failure_closes_and_keeps_incomplete_raw(self):
        class Port(FakeConnector):
            cb_type='real'
            def __init__(self):super().__init__(fail=True);self.closed=0
            def open_port(self):return True
            def close_port(self):self.closed+=1;return True
        fake=Port()
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root=Path(tmp)
            def paths(name):
                if name==c.LOCK:return root/'test.lock'
                if name=='results/cifar_fresh_conv1_no_bl8':return root
                return c.ROOT/name
            with patch.object(c,'load_inputs',return_value=({**self.plan,'review_pending':False},self.route,self.fixture)), \
                 patch.object(c,'check_sources',return_value=({},b'fake-test-only',{})), \
                 patch.object(c,'native_connector',return_value=fake),patch.object(c,'local_path',side_effect=paths), \
                 patch.object(c,'utc_now',return_value=datetime(2026,10,16,17,tzinfo=timezone.utc)),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(c.main(['--run-hardware']),2)
            self.assertEqual(fake.closed,1);self.assertEqual(len(fake.calls),1)
            manifest=json.loads(next(root.glob('*/manifest.json')).read_text())
            self.assertEqual(manifest['status'],'failed');self.assertEqual(manifest['total_impacts'],1)
            self.assertEqual(manifest['completed_microtiles'],0)
            self.assertEqual(list(root.glob('*/summary.json')),[])


if __name__=='__main__':unittest.main()
