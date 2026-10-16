"""Offline only: immutable prefix checks and actual FakeConnector microtile paths."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cifar_feedback_adaptive as a
from test_cifar_feedback import FakeConnector,converters
from test_cifar_memristor import small_inputs


class QuietSubsetFake(FakeConnector):
    def __init__(self,blocked):super().__init__(reset_step=100);self.blocked=set(map(tuple,blocked))
    def impact(self,task):
        mode=task['mode_flag'];wl=task['wl']
        bl=next((i for i,v in enumerate(task['vol']) if v),None) if mode==10 else task['bl']
        result=super().impact(task)
        if mode==10 and (wl,bl) in self.blocked:return 1.,0,100
        return result


def totals():return dict(impact_attempts=0,programming_pulse_attempts=0,programming_decisions=0,programming_skipped_current=0)


class AdaptiveTests(unittest.TestCase):
    def test_exact_bounds_and_pinned_parent_prefix(self):
        plan,routing,fixture=a.load_inputs();selected=a.tiles(routing)
        for subset,expected in ((selected,(868466,117180)),(selected[3:],(756104,101382))):
            budget=a.adaptive_budget(subset,fixture)
            self.assertEqual((budget['maximum_impacts'],budget['maximum_programming_pulses']),expected)
        for source in ('hardware','simulator'):
            binding=plan['parent_bindings'][source]
            chunks,meta=a.verify_parent(binding['session'],source,routing,plan)
            self.assertEqual([c['result']['tile_id'] for c in chunks],[0,1,2])
            self.assertEqual(sum(len(c['result']['effective_route']) for c in chunks),42)
            self.assertTrue(all(c['origin']['reused'] for c in chunks))
            self.assertEqual(meta['parent_total_impacts'],51157 if source=='hardware' else 300519)
            wrong=copy.deepcopy(plan);wrong['parent_bindings'][source]['audit_sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'hash mismatch'):a.verify_parent(binding['session'],source,routing,wrong)
            with self.assertRaises(ValueError):a.verify_parent(binding['session'],'simulator' if source=='hardware' else 'hardware',routing,plan)
        # A file tamper cannot be accepted by refreshing its own manifest, which is pinned.
        real=a.sha256
        with patch.object(a,'sha256',side_effect=lambda p:'0'*64 if str(p).endswith('tile.json') else real(p)):
            with self.assertRaisesRegex(ValueError,'immutable file mismatch'):
                a.verify_parent(plan['parent_bindings']['hardware']['session'],'hardware',routing,plan)

    def test_n1_noisy_measured_rank_does_not_identify_full_matrix(self):
        train=[dict(measured_mean_v=x,reference_v=y) for x,y in
               [([1.,2.],[1.,2.]),([1.1,1.9],[1.,2.]),([.9,2.2],[1.,2.]),([.01,-.01],[0.,0.])]]
        selection=[dict(reference_v=[1.,2.],measured_repeats_v=[[1.,2.],[1.01,1.99],[.99,2.01]])]
        result=a.select_adaptive_matrix(train,selection,.00031)
        self.assertEqual(result['training_design_rank'],2);self.assertEqual(result['reference_design_rank'],1)
        self.assertFalse(result['full_matrix_identifiable_from_reference'])
        self.assertFalse(any(c['eligible'] for c in result['candidates'] if c['name'].startswith('full_')))
        self.assertIn(result['selected_model'],('identity','diagonal'))

    def test_real_fake_13_plus_1_keeps_original_pool_and_raw(self):
        plan,routing,fixture=a.load_inputs();f,parent=a.tiles(routing)[0]
        pool=a.candidate_pools(parent);fake=QuietSubsetFake(pool['positive_cell'][-4:])
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp,contextlib.redirect_stdout(io.StringIO()):
            output=Path(tmp);chunks=[];counter=totals()
            a.collect_parent(fake,converters,output,'synthetic',plan,'simulator',f,parent,fixture,counter,
                             a.adaptive_budget([(f,parent)],fixture),chunks)
            self.assertEqual([len(c['result']['effective_route']) for c in chunks],[13,1])
            self.assertEqual([len(c['result']['reset_candidates']) for c in chunks],[34,34])
            self.assertEqual([c['result']['tile_id'] for c in chunks],[0,1])
            self.assertEqual([s['tap_index'] for c in chunks for s in c['result']['effective_route']],list(range(14)))
            self.assertEqual([len(c['result']['partials']) for c in chunks],[600,600])
            self.assertFalse(chunks[1]['result']['matrix']['full_matrix_identifiable_from_reference'])
            self.assertEqual(counter['impact_attempts'],len(fake.calls))
            self.assertTrue(all(t['mode_flag'] in (7,10) for t in fake.calls))
            self.assertTrue(all(t['id']==0 for t in fake.calls if t['mode_flag']==10))
            for c in chunks:
                folder=output/Path(c['origin']['path']).parent
                self.assertTrue((folder/'raw_parts.json').exists())
                self.assertEqual(c['result']['calibration']['decoder_gain_v_per_weight'],.03)

    def test_zero_capacity_stops_and_preserves_failed_attempts(self):
        f,parent,fixture=small_inputs();pool=a.candidate_pools(parent)
        fake=QuietSubsetFake(pool['positive_cell']);counter=totals();chunks=[]
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            with self.assertRaisesRegex(ValueError,'zero_quiet'):
                a.collect_parent(fake,converters,Path(tmp),'synthetic',a.SPEC,'simulator',f,parent,fixture,counter,
                                 a.adaptive_budget([(f,parent)],fixture),chunks)
            self.assertEqual(chunks,[]);self.assertEqual(counter['impact_attempts'],len(fake.calls))
            saved=json.loads((Path(tmp)/'tiles/f0_p0_m0/tile.json').read_text())
            self.assertFalse(saved['technical_completed']);self.assertFalse((Path(tmp)/'tiles/f0_p0_m1').exists())

    def test_two_microtile_cap_does_not_sweep_again(self):
        f,parent,fixture=small_inputs();s=copy.deepcopy(parent['slots'][0]);s.update(tap_index=2,slot=2,trained_weight=.05,positive_cell=[0,2],negative_cell=[1,2]);parent['slots'].append(s)
        pool=a.candidate_pools(parent);fake=QuietSubsetFake(pool['positive_cell'][1:]);chunks=[];counter=totals()
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp,contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError,'Two-microtile limit'):
                a.collect_parent(fake,converters,Path(tmp),'synthetic',a.SPEC,'simulator',f,parent,fixture,counter,
                                 a.adaptive_budget([(f,parent)],fixture),chunks)
            self.assertEqual([len(c['result']['effective_route']) for c in chunks],[1,1])
            self.assertFalse((Path(tmp)/'tiles/f0_p0_m2').exists())

    def test_exact_dynamic_coverage_and_no_duplicate_or_missing_timestep(self):
        _,routing,fixture=a.load_inputs();chunks=[]
        for f,parent in a.tiles(routing):
            active=a.slots(parent);groups=[active[:-1],active[-1:]] if len(active)>1 else [active]
            for micro,group in enumerate(groups):
                result=dict(filter_id=f['filter_id'],tile_id=2*parent['tile_id']+micro,technical_completed=True,
                    effective_route=group,calibration=dict(effective_weights=[0.]*len(group)),
                    partials=[dict(case=c,repeat=r,t=t,trained_unit_contribution=0.) for c in range(10) for r in range(5) for t in range(12)])
                chunks.append(dict(filter_id=f['filter_id'],parent_tile_id=parent['tile_id'],microtile_id=micro,result=result))
        out=a.aggregate_microtiles(chunks,routing,fixture)
        self.assertEqual(len(out['trajectories']),200)
        self.assertEqual(sum(len(t['actual']) for t in out['trajectories']),2400)
        with self.assertRaisesRegex(ValueError,'Duplicate'):a.aggregate_microtiles(chunks+[chunks[0]],routing,fixture)
        with self.assertRaisesRegex(ValueError,'Missing'):a.aggregate_microtiles(chunks[:-1],routing,fixture)
        chunks[-1]['result']['partials'].pop()
        with self.assertRaisesRegex(ValueError,'Incomplete'):a.aggregate_microtiles(chunks,routing,fixture)

    def test_dispatch_no_default_io_and_explicit_resume(self):
        import experiment
        with patch.object(a,'main',return_value=0) as launch,contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            experiment.main([]);launch.assert_not_called()
            experiment.main(['--cifar-feedback-adaptive','--dry-run','--cifar-resume-session','parent'])
            launch.assert_called_once_with(['--dry-run','--resume-session','parent'])
            for args in (['--cifar-feedback-adaptive'],['--cifar-resume-session','parent','--dry-run'],
                         ['--cifar-feedback-adaptive','--dry-run','--cifar-stage','remaining'],
                         ['--cifar-feedback-adaptive','--run-hardware','--run-simulator']):
                with self.assertRaises(SystemExit):experiment.main(args)
        with patch.object(a,'native_connector',side_effect=AssertionError('no backend')),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(a.main([]),0)


if __name__=='__main__':unittest.main()
