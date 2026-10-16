"""Offline native-converter/FakeConnector tests. No native backend is opened."""
import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cifar_feedback as c
from manager.service import converters
from native_conversions import readback, voltage
from test_cifar_memristor import small_inputs


class FakeConnector:
    cb_type='simulator'
    def __init__(self, initial=0, reset_step=1, set_step=100, fail=False):
        self.initial=initial;self.reset_step=reset_step;self.set_step=set_step
        self.states={};self.calls=[];self.fail=fail
    def impact(self,task):
        self.calls.append(copy.deepcopy(task))
        if self.fail:raise ConnectionError('lost native response')
        assert task['mode_flag'] in (7,10)
        if task['mode_flag']==7:
            key=task['wl'],task['bl']
            if task['vol']:
                old=self.states.get(key,self.initial)
                self.states[key]=max(0,old-self.reset_step) if task['sign']==0 else old+self.set_step
            adc=1000
            return readback(adc,c.EXPECTED_BOARD,converters,read_voltage=.3)['resistance_ohm'],task['id'],adc
        assert task['id']==0 and len(task['vol'])==32 and sum(bool(v) for v in task['vol'])<=1
        assert all(v in (0.,.1) for v in task['vol'])
        bl=next((i for i,v in enumerate(task['vol']) if v),None)
        adc=0 if bl is None else self.states.get((task['wl'],bl),self.initial)
        task['vol'][:]=[999]*32
        return 1.,0,adc


class FeedbackTests(unittest.TestCase):
    @contextlib.contextmanager
    def workspace(self):
        self.open_runs=[]
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            try:yield tmp
            finally:
                for run in self.open_runs:run.log.close()

    def runner(self,tmp,fake):
        run=c.FeedbackRun(fake,converters,Path(tmp),'fake-feedback',c.SPEC,'simulator')
        self.open_runs.append(run)
        return run

    def test_exact_budgets_unique_dac_and_pool(self):
        _,routing,fixture=c.load_inputs();all_tiles=c.tiles(routing)
        self.assertEqual(c.counts(all_tiles,fixture)['maximum_impacts'],580210)
        self.assertEqual(c.counts(all_tiles[:1],fixture)['maximum_impacts'],25612)
        self.assertEqual(c.counts(all_tiles[1:],fixture)['maximum_impacts'],554598)
        self.assertEqual(c.counts(all_tiles,fixture)['maximum_programming_pulses'],74940)
        for key,n in [('reset_ramp_v',55),('set_ramp_v',109)]:
            self.assertEqual(len({voltage(v,c.EXPECTED_BOARD,converters)['dac_code'] for v in c.SPEC[key]}),n)
        pool=c.candidate_pools(all_tiles[0][1]);self.assertEqual([len(v) for v in pool.values()],[17,17])
        union={tuple(cell) for _,tile in all_tiles for cells in c.candidate_pools(tile).values() for cell in cells}
        self.assertEqual(len(union),118)

    def test_last_reset_step_quiet_is_eligible_despite_q(self):
        with self.workspace() as tmp:
            run=self.runner(tmp,FakeConnector(initial=57));out=run.reset_candidate([0,0],{})
            self.assertTrue(out['eligible']);self.assertFalse(out['native_q_diagnostic_crossed'])
            self.assertEqual(out['reset_stop'],'mode10_quiet_crossed')
            self.assertEqual(out['programming_pulse_attempts'],55)
            self.assertEqual(out['impact_attempts'],244)

    def test_reset_exhausted_retains_nonquiet_evidence(self):
        with self.workspace() as tmp:
            run=self.runner(tmp,FakeConnector(initial=100,reset_step=0));out=run.reset_candidate([0,0],{})
            self.assertFalse(out['eligible']);self.assertEqual(out['reset_stop'],'reset_ramp_exhausted')
            self.assertEqual(len(out['mvm_confirmation']),5)

    def test_residual_selects_negative_leg_and_stops_on_overshoot(self):
        with self.workspace() as tmp:
            fake=FakeConnector(set_step=200);fake.states[(0,0)]=100
            run=self.runner(tmp,fake)
            slot=dict(tap_index=0,trained_weight=.1,positive_cell=[0,0],negative_cell=[1,0])
            out=run.program_pair(slot,dict(filter_id=0,tile_id=0,tap=0))
            self.assertEqual(out['active_set_side'],'negative_cell')
            self.assertEqual(out['stop'],'feedback_overshoot')
            writes=[t for t in fake.calls if t['mode_flag']==7 and t['vol']]
            self.assertEqual(len(writes),1);self.assertEqual((writes[0]['wl'],writes[0]['sign']),(1,1))
            self.assertEqual([t['mode_flag'] for t in fake.calls[:2]],[7,7])
            self.assertEqual(len(out['post_feedback']),5)

    def test_current_guard_skip_has_no_pulse_or_mode10_fabrication(self):
        with self.workspace() as tmp:
            run=self.runner(tmp,FakeConnector())
            original=run.read
            def read(*args,**kwargs):
                rows=original(*args,**kwargs)
                if args[2]=='pair_guard_refresh':run.last_mode7[tuple(args[0])]=(100.,rows[-1]['task_id'])
                return rows
            run.read=read
            out=run.program_pair(dict(tap_index=0,trained_weight=.1,positive_cell=[0,0],negative_cell=[1,0]),{})
            self.assertEqual(out['stop'],'current_guard');self.assertEqual(out['programming_pulse_attempts'],0)
            self.assertEqual(out['programming_skipped_current'],1)

    def test_fidelity_noise_signs_quantization_and_invalid(self):
        good=c.fidelity_report([.1,-.1],[[.1,-.1]]*5,1.,.00031)
        self.assertTrue(good['physical_fidelity_pass'])
        noisy=c.fidelity_report([.1],[[v] for v in (.05,.15,.05,.15,.1)],1.,.00031)
        self.assertAlmostEqual(noisy['raw_target_nrmse'],0)
        self.assertFalse(noisy['physical_fidelity_pass']);self.assertGreater(noisy['uncertainty_nrms'],.2)
        zero=c.fidelity_report([.1],[[0.]]*5,1.,.00031)
        self.assertEqual(zero['measured_sign_unresolved_significant_slots'],[0])
        wrong=c.fidelity_report([.1],[[-.1]]*5,1.,.00031)
        self.assertEqual(wrong['resolved_sign_error_slots'],[0])
        tiny=c.fidelity_report([.0001],[[0.]]*5,1.,.00031)
        self.assertEqual(tiny['raw_target_nrmse'],1.);self.assertFalse(tiny['physical_fidelity_pass'])
        for weights,gain in [([float('nan')],1.),([.1],0.)]:
            with self.assertRaises(ValueError):c.fidelity_report(weights,[[.1]]*5,gain,.00031)

    def test_failed_transport_raw_and_attempt_retained(self):
        with self.workspace() as tmp:
            run=self.runner(tmp,FakeConnector(fail=True))
            with self.assertRaises(ConnectionError):run.reset_candidate([0,0],{})
            raw=[json.loads(s) for s in (Path(tmp)/'raw.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual([r['event'] for r in raw],['request','result'])
            self.assertEqual(raw[-1]['status'],'error')
            saved=json.loads((Path(tmp)/'reset_candidates.jsonl').read_text(encoding='utf-8'))
            self.assertEqual(saved['impact_attempts'],1)

    def test_small_full_chain_freezes_before_cifar_and_keeps_failed_fidelity(self):
        f,tile,fixture=small_inputs()
        with self.workspace() as tmp:
            run=self.runner(tmp,FakeConnector(set_step=10));out=run.collect_tile(f,tile,fixture)
            self.assertTrue(out['technical_completed']);self.assertEqual(len(out['partials']),600)
            self.assertEqual(len(out['reset_candidates']),10)
            self.assertEqual(len(out['pairs']),2)
            raw=[json.loads(s) for s in (Path(tmp)/'raw.jsonl').read_text(encoding='utf-8').splitlines()]
            freeze=next(i for i,r in enumerate(raw) if r['event']=='calibration_frozen')
            first=next(i for i,r in enumerate(raw) if r.get('phase')=='cifar')
            self.assertLess(freeze,first)
            self.assertTrue(all(r['command_si']['mode_flag'] in (7,10) for r in raw if r['event']=='request'))

    def test_inadequate_pool_preserved_no_set(self):
        f,tile,fixture=small_inputs()
        with self.workspace() as tmp:
            fake=FakeConnector(initial=100,reset_step=0);run=self.runner(tmp,fake)
            with self.assertRaisesRegex(ValueError,'insufficient_quiet'):run.collect_tile(f,tile,fixture)
            saved=json.loads((Path(tmp)/'tile.json').read_text(encoding='utf-8'))
            self.assertEqual(len(saved['reset_candidates']),10)
            self.assertFalse(any(t['mode_flag']==7 and t['vol'] and t['sign']==1 for t in fake.calls))

    def test_dispatch_default_and_explicit_mode(self):
        import experiment
        with patch.object(c,'main',return_value=0) as launch,contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(experiment.main([]),0);launch.assert_not_called()
            experiment.main(['--cifar-feedback','--dry-run'])
            launch.assert_called_once_with(['--dry-run','--stage','pilot'])
            for args in (['--cifar-feedback'],['--cifar-feedback','--cifar-conv1','--dry-run'],['--cifar-feedback','--dry-run','--phase','phase1']):
                with self.assertRaises(SystemExit):experiment.main(args)

    def test_full_pilot_resume_and_refreshed_hash_missing_partial(self):
        _,routing,fixture=c.load_inputs();f,tile=c.tiles(routing)[0]
        with self.workspace() as tmp:
            directory=Path(tmp);folder=directory/'tiles/f0_t0';folder.mkdir(parents=True)
            run=c.FeedbackRun(FakeConnector(set_step=1),converters,folder,'pilot',c.SPEC,'simulator')
            self.open_runs.append(run)
            result=run.collect_tile(f,tile,fixture)
            self.assertTrue(result['continuation_eligible'])
            self.assertEqual(len(result['partials']),600);self.assertEqual(len(result['reset_candidates']),34)
            for name in ('summary.json','protocol.json'):(directory/name).write_text('{}',encoding='utf-8')
            (directory/'board.ini').write_text('fixture',encoding='utf-8')
            manifest=dict(session_id='pilot',stage='pilot',status='completed',source='simulator',source_sha256={},
                result_sha256={p.relative_to(directory).as_posix():c.sha256(p) for p in directory.rglob('*') if p.is_file()})
            def save():(directory/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
            save();relative=directory.relative_to(c.ROOT).as_posix()
            imported,parent=c.verify_pilot(relative,'simulator',{},fixture,routing)
            self.assertEqual(len(imported['effective_route']),14);self.assertTrue(parent['not_new_measurement'])
            with self.assertRaises(ValueError):c.verify_pilot(relative,'hardware',{},fixture,routing)
            with self.assertRaises(ValueError):c.verify_pilot(relative,'simulator',{'changed':'source'},fixture,routing)
            path=folder/'tile.json';changed=json.loads(path.read_text(encoding='utf-8'));changed['partials'].pop()
            path.write_text(json.dumps(changed),encoding='utf-8')
            manifest['result_sha256']['tiles/f0_t0/tile.json']=c.sha256(path);save()
            with self.assertRaises(ValueError):c.verify_pilot(relative,'simulator',{},fixture,routing)


if __name__=='__main__':unittest.main()
