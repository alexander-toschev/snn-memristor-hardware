"""Offline numerical and FakeConnector checks. No real backend is opened."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from manager.service import converters
import experiment
import mapped_sequential_mvm as run
import mapped_sequential_analysis as analysis

ROOT=Path(__file__).resolve().parent


class FakeConnector:
    cb_type='real'
    def __init__(self,fail_at=None,high_basis=False):
        self.commands=[];self.closed=0;self.fail_at=fail_at;self.high_basis=high_basis
    def open_port(self):return True
    def close_port(self):self.closed+=1;return True
    def impact(self,task):
        self.commands.append(copy.deepcopy(task));assert task['mode_flag'] in (9,10)
        if task['mode_flag']==9:
            adc=converters.convert_res_to_adc(11,3000,.1,14,5,10,10000)
            return converters.convert_adc_to_res(11,3000,.3,14,5,10,adc),0,adc
        assert set(task)=={'mode_flag','vol','wl','id'} and task['id']==0
        active=[b for b,v in enumerate(task['vol']) if v];assert len(active)<=1
        adc=10+sum(100+(task['wl']*17+b*31)%700 for b in active)
        if self.high_basis and active:adc=8000
        if len(self.commands)==self.fail_at:adc=14000
        task['vol'][:]=[converters.convert_volt_to_dac(12,5,v) for v in task['vol']]
        return -111.,0,adc


def anchors():
    return [dict(phase=phase,wl=wl,mask=mask,repeat=r,
        raw_adc=10+(100+wl*17+mask[0]*3 if mask else 0))
        for phase in ('basis','selection','final') for r in range(5)
        for mask in [[]]+[[b] for b in range(32)] for wl in range(8)]


class SequentialTests(unittest.TestCase):
    def test_budget_routing_and_empty_signal_is_distinct(self):
        entries=run.schedule(run.SPEC);self.assertEqual(len(entries),10706)
        self.assertEqual([e['schedule_id'] for e in entries],list(range(1,10707)))
        self.assertEqual(sum(len(p['positive_bl']) for p in run.SPEC['pairs']),47)
        for p in run.SPEC['pairs']:
            for side in ('positive_bl','negative_bl'):self.assertEqual(len(p[side]),len(set(p[side])))
        self.assertTrue(all(len(e.get('mask',[]))<=1 for e in entries))
        for pair_id,pair in enumerate(run.SPEC['pairs']):
            for name in run.SPEC['patterns']:
                masks=analysis.pattern(name,len(pair['positive_bl']))
                self.assertEqual(sorted(j for mask in masks for j in mask),list(range(len(pair['positive_bl']))))
                rows=[e for e in entries if e['phase']=='evaluation' and e['pair_id']==pair_id and e['pattern']==name and e['repeat']==0]
                for wl,columns in zip(pair['wl'],(pair['positive_bl'],pair['negative_bl'])):
                    for e in rows:
                        if e['wl']==wl and e['role']=='partial':self.assertEqual(e['mask'],[columns[e['logical_index']]])
                    for t,active in enumerate(masks):
                        group=[e for e in rows if e['wl']==wl and e['t']==t]
                        self.assertEqual(group[0]['role'],'zero_before');self.assertEqual(group[-1]['role'],'zero_after')
                        if not active:self.assertEqual([e['role'] for e in group],['zero_before','zero_signal','zero_after'])
                        self.assertEqual(len(group),len({e['schedule_id'] for e in group}))

    def test_exact_config_and_no_auto_routing(self):
        altered=copy.deepcopy(run.SPEC);altered['nominal_v']=.2
        with self.assertRaises(ValueError):run.validate(altered)
        with patch.object(run,'main',return_value=0) as called,contextlib.redirect_stdout(io.StringIO()):
            experiment.main([]);called.assert_not_called()
            experiment.main(['--test-mapped-sequential-mvm','--dry-run']);called.assert_called_once_with(['--dry-run'])
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            experiment.main(['--test-mapped-sequential-mvm','--run-hardware','--all'])
        with patch.object(run,'check_sources') as preflight,contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(run.main([]),0);preflight.assert_not_called()

    def test_center_each_partial_and_keep_empty_response(self):
        self.assertEqual(analysis.centered_block(10,[110,210],14),296.)
        self.assertEqual(analysis.centered_block(10,[17],14),5.)
        self.assertNotEqual(analysis.centered_block(10,[17],14),0.)

    def test_shared_baseline_noise_scales_with_number_of_partials(self):
        blocks=[dict(signal_adc=[100.,200.,300.],zero_before_adc=z,zero_after_adc=z) for z in (8.,9.,10.,11.,12.)]
        bound=analysis.block_noise_bound(blocks)
        self.assertEqual(bound['bracket_coefficient'],3)
        self.assertAlmostEqual(bound['conservative_sd_bound_adc'],3*(2.5**.5))
        self.assertAlmostEqual(analysis.stats([analysis.centered_block(b['zero_before_adc'],b['signal_adc'],b['zero_after_adc']) for b in blocks])['sd'],bound['conservative_sd_bound_adc'])

    def test_reference_never_uses_final_or_evaluation(self):
        records=anchors();before=analysis.freeze_reference(records,run.SPEC)
        for r in records:
            if r['phase']=='final':r['raw_adc']=12345
        records.append(dict(phase='evaluation',wl=0,mask=[0],raw_adc=0))
        self.assertEqual(before,analysis.freeze_reference(records,run.SPEC))
        for p in before['pairs']:
            if p['scale']:
                alltraces=[analysis.lif([sign*p['scale']*v for v in seq]) for seq in p['expected_currents_adc'].values() for sign in (1,-1)]
                self.assertTrue(any(any(v['spike'] for v in trace) for trace in alltraces))

    def test_noise_cannot_pass_on_good_mean_and_lif_keeps_negative(self):
        result=analysis.assess([50,150,50,150,100],100)
        self.assertTrue(result['mean_pass']);self.assertFalse(result['repeatability_pass']);self.assertFalse(result['passes'])
        trace=analysis.lif([-2,0,1.5]);self.assertEqual(trace[0]['after'],-2.)
        self.assertEqual(trace[2]['before'],1.);self.assertEqual(trace[2]['spike'],1);self.assertEqual(trace[2]['after'],0.)

    def test_full_fake_run_counts_fresh_lists_and_frozen_reference(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            folder=Path(directory);fake=FakeConnector();worker=run.SequentialRun(fake,converters,folder,'fake',run.SPEC,'hardware',run.schedule(run.SPEC))
            with contextlib.redirect_stdout(io.StringIO()):result=worker.run()
            self.assertEqual(result['status'],'completed',result);self.assertEqual(fake.closed,1)
            self.assertEqual(result['impact_attempts'],10706);self.assertEqual(len(worker.reads),1536)
            self.assertEqual(len(worker.blocks),1440);self.assertEqual(len(worker.records),9170)
            raw=[json.loads(line) for line in (folder/'raw.jsonl').read_text().splitlines()]
            requests=[r for r in raw if r['event']=='request'];responses=[r for r in raw if r['event']=='result']
            self.assertEqual(len(requests),len(responses));self.assertEqual([r['task_id'] for r in requests],list(range(1,10707)))
            frozen=next(r for r in raw if r['event']=='reference_frozen');self.assertEqual(frozen['before_task_id'],3409)
            self.assertEqual(frozen['sha256'],run.digest(json.loads((folder/'frozen_reference.json').read_bytes())))
            self.assertTrue(all(set(r['command_si']['vol'])<={0.,.1} for r in requests if r['mode']==10))
            self.assertTrue(all(c['dac_code'] in (0,82) for r in requests if r['mode']==10 for c in r['native_voltage_conversion']))
            report=json.loads((folder/'analysis.json').read_bytes());self.assertEqual(len(report['cells']),256)
            self.assertEqual(len(report['coefficients']),47);self.assertEqual(len(report['lif']),24)
            self.assertTrue(all(s['signed']['passes'] for s in report['steps']))
            self.assertTrue(all(c['g_consistent'] for c in report['cells']))
            biased=copy.deepcopy(worker.blocks)
            for block in biased:block['centered_adc']+=100
            hidden=analysis.evaluate(worker.records,worker.reads,biased,worker.reference,run.SPEC)
            self.assertTrue(all(s['signed']['passes'] for s in hidden['steps']))
            self.assertTrue(any(s['row_status']=='fail' for s in hidden['steps']))
            self.assertTrue(any(t['numerical_lif_status']=='pass' for t in hidden['lif']))
            self.assertFalse(any(t['operation_qualified'] for t in hidden['lif']))
            noisy=copy.deepcopy(worker.blocks)
            for block in noisy:
                if block['wl']%2==0:block['centered_adc']+=(-10 if block['repeat']%2==0 else 15)
            noisy_report=analysis.evaluate(worker.records,worker.reads,noisy,worker.reference,run.SPEC)
            self.assertTrue(any(not s['signed']['repeatability_pass'] for s in noisy_report['steps']))
            self.assertFalse(any(t['operation_qualified'] for t in noisy_report['lif']))

    def test_error_after_reference_preserves_raw_and_closes(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            folder=Path(directory);fake=FakeConnector(fail_at=3409)
            worker=run.SequentialRun(fake,converters,folder,'fake',run.SPEC,'hardware',run.schedule(run.SPEC))
            with contextlib.redirect_stdout(io.StringIO()):result=worker.run()
            self.assertEqual(result['status'],'failed');self.assertEqual(fake.closed,1);self.assertEqual(result['impact_attempts'],3409)
            raw=[json.loads(line) for line in (folder/'raw.jsonl').read_text().splitlines()]
            self.assertEqual(raw[-1]['raw_adc'],14000);self.assertEqual(raw[-1]['status'],'error');self.assertEqual(worker.blocks,[])
            report=json.loads((folder/'analysis.json').read_bytes());self.assertTrue(all(t['status']=='inconclusive' for t in report['lif']))

    def test_guard_skips_without_remap_or_zero_fill(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            fake=FakeConnector(high_basis=True);worker=run.SequentialRun(fake,converters,Path(directory),'fake',run.SPEC,'hardware',run.schedule(run.SPEC))
            with contextlib.redirect_stdout(io.StringIO()):result=worker.run()
            self.assertEqual(result['status'],'completed',result);self.assertGreater(result['skipped_commands'],0)
            self.assertEqual(result['impact_attempts']+result['skipped_commands'],10706)
            self.assertEqual(fake.closed,1);self.assertEqual(worker.protocol['pairs'],run.SPEC['pairs'])
            self.assertTrue(all(not r['mask'] for r in worker.records if r['phase']=='evaluation'))


if __name__=='__main__':unittest.main()
