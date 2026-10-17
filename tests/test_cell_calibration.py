"""Offline bounded-discovery tests. Fake public Connector plus native converters."""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
for name in ('.deps','vendor/MemriBoard'):
    if (ROOT/name).is_dir(): sys.path.insert(0,str(ROOT/name))
from manager.service import converters
import cell_calibration as calibration
import experiment
import render_cell_calibration as report


class FakeConnector:
    cb_type='real'
    def __init__(self,stuck_cell=None,unconfirmed=False,fail_set=False,miss=False,invalid=False,overrange=False):
        self.commands=[]; self.q={}; self.after_reset=set(); self.closed=0
        self.stuck_cell,self.unconfirmed,self.fail_set=stuck_cell,unconfirmed,fail_set
        self.miss,self.invalid,self.overrange=miss,invalid,overrange
    def open_port(self): return True
    def close_port(self): self.closed+=1; return True
    def impact(self,task):
        self.commands.append(copy.deepcopy(task)); assert task['mode_flag'] in (7,10)
        if task['mode_flag']==10:
            assert task['id']==0 and len(task['vol'])==32 and sum(v!=0 for v in task['vol'])<=1
            assert max(task['vol'])<=.1
            for i,v in enumerate(task['vol']): task['vol'][i]=converters.convert_volt_to_dac(12,5,v)
            return (1e-9,0,14000 if self.overrange else sum(task['vol']))
        key=task['wl'],task['bl']; self.q.setdefault(key,.02)
        assert task['pulse_width']==0 if self.cb_type=='real' else task['pulse_width'] in (0,.000125)
        if task['vol']:
            assert .3<=task['vol']<=3
            if task['sign']==0:
                if key!=self.stuck_cell: self.q[key]=.004
                self.after_reset.add(key)
            else:
                if self.fail_set: raise TimeoutError('SET may have been delivered')
                self.after_reset.discard(key)
                self.q[key]=.004 if self.miss else .051
        q=.006 if self.unconfirmed and key in self.after_reset and not task['vol'] else self.q[key]
        adc=0 if self.invalid else converters.convert_res_to_adc(11,3000,.3,14,5,10,330/q)
        return converters.convert_adc_to_res(11,3000,.3,14,5,10,adc),task['id'],adc
    def __getattr__(self,name): raise AssertionError(f'Forbidden Connector attribute: {name}')


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.plan=json.loads((ROOT/'configs/cell_calibration.json').read_text())
        temporary=tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup); self.output=Path(temporary.name)
    def run_fake(self,fake,cells=((1,0),),source='hardware',plan=None):
        worker=calibration.Discovery(fake,converters,self.output,'offline',plan or self.plan,source)
        with contextlib.redirect_stdout(io.StringIO()): result=worker.run(cells)
        raw=[json.loads(line) for line in (self.output/'raw.jsonl').read_text().splitlines()]
        return result,raw
    def test_crossed_five_targets_is_one_state_terminal_deduplicated_and_wl1_allowed(self):
        fake=FakeConnector(); result,raw=self.run_fake(fake)
        cell=result['cells'][0]
        self.assertEqual(result['status'],'completed')
        self.assertEqual(result['checkpoint_count'],3)
        self.assertEqual(cell['crossed_targets'],[.01,.02,.03,.04,.05])
        self.assertEqual(cell['terminal_checkpoint_id'],cell['checkpoints'][-1]['checkpoint_id'])
        self.assertEqual(result['impact_attempts'],113)  # 3*35 +2 pulses +initial3+final3.
        self.assertEqual(len(raw),226)
        self.assertEqual(len(result['initial_array']),1); self.assertEqual(len(result['final_array']),1)
        self.assertEqual(fake.closed,1)
        self.assertEqual(cell['checkpoints'][-1]['crossed_targets'],[.01,.02,.03,.04,.05])
        self.assertNotIn('mvm',cell['checkpoints'][0])
        transfer=cell['checkpoints'][0]['transfer']
        self.assertEqual([r['dac'] for r in transfer],[0,20,41,61,82,0])
        self.assertEqual([r['native_decoded_v'] for r in transfer],[0,.024,.05,.074,.1,0])
        self.assertTrue(all(set(r['command_si']['vol'])<={0,.025,.05,.075,.1} for r in raw if r['command_si']['mode_flag']==10))
    def test_reset_fail_is_bounded_skips_set_then_continues_other_cell(self):
        fake=FakeConnector(stuck_cell=(0,0)); result,raw=self.run_fake(fake,((0,0),(0,1)))
        first,second=result['cells']
        self.assertEqual(result['status'],'completed')
        self.assertEqual(first['reset_pulses'],271); self.assertEqual(first['set_pulses'],0)
        self.assertEqual(first['set_skipped_reason'],'reset_endpoint_not_reached')
        self.assertEqual(second['set_pulses'],1)
        self.assertFalse(any(t['mode_flag']==7 and t['vol'] and t['sign']==1 and t['bl']==0 for t in fake.commands))
    def test_five_separate_crossings_use_at_most_seven_checkpoints(self):
        class Progressive(FakeConnector):
            def impact(self,task):
                result=super().impact(task)
                if task['mode_flag']==7 and task['vol'] and task['sign']==1:
                    count=sum(t['mode_flag']==7 and t['vol']!=0 and t['sign']==1 for t in self.commands)
                    self.q[task['wl'],task['bl']]=.001+count*.01
                    adc=converters.convert_res_to_adc(11,3000,.3,14,5,10,330/self.q[task['wl'],task['bl']])
                    result=(converters.convert_adc_to_res(11,3000,.3,14,5,10,adc),task['id'],adc)
                return result
        result,raw=self.run_fake(Progressive())
        self.assertEqual(result['status'],'completed'); self.assertEqual(result['checkpoint_count'],7)
        self.assertEqual(result['cells'][0]['set_pulses'],5)
        self.assertEqual([s['crossed_targets'] for s in result['cells'][0]['checkpoints'][2:]],[[.01],[.02],[.03],[.04],[.05]])
        self.assertEqual(result['impact_attempts'],257)  # 7*35 +6write +6initial/final.
    def test_native_id_mismatch_stops_both_modes_and_keeps_response(self):
        for wrong_mode in (7,10):
            class WrongID(FakeConnector):
                def impact(self,task):
                    result=super().impact(task)
                    return (result[0],999,result[2]) if task['mode_flag']==wrong_mode else result
            self.output=self.output/str(wrong_mode); self.output.mkdir()
            fake=WrongID(); result,raw=self.run_fake(fake)
            self.assertEqual(result['status'],'failed'); self.assertEqual(fake.closed,1)
            self.assertEqual(raw[-1]['returned_id'],999); self.assertEqual(raw[-1]['status'],'error')
            self.assertIn('ID mismatch',raw[-1]['error'])
    def test_single_reset_crossing_unconfirmed_by_five_reads_no_retry_or_set(self):
        fake=FakeConnector(unconfirmed=True); result,raw=self.run_fake(fake)
        cell=result['cells'][0]
        self.assertTrue(cell['reset_reached']); self.assertFalse(cell['reset_confirmed'])
        self.assertEqual(cell['set_pulses'],0); self.assertEqual(cell['reset_pulses'],1)
        self.assertEqual(cell['set_skipped_reason'],'reset_unconfirmed_after_readback')
    def test_unreached_set_samples_terminal_continues_without_precision_gate(self):
        fake=FakeConnector(miss=True); result,raw=self.run_fake(fake)
        cell=result['cells'][0]
        self.assertEqual(result['status'],'completed'); self.assertEqual(cell['set_pulses'],271)
        self.assertFalse(cell['tested_window_upper_reached']); self.assertEqual(result['checkpoint_count'],3)
        self.assertEqual(cell['unreached_targets'],[.01,.02,.03,.04,.05])
    def test_transport_error_counts_attempted_write_and_preserves_failure(self):
        fake=FakeConnector(fail_set=True); result,raw=self.run_fake(fake)
        self.assertEqual(result['status'],'failed'); self.assertEqual(result['cells'][0]['set_pulses'],1)
        self.assertEqual(raw[-1]['status'],'error'); self.assertIn('may have been delivered',raw[-1]['error'])
        self.assertEqual(fake.closed,1); self.assertEqual(len(result['final_array']),0)
    def test_whole_budget_stops_before_next_write(self):
        plan=copy.deepcopy(self.plan); plan['maximum_programming_pulses']=1
        fake=FakeConnector(); result,raw=self.run_fake(fake,plan=plan)
        self.assertEqual(result['status'],'failed'); self.assertEqual(result['programming_pulse_attempts'],1)
        self.assertEqual(result['cells'][0]['set_pulses'],0)
    def test_invalid_initial_read_stops_before_any_write(self):
        fake=FakeConnector(invalid=True); result,raw=self.run_fake(fake)
        self.assertEqual(len(fake.commands),1); self.assertEqual(result['programming_pulse_attempts'],0)
        self.assertEqual(result['status'],'failed'); self.assertEqual(raw[-1]['raw_adc'],0)
    def test_observed_headroom_stop_never_forces_terminal_or_programming(self):
        fake=FakeConnector(overrange=True); result,raw=self.run_fake(fake)
        self.assertEqual(result['status'],'failed'); self.assertEqual(result['programming_pulse_attempts'],0)
        self.assertEqual(raw[-1]['raw_adc'],14000); self.assertEqual(raw[-1]['status'],'error')
    def test_simulator_width_only_and_single_adc_metadata(self):
        fake=FakeConnector(); fake.cb_type='simulator'
        result,raw=self.run_fake(fake,source='simulator')
        self.assertEqual(result['status'],'completed')
        writes=[r for r in raw if r['command_si']['mode_flag']==7 and r['command_si']['vol']]
        self.assertTrue(all(r['command_si']['pulse_width']==.000125 and r['simulator_duration_steps']==125 for r in writes))
        self.assertTrue(all(r['native_adc_samples']==1 for r in raw))
    def test_default_no_io_routing_and_dryrun_bounds(self):
        with patch.object(calibration,'main',return_value=0) as route:
            for mode in ('--run-simulator','--run-hardware','--dry-run'):
                experiment.main([mode,'--calibrate-cells']); route.assert_called_with([mode])
            route.reset_mock()
            for args in (['--calibrate-cells'],['--run-hardware','--calibrate-cells','--all'],['--dry-run','--calibrate-cells','--map-cells']):
                with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit): experiment.main(args)
            route.assert_not_called()
        with patch.object(calibration,'native_connector') as factory,contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(calibration.main([]),0); factory.assert_not_called()
        with patch.object(calibration,'local_path',side_effect=lambda n:ROOT/n), \
             patch.object(calibration,'check_sources',return_value=(None,b'',{})), \
             patch.object(calibration,'sha256',return_value='hash'), \
             patch.object(calibration,'native_connector') as factory,contextlib.redirect_stdout(io.StringIO()) as output:
            calibration.main(['--dry-run']); factory.assert_not_called()
            self.assertEqual(json.loads(output.getvalue())['maximum_impacts'],203008)
    def test_offline_report_distinguishes_historical_terminal_and_retains_holdout(self):
        result,raw=self.run_fake(FakeConnector())
        manifest={'result_sha256':{p.name:report.sha(p) for p in self.output.iterdir() if p.is_file()}}
        (self.output/'manifest.json').write_text(json.dumps(manifest))
        report.render(self.output.parent,self.output,self.output/'report')
        rendered=json.loads((self.output/'report/calibration.json').read_text())
        self.assertEqual(len(rendered['cells']),256)
        measured=next(c for c in rendered['cells'] if (c['wl'],c['bl'])==(1,0))
        self.assertEqual(measured['status'],'terminal_screen_and_g_pass')
        self.assertFalse(measured['mvm_remeasured_after_other_cells'])
        self.assertFalse(rendered['automatic_mask_exported'])
        state=rendered['states'][0]
        self.assertEqual(state['descriptive_fit']['input'],'dac')
        self.assertEqual(state['descriptive_fit']['heldout_repeat'],4)
        self.assertAlmostEqual(state['descriptive_fit']['slope'],1)
    def test_offline_report_initial_failure_is_all_unmeasured(self):
        result,raw=self.run_fake(FakeConnector(invalid=True))
        cells,states=report.build(result,[])
        self.assertTrue(all(c['status']=='unmeasured' for c in cells)); self.assertEqual(states,[])

    def test_gui_current_boundary_equality_allowed_above_skipped_both_signs(self):
        for sign,resistance,limit in ((0,25.0,.04),(1,500.0,.002)):
            folder=self.output/str(sign); folder.mkdir()
            fake=FakeConnector(); worker=calibration.Discovery(fake,converters,folder,'boundary',self.plan,'hardware')
            try:
                # Exact numeric fixtures isolate the <= comparison from ADC quantization.
                worker.last_mode7[0,0]=(resistance,99)
                accepted=worker.impact(0,0,7,1.0,'set' if sign else 'reset',sign=sign,step=0)
                self.assertEqual(accepted['predicted_current_a'],limit)
                worker.last_mode7[0,0]=(resistance,99)
                previous_write=worker.last_write
                skipped=worker.impact(0,0,7,1.0001,'set' if sign else 'reset',sign=sign,step=1)
                self.assertIsNone(skipped)
                self.assertEqual(len(fake.commands),1)
                self.assertEqual((worker.sequence,worker.pulses,worker.decisions,worker.skipped),(1,1,2,1))
                self.assertEqual(worker.last_write,previous_write)
                self.assertEqual(worker.last_mode7[0,0],(resistance,99))
            finally: worker.log.close()
            records=[json.loads(line) for line in (folder/'raw.jsonl').read_text().splitlines()]
            self.assertEqual([r['event'] for r in records],['request','result','skipped'])
            self.assertEqual(records[-1]['programming_decision_id'],2)
            self.assertGreater(records[-1]['predicted_current_a'],limit)
            self.assertTrue(all(k not in records[-1] for k in ('task_id','q','raw_adc','native_result','returned_id')))
            self.assertNotIn('id',records[-1]['planned_command_si'])

    def test_latest_native_mode7_read_and_write_update_guard_mvm_never_does(self):
        fake=FakeConnector(); worker=calibration.Discovery(fake,converters,self.output,'last-r',self.plan,'hardware')
        try:
            first=worker.impact(0,0,7,0,'checkpoint_read')
            fake.q[0,0]=.004
            latest=worker.impact(0,0,7,0,'checkpoint_read')
            self.assertNotEqual(first['native_r'],latest['native_r'])
            worker.impact(0,0,10,[.1]+[0]*31,'checkpoint_mvm')
            self.assertEqual(worker.last_mode7[0,0],(latest['native_r'],latest['task_id']))
            write=worker.impact(0,0,7,.3,'set',sign=1,step=0)
            self.assertEqual(write['previous_native_r_ohm'],latest['native_r'])
            self.assertEqual(write['resistance_source_task_id'],latest['task_id'])
            self.assertEqual(worker.last_mode7[0,0],(write['native_r'],write['task_id']))
        finally: worker.log.close()

    def test_reset_current_skips_are_bounded_and_continue_next_cell_without_fake_crossing(self):
        plan=copy.deepcopy(self.plan)
        # Deliberately small test-only limit exercises the state machine with consistent ADC/R.
        plan['programming_guard']['sign0_limit_a']=1e-9
        fake=FakeConnector(); result,raw=self.run_fake(fake,((0,0),(0,1)),plan=plan)
        self.assertEqual(result['status'],'completed')
        self.assertEqual(result['programming_decisions'],542)
        self.assertEqual(result['programming_skipped_current'],542)
        self.assertEqual(result['programming_pulse_attempts'],0)
        for cell in result['cells']:
            self.assertEqual((cell['reset_decisions'],cell['reset_skipped_current'],cell['set_pulses']),(271,271,0))
            self.assertEqual(cell['crossed_targets'],[])
            self.assertFalse(cell['reset_reached'])
            self.assertEqual(len(cell['checkpoints']),1)
        skips=[r for r in raw if r['event']=='skipped']
        self.assertEqual([r['programming_decision_id'] for r in skips],list(range(1,543)))
        self.assertEqual([r['step'] for r in skips],list(range(271))*2)
        self.assertFalse(any(t['mode_flag']==7 and t['vol'] for t in fake.commands))
        requests=[r for r in raw if r['event']=='request']
        self.assertEqual([r['task_id'] for r in requests],list(range(1,len(fake.commands)+1)))

    def test_set_current_skips_keep_reset_terminal_without_fake_q_or_pulses(self):
        with patch.dict(calibration.EXPECTED_BOARD,soft_cc=1e-9):
            result,raw=self.run_fake(FakeConnector())
        cell=result['cells'][0]
        self.assertEqual(result['status'],'completed'); self.assertTrue(cell['reset_confirmed'])
        self.assertEqual((cell['set_decisions'],cell['set_skipped_current'],cell['set_pulses']),(271,271,0))
        self.assertEqual(result['programming_decisions'],272)
        self.assertEqual(result['programming_pulse_attempts'],1)
        self.assertEqual(cell['crossed_targets'],[])
        self.assertEqual(cell['terminal_checkpoint_id'],cell['checkpoints'][1]['checkpoint_id'])
        self.assertEqual(result['checkpoint_count'],2)
        self.assertTrue(all(r['planned_command_si']['sign']==1 for r in raw if r['event']=='skipped'))
        checkpoints=[json.loads(line) for line in (self.output/'checkpoints.jsonl').read_text().splitlines()]
        cells,_=report.build(result,checkpoints)
        rendered=next(c for c in cells if (c['wl'],c['bl'])==(1,0))
        self.assertTrue(rendered['current_limited'])
        self.assertEqual(rendered['set_skipped_current'],271)
        self.assertNotEqual(rendered['status'],'incomplete')
        self.assertEqual(rendered['current_limit_reason'],'predicted_write_current_exceeds_limit')

    def test_invalid_native_resistance_stops_without_gui_epsilon_coercion(self):
        class InvalidR(FakeConnector):
            def impact(self,task):
                native=super().impact(task)
                return (0,native[1],native[2])
        fake=InvalidR(); result,raw=self.run_fake(fake)
        self.assertEqual(result['status'],'failed'); self.assertEqual(len(fake.commands),1)
        self.assertEqual(result['programming_decisions'],0); self.assertEqual(raw[-1]['status'],'error')
        self.assertIn('native mode7 resistance',raw[-1]['error'])

    def test_protocol_version_and_guard_cannot_be_silently_removed(self):
        calibration.validate_plan(self.plan)
        plan=copy.deepcopy(self.plan); plan['protocol_id']='cell-conductance-discovery-v1'
        with self.assertRaises(ValueError): calibration.validate_plan(plan)
        plan=copy.deepcopy(self.plan); plan['programming_guard']['sign0_limit_a']=.05
        with self.assertRaises(ValueError): calibration.validate_plan(plan)


if __name__=='__main__': unittest.main()
