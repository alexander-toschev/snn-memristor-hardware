import unittest
from unittest.mock import patch
from audit_cifar_feedback import CompactChecks,fidelity,pools_for,expected_counts,output_voltage,counts_between,reset_replay,pair_replay


class ScriptedReplay:
    def __init__(self,skip=False):self.position=0;self.events=[];self.ck=CompactChecks();self.skip=skip
    def take(self,phase,cell,vol,**metadata):
        command=dict(mode_flag=7,vol=vol,sign=0 if phase=='candidate_reset' else 1)
        row=dict(event='skipped' if self.skip and vol else 'result',task_id=self.position+1,q=.004,
                 phase=phase,cell=cell,command_si=command,**metadata)
        if row['event']=='skipped':row['planned_command_si']=command
        self.events.append(row);self.position+=1;return row
    def reads(self,phase,cell,n,**meta):return [self.take(phase,cell,0.,repeat=i,**meta) for i in range(n)]


class FeedbackAuditTests(unittest.TestCase):
    def test_fixed_gain_fidelity_known_answer(self):
        result=fidelity([.1,-.2],[[.1,-.2]]*5,1.)
        self.assertEqual(result['raw_target_nrmse'],0.)
        self.assertEqual(result['raw_measured_weight'],[.1,-.2])
        self.assertEqual(result['resolved_sign_error_slots'],[])
        self.assertTrue(result['physical_fidelity_pass'])
        self.assertAlmostEqual(result['uncertainty_nrms'],output_voltage(1)/(.025**.5))

    def test_wrong_sign_cannot_be_rescued_by_common_scale(self):
        result=fidelity([.1],[[ -.1 ]]*5,1.)
        self.assertEqual(result['raw_target_nrmse'],2.)
        self.assertEqual(result['resolved_sign_error_slots'],[0])
        self.assertFalse(result['physical_fidelity_pass'])

    def test_uncertainty_not_divided_by_repeat_count(self):
        result=fidelity([.1],[[.06],[.08],[.1],[.12],[.14]],1.)
        self.assertEqual(result['raw_target_nrmse'],0.)
        self.assertGreater(result['uncertainty_nrms'],.3)
        self.assertFalse(result['physical_fidelity_pass'])

    def test_zero_sample_sd_does_not_remove_adc_floor(self):
        result=fidelity([.00031],[[.00031]]*5,1.)
        self.assertEqual(result['observed_weight_sd'],[0.])
        self.assertAlmostEqual(result['uncertainty_nrms'],1.)
        self.assertFalse(result['physical_fidelity_pass'])

    def test_target_significant_measurement_unresolved(self):
        result=fidelity([.01],[[.00031]]*5,1.)
        self.assertEqual(result['measured_sign_unresolved_significant_slots'],[0])
        self.assertFalse(result['physical_fidelity_pass'])

    def test_source_specific_gain_is_fixed(self):
        result=fidelity([.1],[[.003]]*5,.03)
        self.assertAlmostEqual(result['raw_measured_weight'][0],.1)
        self.assertTrue(result['physical_fidelity_pass'])
        with self.assertRaises(ValueError):fidelity([.1],[[.003]]*5,.031)

    def test_ordered_spares_not_weight_sort(self):
        tile={'slots':[dict(padding=False,trained_weight=w,positive_cell=[0,b],negative_cell=[1,b]) for b,w in [(8,-10.),(2,100.)]]}
        pools=pools_for(tile)
        self.assertEqual(pools['positive_cell'],[[0,8],[0,2],[0,0],[0,1],[0,3]])
        self.assertEqual(pools['negative_cell'],[[1,8],[1,2],[1,0],[1,1],[1,3]])

    def test_compact_comparison_rejects_forged_nested_adc(self):
        ck=CompactChecks();ck.equal({'samples':[1,3]},{'samples':[1,2]},'adc')
        self.assertEqual(ck.count,1);self.assertEqual(ck.errors,['adc'])

    def test_skip_not_impact_or_pulse(self):
        events=[dict(event='skipped'),dict(event='route_frozen'),
            dict(event='result',command_si=dict(mode_flag=7,vol=.3)),
            dict(event='result',command_si=dict(mode_flag=7,vol=0.)),
            dict(event='result',command_si=dict(mode_flag=10,vol=[.1]))]
        self.assertEqual(counts_between(events,0,len(events)),dict(impact_attempts=3,programming_pulse_attempts=1,
                         programming_decisions=2,programming_skipped_current=1))

    def test_last_reset_step_crossing_is_quiet_not_exhausted(self):
        replay=ScriptedReplay();unit=output_voltage(1);signals=[3*unit]*55+[2*unit]*6
        with patch('audit_cifar_feedback.triple',side_effect=[{'voltage_v':v} for v in signals]):
            result=reset_replay(replay,[0,0],{},[])
        self.assertTrue(result['eligible']);self.assertTrue(result['reset_crossed'])
        self.assertEqual(result['reset_stop'],'mode10_quiet_crossed')
        self.assertEqual(result['programming_pulse_attempts'],55)

    def test_reset_current_skip_does_not_create_new_measurement(self):
        replay=ScriptedReplay(skip=True);unit=output_voltage(1)
        with patch('audit_cifar_feedback.triple',side_effect=[{'voltage_v':3*unit}]*6) as measure:
            result=reset_replay(replay,[0,0],{},[])
        self.assertEqual(measure.call_count,6)
        self.assertEqual(result['reset_stop'],'current_guard')
        self.assertFalse(result['eligible'])
        self.assertEqual(result['programming_pulse_attempts'],0)
        self.assertEqual(result['programming_skipped_current'],1)

    def test_pair_leg_depends_on_residual_not_target_sign(self):
        replay=ScriptedReplay();slot=dict(trained_weight=.1,tap_index=0,positive_cell=[0,0],negative_cell=[1,0])
        observations=[([.2,0.],[]),([.1,0.],[])]+[([.1,0.],[])]*5
        with patch('audit_cifar_feedback.replay_block',side_effect=observations):
            result=pair_replay(replay,slot,{},1.,[])
        self.assertEqual(result['active_set_side'],'negative_cell')
        self.assertEqual(result['stop'],'feedback_hit')
        self.assertEqual(result['programming_pulse_attempts'],1)
        self.assertEqual([r['phase'] for r in replay.events[:3]],['pair_guard_refresh','pair_guard_refresh','pair_set'])
        self.assertEqual(replay.events[2]['cell'],[1,0])


if __name__=='__main__':unittest.main()
