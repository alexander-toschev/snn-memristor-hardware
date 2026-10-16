import math
import unittest
from audit_cifar_memristor import Checks,Replay,aggregate_independent,calibration,dac,matrix_select,multiply,neural_trace,output_voltage,paired_difference,resistance,singular2


class IndependentMathTests(unittest.TestCase):
    def test_native_units(self):
        self.assertEqual(dac(.1),dict(requested_v=.1,dac_code=82,native_decoded_v=.1))
        self.assertEqual(output_voltage(16383-1),4.99969)
        self.assertEqual(resistance(1000),29430.32)
        for adc in (0,19,16383):
            with self.assertRaises(ValueError):resistance(adc)

    def test_paired_zeros_per_side_and_per_signal(self):
        m=lambda a:dict(raw_adc=a)
        rows=[dict(zero_before=m(10),zero_after=m(30),signals=[m(50),m(70)]),
              dict(zero_before=m(5),zero_after=m(15),signals=[m(20),m(40)])]
        expected=sum(output_voltage(a)-(output_voltage(10)+output_voltage(30))/2 for a in [50,70])
        expected-=sum(output_voltage(a)-(output_voltage(5)+output_voltage(15))/2 for a in [20,40])
        self.assertAlmostEqual(paired_difference(rows),expected,14)
        self.assertNotEqual(paired_difference(rows),output_voltage(120)-output_voltage(60))

    def test_scalar_not_per_weight_correction(self):
        result=calibration([1.,-2.],[[.01,-.03]]*5)
        self.assertAlmostEqual(result['alpha_v_per_weight'],.014)
        self.assertAlmostEqual(result['effective_weights'][0],5/7)
        self.assertNotEqual(result['effective_weights'],[1.,-2.])
        self.assertTrue(result['decode_usable'])

    def test_unresolved_scale_not_zero_prediction(self):
        for differences in ([[0.,0.]]*5,[[-.01,.02]]*5):
            result=calibration([1.,-2.],differences)
            self.assertFalse(result['decode_usable']);self.assertIsNone(result['effective_weights'])

    def test_quantization_reference_is_diagnostic(self):
        result=calibration([1.],[[output_voltage(1)]]*5)
        self.assertTrue(result['decode_usable'])
        self.assertFalse(result['rms_at_least_five_adc'])
        self.assertTrue(result['quantization_uncertainty_not_in_sample_sd'])

    def test_replay_rejects_wrong_physical_route(self):
        ck=Checks();row=dict(event='result',phase='cifar',wl=0,bl=2,command_si=dict(mode_flag=10,wl=0,vol=[.1]+[0.]*31))
        replay=Replay([row],ck)
        replay.take('cifar',[0,2],[0.,0.,.1]+[0.]*29)
        self.assertTrue(ck.errors)

    def test_empty_step_is_measured_not_fabricated_zero(self):
        m=lambda a:dict(raw_adc=a)
        rows=[dict(zero_before=m(10),zero_after=m(10),signals=[m(11)]),
              dict(zero_before=m(10),zero_after=m(10),signals=[m(10)])]
        self.assertAlmostEqual(paired_difference(rows),output_voltage(11)-output_voltage(10))
        self.assertNotEqual(paired_difference(rows),0.)

    def test_partial_filter_is_not_full_trajectory(self):
        ck=Checks()
        trajectories,_=aggregate_independent([],{'filters':[dict(filter_id=0,tiles=[dict(tile_id=0)])]}, {'cases':[]},ck)
        self.assertEqual(trajectories,[])

    def test_one_lif_and_negative_state(self):
        p={k:[[v] for _ in range(12)] for k,v in [('weight',1.),('bias',0.),('running_mean',0.),('running_var',1.-1e-5)]}
        f={'bntt':{'parameters_t12x64':p}}
        trace=neural_trace([-2.,0.,1.5]+[0.]*9,f,0)
        self.assertEqual(trace[0]['lif_after_reset'],-2.)
        self.assertEqual(trace[1]['lif_after_reset'],-1.)
        self.assertEqual(trace[2]['lif_before_reset'],1.)
        self.assertEqual(trace[2]['lif_spike'],1)
        self.assertEqual(trace[2]['lif_after_reset'],0.)

    def test_nan_never_equal(self):
        c=Checks();c.equal(float('nan'),0.,'nan');self.assertEqual(c.errors,['nan'])

    def test_singular_values_rotation_and_rank(self):
        self.assertEqual(singular2([[0.,-2.],[2.,0.]]),[2.,2.])
        self.assertEqual(singular2([[1.,1.],[1.,1.]]),[2.,0.])
        self.assertEqual(multiply([[1.,2.],[-1.,1.]],[3.,4.]),[11.,1.])

    def test_matrix_fixed_target_recovers_two_independent_gains(self):
        train=[dict(measured_mean_v=[1.,0.],reference_v=[2.,0.]),dict(measured_mean_v=[0.,1.],reference_v=[0.,.5])]
        selection=[dict(measured_repeats_v=[[1.,1.]]*3,reference_v=[2.,.5])]
        result=matrix_select(train,selection)
        self.assertEqual(result['selected_model'],'diagonal')
        self.assertEqual(result['matrix'],[[2.,0.],[0.,.5]])
        self.assertTrue(result['selection_pass'])
        full=next(c for c in result['candidates'] if c['name']=='full_ridge_0.01')
        self.assertAlmostEqual(full['matrix'][0][0],2/1.01)

    def test_matrix_score_includes_repeat_noise(self):
        train=[dict(measured_mean_v=[1.,0.],reference_v=[1.,0.]),dict(measured_mean_v=[0.,1.],reference_v=[0.,1.])]
        selection=[dict(measured_repeats_v=[[.9,.9],[1.,1.],[1.1,1.1]],reference_v=[1.,1.])]
        result=matrix_select(train,selection)
        identity=next(c for c in result['candidates'] if c['name']=='identity')
        self.assertAlmostEqual(identity['selection_score'],8/3)
        self.assertFalse(identity['selection_pass'])

    def test_rank_deficiency_omits_full_correction(self):
        train=[dict(measured_mean_v=[1.,1.],reference_v=[1.,1.]),dict(measured_mean_v=[2.,2.],reference_v=[2.,2.])]
        result=matrix_select(train,[dict(measured_repeats_v=[[1.,1.]]*3,reference_v=[1.,1.])])
        self.assertEqual(result['training_design_rank'],1)
        self.assertIsNone(result['training_design_condition'])
        self.assertEqual([c['name'] for c in result['candidates']],['identity','diagonal'])
        self.assertEqual(result['selected_model'],'identity')

    def test_unbounded_contraction_not_selected_despite_low_error(self):
        train=[dict(measured_mean_v=[1.,0.],reference_v=[.01,0.]),dict(measured_mean_v=[0.,1.],reference_v=[0.,.01])]
        result=matrix_select(train,[dict(measured_repeats_v=[[1.,1.]]*3,reference_v=[.01,.01])])
        self.assertEqual(result['selected_model'],'identity')
        self.assertFalse(next(c for c in result['candidates'] if c['name']=='diagonal')['eligible'])

    def test_sum_tiles_before_single_lif_and_references_remain_distinct(self):
        bn={k:[[v] for _ in range(12)] for k,v in [('weight',1.),('bias',0.),('running_mean',0.),('running_var',1.-1e-5)]}
        refs={k:[[v] for _ in range(12)] for k,v in [('conv_sum',.4),('bntt_current',.4),('lif_before_reset',.4),('lif_after_reset',.4),('lif_spikes',0)]}
        fixture={'bntt':{'parameters_t12x64':bn},'cases':[dict(original_test_index=i,spikes_t12x75=[[1,1]]*12,reference_t12x64=refs) for i in range(10)]}
        route={'filters':[dict(filter_id=0,tiles=[dict(tile_id=i,slots=[dict(padding=False,tap_index=i)]) for i in range(2)])]}
        results=[dict(filter_id=0,tile_id=i,calibration={'effective_weights':[.1]},partials=[dict(case=c,repeat=r,t=t,trained_unit_contribution=.75) for c in range(10) for r in range(5) for t in range(12)]) for i in range(2)]
        ck=Checks();trajectories,_=aggregate_independent(results,route,fixture,ck)
        self.assertFalse(ck.errors);self.assertEqual(len(trajectories),50)
        first=trajectories[0]
        self.assertEqual(first['actual'][0]['conv_sum'],1.5)
        self.assertEqual(first['actual'][0]['lif_spike'],1)
        self.assertEqual(first['actual'][1]['lif_before_reset'],1.5)
        self.assertEqual(first['effective_weight_reference'][0]['conv_sum'],.2)
        self.assertEqual(first['comparison']['trained']['spike_mismatches'],12)

    def test_duplicate_tiles_is_explicit_audit_error(self):
        ck=Checks();r=dict(filter_id=0,tile_id=0)
        aggregate_independent([r,r],{'filters':[]},{'cases':[]},ck)
        self.assertIn('aggregate/no_duplicate_tile',ck.errors)


if __name__=='__main__':unittest.main()
