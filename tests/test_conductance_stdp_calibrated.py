"""Offline fake-data tests only; no native imports or Connector construction."""
import copy
import unittest
from unittest import mock

from experiments.conductance_stdp_calibrated_20261019 import core as c
from tests.test_conductance_stdp import Connector,Converters,Trace,Clock


class VariedConnector(Connector):
    def impact(self,task):
        self.calls.append(copy.deepcopy(task))
        if task['mode_flag']==7:return (100000.,task['id'],100)
        amplitudes={6:3,8:10,9:3,14:10}
        return (0.,0,100+sum(amplitudes[bl] for bl,v in enumerate(task['vol']) if v))


def runner(connector=None):
    trace=Trace();clock=Clock()
    return c.Collector(connector or VariedConnector(),Converters,trace,'simulator',clock.sleep,clock.now)


class CalibratedTests(unittest.TestCase):
    def test_natural_fake_fit_choice_and_full_schedule(self):
        r=runner()
        with mock.patch('builtins.print'):result=r.run()
        self.assertEqual(result['status'],'completed')
        self.assertEqual(result['gain'],400.)
        self.assertEqual([w['phase'] for w in result['windows']],['fit']*2+['pretest']*2+['train']*16+['posttest']*4+['retention']*4)
        self.assertEqual(len(result['checkpoints']),232)
        self.assertEqual(sum(row['event']=='fit_measurement_step' for row in r.writer.rows),64)
        self.assertEqual(sum(row['event']=='lif_step' for row in r.writer.rows),832)
        self.assertEqual(sum(task['mode_flag']==10 for task in r.connector.calls),4776)
        training=[value for value in result['updates'] if value['phase']=='training_update']
        self.assertTrue(training)
        self.assertEqual({value['sign'] for value in training},{0,1})
        self.assertTrue(all(value['wl']==5 for value in training))
        self.assertTrue(all(value['pulse_task_id']==value['guard']['task_id']+1==value['sham']['task_id']+2 for value in training))
        a=sum(value['admitted'] for value in result['updates'] if value['phase']=='preparation')
        b=sum(value['admitted'] for value in training)
        self.assertEqual(result['impacts'],6456+a+2*len(training)+b)
        self.assertLessEqual(result['impacts'],6712);self.assertLessEqual(result['pulses'],128)
        freeze=next(i for i,row in enumerate(r.writer.rows) if row['event']=='gain_frozen')
        first_lif=next(i for i,row in enumerate(r.writer.rows) if row['event']=='lif_step')
        self.assertLess(freeze,first_lif)
        self.assertTrue(all(row['gain']==400. for row in r.writer.rows if row['event']=='lif_step'))
        self.assertTrue(r.connector.closed and r.writer.finished)

    def test_noeligible_preserves_prefix_and_stops(self):
        r=runner()
        original=r.connector.impact
        def silent(task):
            result=original(task)
            return result if task['mode_flag']==7 else (0.,0,100)
        with mock.patch.object(r.connector,'impact',side_effect=silent):result=r.run()
        self.assertEqual(result['status'],'uninformative_calibration')
        self.assertIsNone(result['gain'])
        self.assertEqual((result['impacts'],result['pulses'],len(result['checkpoints'])),(1536,64,80))
        self.assertEqual(sum(task['mode_flag']==10 for task in r.connector.calls),912)
        self.assertEqual(len(result['windows']),2)
        self.assertFalse(any(row['event']=='lif_step' for row in r.writer.rows))
        self.assertFalse(any(value['phase']=='training_update' for value in result['updates']))
        self.assertTrue(r.connector.closed and r.writer.finished)

    def test_control_data_and_unrelated_records_do_not_choose_gain(self):
        r=runner();fits=[r.window('fit',index) for index in range(2)]
        before=c.choose_gain(fits)
        changed=copy.deepcopy(fits)
        for window in changed:
            for step in window['steps']:
                if step['wl']==7:step['measurement']['centered_v']=float('nan')
        self.assertEqual(c.choose_gain(changed),before)
        self.assertEqual(before['selected_gain'],400.)
        self.assertEqual(len(before['candidates']),9)

    def test_zero_step_spiking_is_not_eligible(self):
        r=runner();fits=[r.window('fit',index) for index in range(2)]
        for window in fits:
            for step in window['steps']:
                if step['wl']==5:step['measurement']['centered_v']=.2
        choice=c.choose_gain(fits)
        self.assertIsNone(choice['selected_gain'])
        self.assertTrue(all(any(window['zero_input_posts'] for window in candidate['windows']) for candidate in choice['candidates']))

    def test_shared_gain_and_window_schedules_are_preregistered(self):
        self.assertEqual(c.GAINS,(10.,20.,40.,80.,100.,200.,400.,800.,1600.))
        self.assertEqual(c.timing('fit',0),(1,5,9,13))
        self.assertEqual(c.timing('train',0),(3,10,11,2))
        self.assertEqual(c.timing('train',1),(2,11,10,3))
        self.assertEqual(c.timing('pretest',0),(1,4,9,12))
        with self.assertRaisesRegex(ValueError,'not frozen'):runner().window('train',0)
        with self.assertRaises(ValueError):c.Neuron(50.)

    def test_native_registration_is_enforced_without_launch(self):
        from experiments.conductance_stdp_calibrated_20261019 import experiment
        self.assertNotIn(experiment.protocol_sha(),experiment.APPROVED_PROTOCOL_SHA256['hardware'])
        # Remain offline even after root legitimately registers native SIM.
        closed={source:frozenset() for source in ('simulator','hardware')}
        with mock.patch.object(experiment,'APPROVED_PROTOCOL_SHA256',closed):
            for flag in ('--run-simulator','--run-hardware'):
                with self.assertRaisesRegex(ValueError,'registration required'):experiment.main([flag])

    def test_upper_bound_and_failed_close(self):
        r=runner();original=c.Neuron.step
        def forced(neuron,pre,voltage):
            value=original(neuron,pre,voltage);neuron.e=[.5]*4;return value
        with mock.patch.object(c.Neuron,'step',forced),mock.patch('builtins.print'):result=r.run()
        self.assertEqual((result['impacts'],result['pulses'],result['decisions']),(6712,128,128))
        r=runner()
        with mock.patch.object(r,'all_cells',side_effect=ValueError('stop')),mock.patch.object(r.connector,'close_port',return_value=False),self.assertRaisesRegex(ValueError,'close failed'):
            r.run()
        self.assertEqual(r.writer.rows[-1]['status'],'failed');self.assertTrue(r.writer.finished)


if __name__=='__main__':unittest.main()
