"""Offline fake-Connector tests only. These are not native SIM or board evidence."""
import copy
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from experiments.mnist_conductance_20261019 import core, experiment


class Writer:
    def __init__(self):self.events=[];self.checkpoints=[];self.finished=False;self.count=0
    def write(self,row):
        self.count+=1
        if row['event'] in ('pulse_decision','pulse_complete','call_error','result','session_end','port_close'):
            if row['event']!='result' or row.get('phase') not in ('vector_guard','image_signal','checkpoint_r','checkpoint_signal','checkpoint_zero_before','checkpoint_zero_after'):
                self.events.append(copy.deepcopy(row))
    def checkpoint(self,**row):self.checkpoints.append(row)
    def finish(self):self.finished=True


class Converters:
    @staticmethod
    def convert_volt_to_dac(*args):return 41
    @staticmethod
    def convert_dac_to_volt(*args):return .050061050061
    @staticmethod
    def convert_adc_to_volt(gain,bits,ref,adc):return (adc-100)*.0001875


class Connector:
    cb_type='simulator'
    def __init__(self):self.commands=[];self.close_count=0;self.resistance=10000.;self.bad=None
    def open_port(self):return True
    def close_port(self):self.close_count+=1;return True
    def impact(self,command):
        self.commands.append(copy.deepcopy(command))
        if self.bad is not None:return self.bad
        adc=100 if command['mode_flag']==7 else 100+10*sum(v!=0 for v in command['vol'])
        return self.resistance,command['id'],adc


class Clock:
    def __init__(self):self.value=0.
    def now(self):self.value+=.000001;return self.value
    def sleep(self,seconds):self.value+=seconds


def fixture():
    schedule=[[] for _ in range(32)];schedule[1]=list(range(16));schedule[2]=list(range(16,32))
    return dict(images=[dict(train_index=i,schedule=copy.deepcopy(schedule),image_sha256='synthetic',
                            schedule_sha256=core.digest(schedule)) for group in core.GROUPS.values() for i in group])


def collector(reference=None):
    connector=Connector();writer=Writer();clock=Clock()
    runner=core.Collector(connector,Converters,writer,'simulator',fixture(),sleep=clock.sleep,monotonic=clock.now,reference=reference)
    return runner,connector,writer


class ConductanceTests(unittest.TestCase):
    def test_all_event_eligibility_excludes_zero_lag(self):
        pre=[1]+[0]*31;zero=[0]*32
        n=core.Neuron(100.);n.step(pre,.01)
        self.assertEqual(n.e[0],0.)
        n=core.Neuron(100.);n.step(pre,0.);n.step(zero,.01)
        self.assertAlmostEqual(n.e[0],math.exp(-.25))
        n=core.Neuron(100.);n.step(zero,.01);n.step(pre,0.)
        self.assertAlmostEqual(n.e[0],-1.05*math.exp(-.25))
        n=core.Neuron(100.);n.step(pre,0.);n.step(pre,0.);n.step(zero,.01)
        self.assertAlmostEqual(n.e[0],math.exp(-.5)+math.exp(-.25))

    def test_lif_steps_leak_negative_and_choice(self):
        n=core.Neuron(100.)
        values=[n.step([0]*32,.006) for _ in range(3)]
        self.assertEqual([v['post'] for v in values],[0,0,1])
        self.assertAlmostEqual(values[-1]['u_unreset'],1.05)
        self.assertEqual(n.u,0.)
        self.assertLess(n.step([0]*32,-.001)['u_after'],0.)
        self.assertEqual(core.shadow_choice([-.5,.5]+[0.]*30),dict(bl=0,eligibility=-.5,sign=0))
        self.assertIsNone(core.shadow_choice([.249]*32))

    def test_paired_guard_post_reads_and_reference_no_pulse(self):
        run,connector,writer=collector()
        result=run.pulse(2,9,1,'training_update',train_index=0)
        self.assertEqual([(c['wl'],c['vol']) for c in connector.commands],[(6,0.),(2,0.),(2,1.7),(2,0.),(6,0.)])
        self.assertTrue(result['admitted']);self.assertEqual(connector.commands[2]['pulse_width'],.0001)
        connector.commands.clear();connector.resistance=100.
        result=run.pulse(2,9,1,'training_update',train_index=1)
        self.assertFalse(result['admitted']);self.assertEqual(len(connector.commands),2)
        with self.assertRaises(ValueError):run.pulse(6,9,1,'training_update')
        ref,c,w=collector(dict(gain=40.,session='synthetic'))
        result=ref.pulse(2,9,1,'training_update')
        self.assertFalse(result['admitted']);self.assertTrue(result['reference_suppressed'])
        self.assertEqual([(x['wl'],x['vol']) for x in c.commands],[(6,0.),(2,0.),(2,0.),(6,0.)])

    def test_current_guard_budget_and_invalid_raw_response_retained(self):
        run,c,w=collector();c.resistance=10.
        with self.assertRaisesRegex(ValueError,'current'):run.vector(0,[0,1,2,3],'test')
        self.assertTrue(all(command['mode_flag']==7 for command in c.commands))
        run.calls=core.SPEC['maximum_impacts']
        with self.assertRaisesRegex(ValueError,'budget'):run.read_r(0,0,'test')
        run,c,w=collector();c.bad=(float('nan'),1,100)
        with self.assertRaisesRegex(ValueError,'native R'):run.read_r(0,0,'test')
        self.assertEqual(next(e for e in w.events if e['event']=='result')['native_result'][0],'nan')

    def test_image_micro_sum_then_single_lif_and_zero_before_updates(self):
        run,c,w=collector();run.gain=40.
        image=run.image('train',0)
        self.assertAlmostEqual(image['voltages'][0][1],.03)
        self.assertEqual(run.outcomes[0]['neurons'][0]['posts'][1:3],[1,1])
        first_pulse=next(i for i,x in enumerate(c.commands) if x['mode_flag']==7 and x['vol'])
        last_image_vector=max(i for i,x in enumerate(c.commands) if x['mode_flag']==10)
        self.assertLess(last_image_vector,first_pulse)
        self.assertEqual(sum(x['mode_flag']==10 for x in c.commands),80)
        self.assertEqual(run.pulses,4)
        self.assertTrue(all(sum(row)==32 for row in run.image_cell_participation))

    def test_complete_fake_run_main_and_no_eligible_bounds(self):
        run,c,w=collector()
        result=run.run()
        self.assertEqual(result['status'],'completed');self.assertEqual(result['measured_images'],208)
        self.assertEqual(len(result['checkpoints']),14);self.assertEqual(result['gain'],40.)
        self.assertEqual(result['pulses'],2560);self.assertEqual(result['decisions'],2560)
        self.assertEqual(result['impacts'],132352)  # Eight masks/image; max32 masks cap remains172288.
        self.assertEqual(result['diagnostics']['training_admitted_pulses'],512)
        self.assertTrue(result['diagnostics']['phases']['train']['all_learning_traces_identical'])
        self.assertTrue(w.finished);self.assertEqual(c.close_count,1)
        run,c,w=collector()
        with mock.patch.object(Converters,'convert_adc_to_volt',return_value=0.):
            result=run.run()
        self.assertEqual(result['status'],'uninformative_calibration');self.assertEqual(result['measured_images'],32)
        self.assertLessEqual(result['impacts'],34560);self.assertEqual(result['pulses'],2048)
        self.assertFalse(result['outcomes'])

    def test_open_close_failure_and_reference_hardware_gate(self):
        run,c,w=collector()
        with mock.patch.object(c,'open_port',return_value=None),self.assertRaises(ValueError):run.run()
        self.assertEqual(c.close_count,1);self.assertTrue(w.finished)
        with self.assertRaisesRegex(ValueError,'SIM only'):
            experiment.main(['--run-hardware','--reference-main','results/example'])
        with self.assertRaisesRegex(ValueError,'gate closed'):experiment.main(['--run-hardware'])
        with mock.patch.object(experiment,'APPROVED_SIM_SPEC_SHA256',frozenset()),self.assertRaisesRegex(ValueError,'review'):
            experiment.main(['--run-simulator'])

    def test_failed_close_cannot_publish_completed_session(self):
        run,c,w=collector()
        with mock.patch.object(c,'open_port',return_value=None),mock.patch.object(c,'close_port',return_value=False),\
             self.assertRaisesRegex(ValueError,'close failed'):
            run.run()
        self.assertTrue(w.finished)
        end=next(event for event in w.events if event['event']=='session_end')
        self.assertEqual(end['status'],'failed')
        self.assertIn('close failed',end['error'])
        self.assertEqual(next(event for event in w.events if event['event']=='port_close')['result'],False)

    def test_complete_reference_preserves_schedule_without_training_pulses(self):
        run,c,w=collector(dict(gain=40.,session='synthetic-reviewed-main'))
        result=run.run()
        self.assertEqual(result['status'],'completed');self.assertEqual(result['measured_images'],208)
        self.assertEqual(len(result['checkpoints']),14);self.assertEqual(result['impacts'],131840)
        self.assertEqual(result['pulses'],2048);self.assertEqual(result['decisions'],2560)
        self.assertEqual(result['diagnostics']['reference_suppressed'],512)
        self.assertEqual(result['diagnostics']['training_admitted_pulses'],0)
        self.assertEqual(result['calibration']['candidates'],[])
        self.assertTrue(all(not u['admitted'] and 'post' in u and 'control_post' in u
                            for u in result['updates'] if u['phase']=='training_update'))

    def test_real_input_metadata_only_and_silent_columns(self):
        data=experiment.load_fixture()
        active={bl for row in data['images'] for mask in row['schedule'] for bl in mask}
        self.assertEqual(set(range(32))-active,{0,16,24})
        self.assertEqual(len(active)*8,232)

    def test_initial_sim_array_snapshot_survives_runtime_writes(self):
        with tempfile.TemporaryDirectory(dir=experiment.ROOT/'.tmp') as folder:
            output=Path(folder);serial=output/'array';serial.with_suffix('.cb').write_bytes(b'initial synthetic bytes')
            pin=experiment.preserve_initial_array(serial,output)
            serial.with_suffix('.cb').write_bytes(b'changed runtime bytes')
            self.assertEqual(experiment.sha(output/'array_initial.cb'),pin)
            self.assertNotEqual(experiment.sha(serial.with_suffix('.cb')),pin)
            with self.assertRaises(FileExistsError):experiment.preserve_initial_array(serial,output)
            reference_dir=output/'reference';reference_dir.mkdir();refserial=reference_dir/'array'
            refserial.with_suffix('.cb').write_bytes(b'other initial state')
            with self.assertRaisesRegex(ValueError,'initial array differs'):
                experiment.preserve_initial_array(refserial,reference_dir,dict(array_initial_sha256=pin))


if __name__=='__main__':unittest.main()
