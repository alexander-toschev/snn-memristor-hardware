"""Offline fake Connector checks; no native imports, drivers or network."""
import copy
import math
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from experiments.conductance_stdp_20261019 import core as c
from experiments.conductance_stdp_20261019 import raw


class Trace:
    def __init__(self):self.rows=[];self.finished=False
    def write(self,row):self.rows.append(copy.deepcopy(row))
    def finish(self):self.finished=True


class Clock:
    def __init__(self):self.value=0.
    def now(self):return self.value
    def sleep(self,seconds):self.value+=seconds


class Connector:
    cb_type='simulator'
    def __init__(self):self.calls=[];self.opened=False;self.closed=False
    def open_port(self):self.opened=True;return True
    def close_port(self):self.closed=True;return True
    def impact(self,task):
        self.calls.append(copy.deepcopy(task))
        if task['mode_flag']==7:return (100000.,task['id'],100)
        return (0.,0,100+40*sum(v>0 for v in task['vol']))


class Converters:
    @staticmethod
    def convert_volt_to_dac(bits,ref,v):return int(round(v*((2**bits)-1)/ref))
    @staticmethod
    def convert_dac_to_volt(bits,ref,code):return round(code*ref/((2**bits)-1),3)
    @staticmethod
    def convert_adc_to_volt(gain,bits,ref,adc):return round(adc*ref/((2**bits)-1)/gain,5)


def runner():
    connector=Connector();trace=Trace();clock=Clock()
    return c.Collector(connector,Converters,trace,'simulator',clock.sleep,clock.now)


class STDPTests(unittest.TestCase):
    def test_lif_and_analytic_all_event_eligibility(self):
        n=c.Neuron();self.assertEqual(n.step([1,0,0,0],.01)['eligibility'][0],0.)
        n=c.Neuron();n.step([1,0,0,0],0.)
        self.assertAlmostEqual(n.step([0]*4,.01)['eligibility'][0],math.exp(-.25))
        n=c.Neuron();n.step([0]*4,.01)
        self.assertAlmostEqual(n.step([1,0,0,0],0.)['eligibility'][0],-1.05*math.exp(-.25))
        n=c.Neuron();n.step([1,0,0,0],0.);n.step([1,0,0,0],0.)
        self.assertAlmostEqual(n.step([0]*4,.01)['eligibility'][0],math.exp(-.5)+math.exp(-.25))
        n=c.Neuron();steps=[n.step([0]*4,.006) for _ in range(3)]
        self.assertEqual([x['post'] for x in steps],[0,0,1]);self.assertEqual(n.u,0.)
        self.assertLess(n.step([0]*4,-.001)['u_after'],0.)

    def test_full_schedule_counts_control_and_retention(self):
        r=runner()
        with mock.patch('builtins.print'):result=r.run()
        self.assertEqual(len(result['windows']),28)
        self.assertEqual(len(result['checkpoints']),21*8+64)
        self.assertEqual(len([x for x in r.writer.rows if x['event']=='lif_step']),28*16*2)
        self.assertLessEqual(r.calls,6712);self.assertLessEqual(r.pulses,128)
        prep=[x for x in result['updates'] if x['phase']=='preparation']
        self.assertEqual(len(prep),64)
        for bl in c.BL:
            for wl in c.WL:
                actual=[x['sign'] for x in prep if x['wl']==wl and x['bl']==bl]
                self.assertEqual(actual,([1]*4+[0]*4) if bl in (6,8) else ([0]*4+[1]*4))
        train=[x for x in result['updates'] if x['phase']=='training_update']
        self.assertTrue(train)
        self.assertEqual(len([x for x in r.connector.calls if x['mode_flag']==10]),4776)
        admitted_prep=sum(x['admitted'] for x in prep);admitted_train=sum(x['admitted'] for x in train)
        self.assertEqual(r.calls,6456+admitted_prep+2*len(train)+admitted_train)
        for update in train:
            self.assertEqual(update['wl'],5)
            self.assertEqual(update['guard']['task_id'],update['sham']['task_id']+1)
            self.assertEqual(update['pulse_task_id'],update['guard']['task_id']+1)
        wait=[x for x in r.writer.rows if x['event']=='retention_end'][0]
        self.assertGreaterEqual(wait['elapsed_s'],60.)
        self.assertTrue(r.connector.closed and r.writer.finished)

    def test_upper_budget_when_every_training_decision_admitted(self):
        r=runner();original=c.Neuron.step
        def forced(neuron,pre,voltage):
            value=original(neuron,pre,voltage);neuron.e=[.5]*4;return value
        with mock.patch.object(c.Neuron,'step',forced),mock.patch('builtins.print'):result=r.run()
        self.assertEqual((result['impacts'],result['pulses'],result['decisions']),(6712,128,128))
        with self.assertRaisesRegex(ValueError,'budget'):r.read_r(5,6,'extra')

    def test_skipped_prep_still_checkpoint_and_paired_training_reads(self):
        r=runner();original=r.connector.impact
        def low(task):return (1.,task['id'],100) if task['mode_flag']==7 else original(task)
        with mock.patch.object(r.connector,'impact',side_effect=low):
            update=r.pulse(5,6,1,'training_update',window_index=0,eligibility=.5)
        self.assertFalse(update['admitted']);self.assertEqual(r.pulses,0);self.assertEqual(r.calls,2)
        self.assertEqual([x['phase'] for x in r.writer.rows if x['event']=='request'],['paired_sham','pulse_guard'])
        with self.assertRaisesRegex(ValueError,'Control'):r.pulse(7,6,1,'training_update')
        # A skipped preparation decision still gets its 15 physical checkpoint calls.
        r=runner()
        with mock.patch.object(r,'pulse',return_value={'admitted':False}) as pulse,mock.patch.object(r,'checkpoint') as checkpoint,mock.patch.object(r,'all_cells'),mock.patch.object(r,'window'),mock.patch('builtins.print'):
            r.run()
        self.assertEqual(pulse.call_count,64);self.assertEqual(checkpoint.call_count,64)

    def test_fresh_mvm_guard_and_invalid_response_preservation(self):
        r=runner()
        with mock.patch.object(r.connector,'impact',side_effect=lambda task:(1.,task['id'],100)),self.assertRaisesRegex(ValueError,'MVM current'):
            r.vector(5,[6],'test')
        self.assertEqual(r.calls,1)
        for invalid in ([1,2],(float('nan'),1,100),(100.,99,100),(100.,1,19)):
            r=runner()
            with mock.patch.object(r.connector,'impact',return_value=invalid),self.assertRaises(ValueError):r.read_r(5,6,'test')
            saved=next(x for x in r.writer.rows if x['event']=='result')
            self.assertEqual(saved['native_result'],c.safe(invalid))

    def test_abort_closes_and_trace_finishes_even_when_final_log_fails(self):
        r=runner();original=r.writer.write
        def write(row):
            if row['event']=='session_end':raise OSError('full')
            original(row)
        with mock.patch.object(r,'all_cells',side_effect=ValueError('stop')),mock.patch.object(r.writer,'write',side_effect=write),self.assertRaises(OSError):r.run()
        self.assertTrue(r.connector.closed and r.writer.finished)

    def test_partial_open_and_failed_close_are_failures(self):
        for result in (None,False,0):
            r=runner()
            with mock.patch.object(r.connector,'open_port',return_value=result),self.assertRaisesRegex(ValueError,'open failed'):r.run()
            self.assertTrue(r.connector.closed and r.writer.finished)
        r=runner()
        with mock.patch.object(r.connector,'open_port',side_effect=OSError('partial')),self.assertRaises(OSError):r.run()
        self.assertTrue(r.connector.closed and r.writer.finished)
        r=runner()
        with mock.patch.object(r,'all_cells',side_effect=ValueError('stop')),mock.patch.object(r.connector,'close_port',return_value=False),self.assertRaisesRegex(ValueError,'close failed'):r.run()
        self.assertEqual(r.writer.rows[-1]['status'],'failed');self.assertTrue(r.writer.finished)

    def test_raw_failure_keeps_indexed_lossless_prefix(self):
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)/'raw';writer=raw.Writer(output)
            row={'native_result':['nan',1,100],'event':'result'};writer.write(row)
            actual_size=writer.size;writer.size=64*1024**2
            with self.assertRaises(OSError):writer.write({'extra':1})
            writer.size=actual_size;writer.finish();writer.finish()
            encoded=gzip.decompress((output/'raw.jsonl.gz').read_bytes())
            self.assertEqual(json.loads(encoded),row)
            index=json.loads((output/'index.json').read_text())
            self.assertEqual(index['expanded_sha256'],hashlib.sha256(encoded).hexdigest())
            self.assertEqual(index['lines'],1)

    def test_hardware_gate_closed_and_reviewed_components_pinned(self):
        from experiments.conductance_stdp_20261019 import experiment
        self.assertIn(experiment.protocol_sha(),experiment.APPROVED_PROTOCOL_SHA256['simulator'])
        self.assertNotIn(experiment.protocol_sha(),experiment.APPROVED_PROTOCOL_SHA256['hardware'])
        with self.assertRaisesRegex(ValueError,'registration required'):experiment.main(['--run-hardware'])
        with mock.patch.object(experiment,'REVIEWED_COMPONENT_SHA256',{'core.py':'0'*64}),self.assertRaisesRegex(ValueError,'component changed'):
            experiment.main(['--run-simulator'])


if __name__=='__main__':unittest.main()
