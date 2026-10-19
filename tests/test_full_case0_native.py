"""Fake public Connector checks only; never imports or launches a native driver."""
import copy
import json
from pathlib import Path
import unittest
from unittest import mock

from experiments.cifar_full_case0_20261019 import collector as c,audit_native as a


class MemoryTrace:
    def __init__(self):self.rows=[];self.finished=False
    def write(self,row):self.rows.append(copy.deepcopy(row))
    def finish(self):self.finished=True


class FakeConnector:
    cb_type='simulator'
    def __init__(self):self.calls=[];self.closed=False
    def open_port(self):return True
    def close_port(self):self.closed=True;return True
    def impact(self,task):
        self.calls.append(copy.deepcopy(task))
        if task['mode_flag']==7:return (1000000.,task['id'],100)
        return (0.,0,100+sum(v>0 for v in task['vol'])*20)


class Converters:
    @staticmethod
    def convert_volt_to_dac(bits,ref,v):return abs(int(round(v*((2**bits)-1)/ref)))
    @staticmethod
    def convert_dac_to_volt(bits,ref,code):return round(code/(((2**bits)-1)/ref),3)
    @staticmethod
    def convert_adc_to_volt(gain,bits,ref,adc):return round(adc*ref/((2**bits)-1)/gain,5)


class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        cls.p=json.loads((Path(c.__file__).parent/'protocol.json').read_text())
        cls.pin=c.digest(cls.p);cls.image=[0]*3072;cls.weights=np.zeros((64,75),dtype=np.float32)
        cls.trace=MemoryTrace();cls.connector=FakeConnector()
        with mock.patch.object(c,'APPROVED_PROTOCOL_SHA256',frozenset([cls.pin])),mock.patch.object(c,'load_assets',return_value=(cls.image,cls.weights)),mock.patch('builtins.print'):
            runner=c.Collector(cls.connector,Converters,cls.trace,cls.p,cls.pin,'simulator')
            cls.result=runner.run(cls.image,cls.weights,'smoke')

    def test_complete_smoke_independently_replays_and_is_not_full(self):
        payload,report=a.audit(iter(self.trace.rows),self.image,self.weights,self.pin)
        self.assertIsNone(payload)
        self.assertFalse(report['complete_full_tensor'])
        self.assertFalse(report['hardware_import_enabled'])
        self.assertEqual(report['observed_coefficients'],300)
        self.assertEqual(report['covered_tensor_elements'],144)
        self.assertTrue(self.connector.closed and self.trace.finished)

    def test_unregistered_protocol_rejected_before_io(self):
        connector=FakeConnector()
        with mock.patch.object(c,'APPROVED_PROTOCOL_SHA256',frozenset()),self.assertRaises(ValueError):
            c.Collector(connector,Converters,MemoryTrace(),self.p,self.pin,'simulator')
        self.assertEqual(connector.calls,[])

    def test_signed_set_legs_are_bounded_and_approximate_preserved(self):
        weights=self.weights.copy();weights[0,0]=.1;weights[0,1]=-.1
        trace=MemoryTrace();connector=FakeConnector()
        with mock.patch.object(c,'APPROVED_PROTOCOL_SHA256',frozenset([self.pin])):
            runner=c.Collector(connector,Converters,trace,self.p,self.pin,'simulator')
        runner.stage=0;runner.program_stage(weights)
        pulses=[r['command_si'] for r in trace.rows if r['event']=='request' and r['phase']=='program_set']
        self.assertEqual(len(pulses),16)
        for bl,wl in ((0,0),(1,1)):
            selected=[p for p in pulses if p['bl']==bl]
            self.assertEqual([p['vol'] for p in selected],self.p['set_levels_v'])
            self.assertTrue(all(p['wl']==wl and p['sign']==1 for p in selected))
        completed=[r for r in trace.rows if r['event']=='coefficient_complete' and r['filter']==0 and r['tap']<2]
        self.assertEqual(len(completed),2)
        self.assertTrue(all(not r['within_tolerance'] and r['approximate_is_valid_result'] for r in completed))

    def test_current_skips_pulse_and_initial_overshoot_skips_set(self):
        trace=MemoryTrace();connector=FakeConnector()
        with mock.patch.object(c,'APPROVED_PROTOCOL_SHA256',frozenset([self.pin])):
            runner=c.Collector(connector,Converters,trace,self.p,self.pin,'simulator')
        runner.stage=0
        with mock.patch.object(connector,'impact',side_effect=lambda task:(1.,task['id'],100)):
            resistance,admitted=runner.pulse(0,0,.3,1,'program_set',filter=0,tap=0,step=0)
        self.assertEqual(resistance,1.);self.assertFalse(admitted)
        self.assertEqual(runner.pulses,0)
        self.assertEqual(trace.rows[-1]['event'],'write_skipped')
        self.assertFalse(any(r['event']=='request' and r['command_si']['vol']!=0 for r in trace.rows))
        weights=self.weights.copy();weights[0,0]=.1
        original=connector.impact
        def overshoot(task):
            value=original(task)
            if task['mode_flag']==10 and task['wl']==0 and task['vol'][0]>0:
                return (0.,0,value[2]+20)
            return value
        with mock.patch.object(connector,'impact',side_effect=overshoot):runner.program_stage(weights)
        self.assertFalse(any(r['event']=='request' and r['phase']=='program_set' for r in trace.rows))
        first=next(r for r in trace.rows if r['event']=='coefficient_complete')
        self.assertGreater(first['actual_v'],first['target_v'])
        self.assertFalse(first['within_tolerance'])

    def test_malformed_and_nonfinite_native_response_preserved(self):
        for native in ([1,2],(float('nan'),1,100),(1000000.,1,float('inf'))):
            trace=MemoryTrace();connector=FakeConnector()
            with mock.patch.object(c,'APPROVED_PROTOCOL_SHA256',frozenset([self.pin])):
                runner=c.Collector(connector,Converters,trace,self.p,self.pin,'simulator')
            with mock.patch.object(connector,'impact',return_value=native),self.assertRaises(ValueError):
                runner.read_r(0,0,'test_read')
            result=next(r for r in trace.rows if r['event']=='result')
            self.assertEqual(result['native_result'],c.json_safe(native))
            self.assertEqual(trace.rows[-1]['event'],'abort')

    def test_changed_calibration_or_native_command_rejected(self):
        for kind in ('calibration','command'):
            rows=copy.deepcopy(self.trace.rows)
            if kind=='calibration':
                freeze=next(r for r in rows if r['event']=='calibration_frozen')
                freeze['matrix']['matrix'][0][0]+=1
            else:
                request=next(r for r in rows if r['event']=='request' and r['command_si']['mode_flag']==10)
                request['command_si']['vol'][31]=.05
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                a.audit(iter(rows),self.image,self.weights,self.pin)

    def test_incorrect_image_rejected_before_open(self):
        connector=FakeConnector();trace=MemoryTrace()
        with mock.patch.object(c,'APPROVED_PROTOCOL_SHA256',frozenset([self.pin])):
            runner=c.Collector(connector,Converters,trace,self.p,self.pin,'simulator')
        with mock.patch.object(c,'load_assets',return_value=(self.image,self.weights)),mock.patch.object(connector,'open_port') as opened,self.assertRaises(ValueError):
            runner.run([1]*3072,self.weights,'smoke')
        opened.assert_not_called()

    def test_finish_runs_even_if_terminal_event_cannot_be_written(self):
        trace=MemoryTrace();connector=FakeConnector()
        with mock.patch.object(c,'APPROVED_PROTOCOL_SHA256',frozenset([self.pin])):
            runner=c.Collector(connector,Converters,trace,self.p,self.pin,'simulator')
        original=trace.write
        def limited(row):
            if row['event']=='session_complete':raise OSError('raw budget exhausted')
            original(row)
        with mock.patch.object(c,'load_assets',return_value=(self.image,self.weights)),mock.patch.object(runner,'program_stage',side_effect=ValueError('stop')),mock.patch.object(trace,'write',side_effect=limited),mock.patch('builtins.print'),self.assertRaises(OSError):
            runner.run(self.image,self.weights,'smoke')
        self.assertTrue(trace.finished and connector.closed)


if __name__=='__main__':unittest.main()
