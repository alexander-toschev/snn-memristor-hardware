"""Manufactured raw journals exercise audit rejection paths; no Connector imports."""
import json
from pathlib import Path
import tempfile
import unittest
from audit_cell_range_repeat import Checks,audit_raw


class NumericalFixture:
    @staticmethod
    def convert_volt_to_dac(bits,ref,v):return abs(round(v*((1<<bits)-1)/ref))
    @staticmethod
    def convert_dac_to_volt(bits,ref,code):return round(code*ref/((1<<bits)-1),3)
    @staticmethod
    def convert_adc_to_res(gain,load,vol,bits,ref,switches,adc):return round(gain*load*vol*(1<<bits)/(max(adc,20)*ref)-switches-load,2)
    @staticmethod
    def convert_adc_to_volt(gain,bits,ref,adc):return round(adc*ref/(((1<<bits)-1)*gain),5)


BOARD=dict(gain=11,res_load=3000,vol_read=.3,res_switches=10,adc_bit=14,dac_bit=12,
    vol_ref_adc=5,vol_ref_dac=5,sum_gain=330,soft_cc=.002)


class RawTests(unittest.TestCase):
    def fixture(self):
        adc=10000;r=234.03
        req=dict(event='request',session_id='offline',source='hardware',task_id=1,phase='cycle_initial',cycle_id=1,wl=0,bl=8,repeat=0,
            timestamp_utc='2026-10-15T00:00:00+00:00',monotonic_time_s=1.,
            command_si=dict(mode_flag=7,vol=0.,pulse_width=0.,sign=0,wl=0,bl=8,id=1),
            native_voltage_conversion=dict(requested_v=0.,dac_code=0,native_decoded_v=0.),
            physical_pulse_width_s=None,simulator_duration_steps=None,native_adc_samples=10)
        result={**req,'event':'result','status':'ok','latency_s':.001,'native_result':[r,1,adc],
            'native_r':r,'returned_id':1,'raw_adc':adc,'resistance_ohm':r,'conductance_siemens':1/r,
            'q':330/r,'read_voltage_parameter_v':.3}
        skip=dict(event='skipped',session_id='offline',source='hardware',phase='interval_set',cycle_id=1,wl=0,bl=8,step=20,
            timestamp_utc='2026-10-15T00:00:01+00:00',monotonic_time_s=2.,
            planned_command_si=dict(mode_flag=7,vol=.5,pulse_width=0.,sign=1,wl=0,bl=8),
            native_voltage_conversion=dict(requested_v=.5,dac_code=410,native_decoded_v=.501),
            reason='predicted_write_current_exceeds_limit',programming_decision_id=1,
            previous_native_r_ohm=r,resistance_source_task_id=1,predicted_current_a=.5/r,current_limit_a=.002)
        return [req,result,skip]

    def run_journal(self,rows):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path=Path(directory)/'raw.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            ck=Checks();raw=audit_raw(path,dict(source='hardware',session_id='offline'),BOARD,NumericalFixture,ck)
            return ck,raw

    def test_skip_is_not_an_actual_call_or_result(self):
        ck,raw=self.run_journal(self.fixture())
        self.assertEqual(ck.errors,[])
        self.assertEqual((len(raw['records']),raw['decisions'],raw['skips'],raw['pulses']),(1,1,1,0))

    def test_fabricated_skip_adc_and_stale_resistance_detected(self):
        rows=self.fixture();rows[-1].update(raw_adc=10000,previous_native_r_ohm=123,resistance_source_task_id=99)
        ck,_=self.run_journal(rows)
        for suffix in ('no_fabricated_result','previous_native_r_ohm','resistance_source_task_id'):
            self.assertTrue(any(e.endswith(suffix) for e in ck.errors))

    def test_wrong_native_id_is_detected(self):
        rows=self.fixture()[:2];rows[1]['returned_id']=2;rows[1]['native_result'][1]=2
        ck,_=self.run_journal(rows)
        self.assertTrue(any(e.endswith('valid_ID_ADC') for e in ck.errors))

    def test_unpaired_request_cannot_pass(self):
        with self.assertRaisesRegex(ValueError,'Unpaired'):self.run_journal(self.fixture()[:1])

    def test_events_after_error_cannot_pass(self):
        rows=self.fixture()[:2]
        # A diagnostic event after a failed response must be rejected as further protocol activity.
        rows[1].update(status='error',error='response invalid')
        rows.append(dict(event='delay',cycle_id=1,wl=0,bl=8,target_q=.04,requested_s=1.,elapsed_s=1.,started_monotonic_s=2.))
        ck,_=self.run_journal(rows)
        self.assertTrue(any(e.endswith('no_events_after_fault') for e in ck.errors))


if __name__=='__main__':unittest.main()
