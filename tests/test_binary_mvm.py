"""Public fake Connector and pure arithmetic tests; no hardware access."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from collections import Counter
from manager.service import converters
import binary_mvm as runtime
from binary_mvm_analysis import lif,reference,repeatability,error_test,stats
import experiment

ROOT=Path(__file__).resolve().parents[1]


class FakeConnector:
    cb_type='real'
    def __init__(self,*,zero=False,bad_id=False,headroom=False,lost=False,low_r=False,common_nonlinearity=False):
        self.zero,self.bad_id,self.headroom,self.lost,self.low_r=zero,bad_id,headroom,lost,low_r
        self.common_nonlinearity=common_nonlinearity;self.commands=[];self.closed=0
    def open_port(self):return True
    def close_port(self):self.closed+=1;return True
    def impact(self,task):
        self.commands.append(copy.deepcopy(task))
        assert task['mode_flag'] in (9,10)
        if task['mode_flag']==9:
            adc=converters.convert_res_to_adc(11,3000,.1,14,5,10,50 if self.low_r else 10000)
            return converters.convert_adc_to_res(11,3000,.3,14,5,10,adc),0,adc
        assert set(task)=={'mode_flag','vol','wl','id'} and task['id']==0
        assert len(task['vol'])==32 and all(v in (0.,.1) for v in task['vol'])
        if self.lost:raise TimeoutError('MVM response lost')
        for i,v in enumerate(task['vol']):task['vol'][i]=converters.convert_volt_to_dac(12,5,v)
        adc=sum((30+(task['wl']%2)*3+(bl%5)*5) for bl,code in enumerate(task['vol']) if code)
        data_active=sum(bool(task['vol'][bl]) for bl in runtime.pair_for(task['wl'],runtime.SPEC)[1]['data_bl'])
        if self.common_nonlinearity and data_active>=2:adc+=100
        return -123.,(123 if self.bad_id else 0),(14000 if self.headroom else 0 if self.zero else adc)
    def __getattr__(self,name):raise AssertionError('Forbidden Connector access: '+name)


class BinaryTests(unittest.TestCase):
    def run_fake(self,fake):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            worker=runtime.BinaryRun(fake,converters,Path(temp),'offline',copy.deepcopy(runtime.SPEC),'hardware')
            with contextlib.redirect_stdout(io.StringIO()):summary=worker.run()
            raw=[json.loads(x) for x in (Path(temp)/'raw.jsonl').read_text().splitlines()]
            saved=json.loads((Path(temp)/'summary.json').read_text())
            self.assertEqual(saved['status'],summary['status'])
        return summary,raw

    def test_all_3072_public_calls_order_ttfs_and_independent_phases(self):
        plan=copy.deepcopy(runtime.SPEC);runtime.validate(plan);schedule=list(runtime.tasks(plan))
        self.assertEqual(len(schedule),3072)
        self.assertEqual(dict(Counter(x['phase'] for x in schedule)),plan['phase_counts'])
        for seq in plan['sequences'].values():
            self.assertEqual(len(seq),12)
            for bit in (1,2,4):self.assertEqual(sum(bool(x&bit) for x in seq),1)
        fake=FakeConnector();summary,raw=self.run_fake(fake)
        self.assertEqual(summary['status'],'completed');self.assertEqual(fake.closed,1)
        self.assertEqual(summary['impact_attempts'],3072);self.assertEqual(len(raw),6144)
        requests=[x for x in raw if x['event']=='request']
        self.assertEqual([r['task_id'] for r in requests],list(range(1,3073)))
        self.assertEqual(sum(c['mode_flag']==9 for c in fake.commands),192)
        self.assertEqual(sum(c['mode_flag']==10 for c in fake.commands),2880)
        for r in requests:
            if r['mode']==10:
                self.assertTrue(all(x in (0.,.1) for x in r['command_si']['vol']))
                self.assertTrue(all(x['dac_code'] in (0,82) for x in r['native_voltage_conversion']))
        ids={phase:{r['task_id'] for r in requests if r['phase']==phase} for phase in ('calibration','validation','sequence')}
        self.assertFalse(ids['calibration']&ids['validation']);self.assertFalse(ids['calibration']&ids['sequence'])
        analysis=summary['analysis']
        self.assertTrue(all(r['status']=='pass' for r in analysis['row_additivity']))
        self.assertTrue(all(r['status']=='pass' for r in analysis['signed_pairs']))
        self.assertTrue(any(r['status']=='pass' for r in analysis['digital_lif']))
        self.assertTrue(any(r['status']=='inconclusive' for r in analysis['digital_lif']))
        self.assertFalse(analysis['overall_qualified'])

    def test_common_mode_row_nonlinearity_does_not_automatically_fail_pair(self):
        summary,_=self.run_fake(FakeConnector(common_nonlinearity=True))
        self.assertTrue(all(r['status']=='fail' for r in summary['analysis']['row_additivity']))
        self.assertTrue(all(r['status']=='pass' for r in summary['analysis']['signed_pairs']))

    def test_all_zero_is_inconclusive_not_lif_success(self):
        summary,_=self.run_fake(FakeConnector(zero=True))
        self.assertEqual(summary['status'],'completed')
        self.assertTrue(all(r['status']=='inconclusive' for r in summary['analysis']['signed_pairs']))
        self.assertTrue(all(r['status']=='inconclusive' for r in summary['analysis']['digital_lif']))

    def test_transport_id_and_observed_headroom_global_stop_retain_error(self):
        for options in (dict(lost=True),dict(bad_id=True),dict(headroom=True)):
            fake=FakeConnector(**options);summary,raw=self.run_fake(fake)
            self.assertEqual(summary['status'],'failed');self.assertEqual(len(fake.commands),97)
            self.assertEqual(fake.closed,1);self.assertEqual(raw[-1]['status'],'error')
            self.assertNotIn('analysis',summary)
            if options.get('headroom'):self.assertEqual(raw[-1]['raw_adc'],14000)

    def test_precomputed_model_guard_stops_before_any_mvm(self):
        fake=FakeConnector(low_r=True);summary,raw=self.run_fake(fake)
        self.assertEqual(summary['status'],'failed');self.assertEqual(len(fake.commands),96)
        self.assertTrue(all(c['mode_flag']==9 for c in fake.commands));self.assertEqual(fake.closed,1)
        self.assertEqual(raw[-1]['event'],'rejected_mvm')

    def test_native_zero_mvm_does_not_use_invalid_native_r(self):
        # Native mode10 tuple R=-123 is deliberately meaningless; ADC0 remains valid.
        summary,raw=self.run_fake(FakeConnector(zero=True))
        rows=[r for r in raw if r['event']=='result' and r['mode']==10]
        self.assertTrue(all(r['status']=='ok' and r['raw_adc']==0 for r in rows))

    def test_lif_threshold_inclusive_leak_signed_and_hard_reset(self):
        out=lif([1.,-.4,.3,1.])
        self.assertEqual([r['spike'] for r in out],[1,0,0,1])
        self.assertEqual(out[0]['after_reset'],0.);self.assertEqual(out[1]['after_reset'],-.4)
        self.assertAlmostEqual(out[2]['before_reset'],.1)

    def test_reference_shared_zero_covariance_and_repeatability(self):
        c={0:dict(mean=10,sd=2,n=5),1:dict(mean=20,sd=3,n=5),2:dict(mean=30,sd=4,n=5),4:dict(mean=40,sd=5,n=5)}
        prediction,var=reference(c,7);self.assertEqual(prediction,70)
        self.assertAlmostEqual(var,(9+16+25+4*4)/5)
        self.assertTrue(error_test(100,100,100,10000)['passes'])
        self.assertFalse(repeatability(100,20,100,1)['passes'])
        self.assertFalse(repeatability(0,3,0,1,zero=True)['passes'])
        self.assertTrue(repeatability(-100,1,-100,1)['passes'])

    def test_dispatch_no_default_io_and_reject_operational_override(self):
        with patch.object(runtime,'main',return_value=0) as run:
            for flag in ('--dry-run','--run-simulator','--run-hardware'):
                experiment.main(['--test-binary-mvm',flag]);run.assert_called_with([flag])
            run.reset_mock()
            for args in (['--test-binary-mvm'],['--test-binary-mvm','--dry-run','--all'],['--test-binary-mvm','--dry-run','--map-cells']):
                with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):experiment.main(args)
            run.assert_not_called()
        with patch.object(runtime,'native_connector') as connect,contextlib.redirect_stdout(io.StringIO()):
            runtime.main([]);connect.assert_not_called()
        invalid=copy.deepcopy(runtime.SPEC);invalid['maximum_impacts']=4000
        with self.assertRaises(ValueError):runtime.validate(invalid)


if __name__=='__main__':unittest.main()
