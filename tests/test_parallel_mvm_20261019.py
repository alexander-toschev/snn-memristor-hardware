"""Offline fake-Connector qualification tests. Never native SIM or physical I/O."""
import contextlib
import copy
import io
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
for directory in (ROOT,ROOT/'.deps',ROOT/'vendor/MemriBoard'):
    sys.path.insert(0,str(directory))
from manager.service import converters
from experiments.parallel_mvm_20261019 import experiment as mvm


class FakeConnector:
    cb_type='real'
    def __init__(self,invalid=False,saturated=False):
        self.commands=[];self.closed=0;self.invalid=invalid;self.saturated=saturated
        rng=random.Random(1234)
        self.weights=[[rng.randint(1,15) for _ in range(32)] for _ in range(8)]
    def open_port(self):return True
    def close_port(self):self.closed+=1;return True
    def impact(self,task):
        self.commands.append(copy.deepcopy(task))
        if task['mode_flag']==7:
            if task['vol']!=0:raise AssertionError('Programming forbidden')
            adc=0 if self.invalid else converters.convert_res_to_adc(11,3000,.3,14,5,10,33000.)
            return converters.convert_adc_to_res(11,3000,.3,14,5,10,adc),task['id'],adc
        adc=16383 if self.saturated else 20+round(sum(v/.1*w for v,w in zip(task['vol'],self.weights[task['wl']])))
        return 1e-9,0,adc


class ParallelTests(unittest.TestCase):
    def setUp(self):
        (ROOT/'.tmp').mkdir(exist_ok=True)
        temporary=tempfile.TemporaryDirectory(dir=ROOT/'.tmp');self.addCleanup(temporary.cleanup)
        self.output=Path(temporary.name)
    def worker(self,fake):
        return mvm.Acquisition(fake,converters,self.output,'offline',copy.deepcopy(mvm.SPEC),'hardware')
    def test_schedule_exact_budgets_and_disjoint_masks(self):
        entries=mvm.schedule(mvm.SPEC)
        self.assertEqual(len(entries)*8,4832)
        self.assertEqual(1536+8*sum(len(e['mask']) for e in entries),13472)
        fit={tuple(e['mask']) for e in entries if e['phase']=='fit' and e['mask']}
        test={tuple(e['mask']) for e in entries if e['phase']=='test' and e['mask']}
        self.assertEqual(len(fit),30);self.assertEqual(len(test),63)
        self.assertFalse(fit&test)
        self.assertEqual({len(x) for x in test},{1,2,4,8,16,32})
        for d in (2,4,8,16):
            self.assertEqual(sorted(bl for mask in fit if len(mask)==d for bl in mask),list(range(32)))
        configured=json.loads((ROOT/mvm.CONFIG).read_text());mvm.validate(configured)
    def test_full_fake_run_has_fresh_reads_complete_coverage_and_frozen_fit(self):
        fake=FakeConnector();worker=self.worker(fake)
        with contextlib.redirect_stdout(io.StringIO()):result=worker.run()
        self.assertEqual(result['status'],'completed',result.get('error'))
        self.assertEqual(result['impact_attempts'],18304)
        self.assertEqual((result['mode7_reads'],result['mode10_attempts']),(13472,4832))
        self.assertEqual(result['programming_pulse_attempts'],0)
        self.assertEqual(fake.closed,1)
        self.assertEqual(result['complete_vectors'],604)
        self.assertEqual(result['nominal_v'],.1)
        self.assertEqual(worker.calibration['status'],'available')
        first_test=min(v['vector_id'] for v in worker.vectors if v['phase']=='test')
        self.assertLess(max(worker.calibration['fitting_vector_ids']),first_test)
        self.assertTrue(all(v['nominal_v']==.1 for v in worker.vectors if v['phase']!='basis'))
        full=next(v for v in worker.vectors if len(v['mask'])==32)
        refresh=[r for r in worker.reads if full['task_ids'][0]-256<=r['task_id']<full['task_ids'][0]]
        self.assertEqual(len(refresh),256)
        self.assertEqual({(r['wl'],r['bl']) for r in refresh},set(mvm.CELLS))
        analysis=json.loads((self.output/'analysis.json').read_text())
        self.assertEqual(len(analysis['per_wl_density']),48)
        self.assertEqual(len(analysis['lif']),12)
        self.assertEqual({t['gain_per_volt'] for t in analysis['lif']},{10.,100.,1000.})
        self.assertFalse(analysis['trained_weight_accuracy_measured'])
    def test_programming_denied_before_connector(self):
        fake=FakeConnector();worker=self.worker(fake)
        try:
            with self.assertRaisesRegex(ValueError,'forbids programming'):
                worker.impact(0,0,7,.3,'bad',sign=1)
            self.assertEqual(fake.commands,[])
        finally:worker.log.close()
    def test_invalid_initial_read_globally_aborts_and_closes(self):
        fake=FakeConnector(invalid=True)
        with contextlib.redirect_stdout(io.StringIO()):result=self.worker(fake).run()
        self.assertEqual(result['status'],'failed')
        self.assertEqual(result['impact_attempts'],1)
        self.assertEqual(result['mode10_attempts'],0)
        self.assertEqual(fake.closed,1)
        record=json.loads((self.output/'raw.jsonl').read_text().splitlines()[-1])
        self.assertEqual(record['raw_adc'],0)
    def test_saturated_mvm_response_retained_and_aborts(self):
        fake=FakeConnector(saturated=True)
        with contextlib.redirect_stdout(io.StringIO()):result=self.worker(fake).run()
        self.assertEqual(result['status'],'failed')
        self.assertEqual(result['mode10_attempts'],1)
        record=json.loads((self.output/'raw.jsonl').read_text().splitlines()[-1])
        self.assertEqual(record['raw_adc'],16383)
        self.assertEqual(record['status'],'error')
    def test_rank_deficient_fit_is_unavailable_not_identity_calibration(self):
        basis={str(level):[[dict(mean=10.,sd=0.) for _ in range(8)]]+
               [[dict(mean=11.,sd=0.) for _ in range(8)] for bl in range(32)] for level in (.025,.05,.1)}
        vectors=[dict(phase='fit',mask=[i],vector_id=i,adc=[11.+i]*8) for i in range(12)]
        fitted=mvm.freeze_calibration(vectors,basis,.1)
        self.assertEqual(fitted['status'],'unavailable')
        self.assertNotIn('C',fitted)
        self.assertIsNone(mvm.apply_calibration([1]*8,fitted))
    def test_missing_wl_never_imputed_for_matrix(self):
        fitted=dict(status='available',C=[[float(i==j) for j in range(8)] for i in range(8)],offset=[0.]*8)
        self.assertIsNone(mvm.apply_calibration([0.,1.,2.,None,4.,5.,6.,7.],fitted))
    def test_dense_guard_uses_selected_wl_and_no_bl4_cap(self):
        worker=self.worker(FakeConnector())
        try:
            worker.g={(wl,bl):1/33000. for wl,bl in mvm.CELLS}
            self.assertTrue(worker.guard(0,list(range(32)),.1,use_basis=False)['allowed'])
            worker.g[0,0]=1.
            predicted=worker.guard(0,[0],.025,use_basis=False)
            self.assertEqual(predicted['native_decoded_v'],.024)
            self.assertEqual(predicted['guard_voltage_v'],.025)
            self.assertEqual(predicted['current_model_a'],.025)
            self.assertFalse(worker.guard(0,list(range(32)),.1,use_basis=False)['allowed'])
            self.assertTrue(worker.guard(1,list(range(32)),.1,use_basis=False)['allowed'])
        finally:worker.log.close()
    def test_budget_and_default_no_io(self):
        worker=self.worker(FakeConnector())
        try:
            worker.sequence=18304
            with self.assertRaisesRegex(ValueError,'budget'):
                worker.impact(0,0,10,[.1]*32,'test')
        finally:worker.log.close()
        with patch.object(mvm,'check_sources') as check,contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(mvm.main([]),0);check.assert_not_called()


if __name__=='__main__':unittest.main()
