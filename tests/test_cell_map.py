"""Offline cell-map tests: real native converters, fake public Connector only."""
import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
for relative in ('.deps','vendor/MemriBoard'):
    if (ROOT/relative).is_dir(): sys.path.insert(0,str(ROOT/relative))
from manager.service import converters
import cell_map
import experiment


class FakeConnector:
    def __init__(self,invalid_cell=False,overrange=False,fail=False):
        self.commands=[]
        self.closed=0
        self.invalid_cell,self.overrange,self.fail=invalid_cell,overrange,fail
    def open_port(self): return True
    def close_port(self): self.closed+=1; return True
    def impact(self,task):
        assert task['mode_flag'] in (9,10)
        self.commands.append(json.loads(json.dumps(task)))
        if self.fail and len(self.commands)==770: raise TimeoutError('missing response')
        if task['mode_flag']==9:
            return (1234,0,0 if self.invalid_cell and task['wl']==0 and task['bl']==0 else 500)
        assert set(task)=={'mode_flag','vol','id','wl'} and task['id']==0
        vector=task['vol']
        assert len(vector)==32 and set(vector)<={0,.05,.1}
        assert sum(v!=0 for v in vector)<=1
        level=max(vector)
        adc=14000 if self.overrange else (100 if level==0 else 150 if level==.05 else 200)
        vector[:]=[82]*32
        return (1e-9,0,adc)


class MapTests(unittest.TestCase):
    def setUp(self):
        self.criteria=cell_map.load_criteria(ROOT/'configs/cell_map.json')['criteria']
        temp=tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.output=Path(temp.name)
    def run_fake(self,fake):
        summary=cell_map.collect(fake,converters,self.output,'offline',self.criteria)
        raw=[json.loads(s) for s in (self.output/'raw.jsonl').read_text().splitlines()]
        cells=json.loads((self.output/'cells.json').read_text())
        return summary,raw,cells
    def test_all_cells_4136_counts_sequence_native_ids_dac_and_mask(self):
        fake=FakeConnector()
        summary,raw,cells=self.run_fake(fake)
        self.assertEqual(summary['status'],'completed')
        self.assertEqual(summary['attempted'],4136)
        self.assertEqual(summary['counts'],dict(candidate=256,reject=0,unmeasured=0))
        self.assertEqual(fake.closed,1)
        requests=[r for r in raw if r['event']=='request']
        self.assertEqual([r['task_id'] for r in requests],list(range(1,4137)))
        mvm=[r for r in requests if r['command_si']['mode_flag']==10]
        expected=[(repeat,bl,level,wl) for repeat in range(5)
                  for bl,level in [(None,0.0)]+[(b,v) for b in range(32) for v in (.05,.1)]
                  for wl in range(8)]
        self.assertEqual([(r['repeat'],r['bl_basis'],r['nominal_v'],r['command_si']['wl']) for r in mvm],expected)
        for r in mvm:
            code={0:0,.05:41,.1:82}[r['nominal_v']]
            expected_dac=[code if i==r['bl_basis'] else 0 for i in range(32)]
            self.assertEqual(r['vol_dac'],expected_dac)
            self.assertEqual(r['command_si']['id'],0)
        with (self.output/'candidate_cells.csv').open(newline='') as stream:
            mask=list(csv.reader(stream))
        self.assertEqual(mask[0],['wl','bl'])
        self.assertEqual(len(mask),257)
        self.assertEqual(cells[0]['metrics']['snr_low'],50)
    def test_invalid_model_cell_skips_explicitly_and_others_continue(self):
        summary,raw,cells=self.run_fake(FakeConnector(invalid_cell=True))
        self.assertEqual(summary['status'],'completed')
        self.assertEqual(summary['attempted'],4126)
        self.assertEqual(summary['skipped_mvm'],10)
        self.assertEqual(sum(r['event']=='skipped' for r in raw),10)
        self.assertEqual(cells[0]['status'],'reject')
        self.assertIsNone(cells[0]['metrics']['pre_g']['mean'])
        self.assertEqual(summary['counts']['candidate'],255)
    def test_observed_overrange_preserves_partial_and_empty_candidate_mask(self):
        fake=FakeConnector(overrange=True)
        summary,raw,cells=self.run_fake(fake)
        self.assertEqual(summary['status'],'failed')
        self.assertEqual(summary['attempted'],769)
        self.assertEqual(raw[-1]['raw_adc'],14000)
        self.assertEqual(raw[-1]['status'],'error')
        self.assertEqual(summary['counts']['unmeasured'],256)
        self.assertEqual((self.output/'candidate_cells.csv').read_text().strip(),'wl,bl')
        self.assertEqual(fake.closed,1)
    def test_transport_failure_stops_and_flushes_request_result(self):
        fake=FakeConnector(fail=True)
        summary,raw,cells=self.run_fake(fake)
        self.assertEqual(summary['attempted'],770)
        self.assertEqual(raw[-2]['event'],'request')
        self.assertEqual(raw[-1]['event'],'result')
        self.assertIn('missing response',raw[-1]['error'])
        self.assertEqual(fake.closed,1)
    def test_failure_mid_postread_never_exports_partial_candidate_mask(self):
        class LateFailure(FakeConnector):
            def impact(self,task):
                result=super().impact(task)
                if len(self.commands)==3372: raise TimeoutError('late post-read failure')
                return result
        summary,raw,cells=self.run_fake(LateFailure())
        self.assertEqual(summary['status'],'failed')
        self.assertEqual(cells[0]['status'],'candidate')  # Individual provisional metrics retained.
        self.assertFalse(summary['mask_usable'])
        self.assertEqual((self.output/'candidate_cells.csv').read_text().strip(),'wl,bl')
        self.assertIn('Run status: failed',(self.output/'map.html').read_text())
    def test_clipped_zero_response_never_candidate(self):
        cell=dict(wl=0,bl=0,pre_g=[1e-5]*3,post_g=[1e-5]*3,low_adc=[0]*5,high_adc=[0]*5,reasons=[])
        result=cell_map.classify(cell,[0]*5,self.criteria)
        self.assertEqual(result['status'],'reject')
        self.assertIsNone(result['metrics']['cv_high'])
        self.assertIn('weak_or_nonpositive_signal_high',result['reasons'])
    def test_zero_noise_is_included_and_drift_linearity_reject(self):
        cell=dict(wl=0,bl=0,pre_g=[1e-5]*3,post_g=[2e-5]*3,low_adc=[150]*5,high_adc=[170]*5,reasons=[])
        result=cell_map.classify(cell,[90,100,110,90,110],self.criteria)
        self.assertEqual(result['status'],'reject')
        self.assertGreater(result['metrics']['signal_high_sd_adc'],0)
        self.assertIn('pre_post_conductance_difference',result['reasons'])
        cell.update(pre_g=[1e-5]*3,post_g=[1e-5]*3,low_adc=[80]*5,high_adc=[200]*5)
        result=cell_map.classify(cell,[0]*5,self.criteria)
        self.assertEqual(result['reasons'],['nonlinear_zero_adjusted_response'])
        self.assertEqual(result['metrics']['linearity_epsilon_adc'],40)
        self.assertEqual(result['metrics']['linearity_tolerance_adc'],10)
    def test_dispatch_and_dry_run_no_connector(self):
        with patch.object(cell_map,'main',return_value=0) as run:
            experiment.main(['--dry-run','--map-cells'])
            run.assert_called_once_with(['--dry-run'])
            run.reset_mock()
            for args in (['--map-cells'],['--run-hardware','--map-cells','--all'],['--dry-run','--map-cells','--check-mvm']):
                with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit): experiment.main(args)
            run.assert_not_called()
        with patch.object(cell_map,'sha256',return_value=cell_map.READ_ALL_SHA256), \
             patch.object(cell_map,'preflight',return_value=(None,b'',{})), \
             patch.object(cell_map,'load_criteria',return_value={'criteria':self.criteria}), \
             patch.object(cell_map,'native_connector') as factory,contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cell_map.main(['--dry-run']),0)
            factory.assert_not_called()


if __name__=='__main__': unittest.main()
