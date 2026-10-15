"""Offline arithmetic and public fake Connector tests; never hardware."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from manager.service import converters
import all256_mvm as runtime
import mvm_calibration as analysis
import experiment

ROOT=Path(__file__).resolve().parent


class FakeConnector:
    cb_type='real'
    def __init__(self,fail=False,high_basis=False):self.commands=[];self.closed=0;self.fail=fail;self.high_basis=high_basis
    def open_port(self):return True
    def close_port(self):self.closed+=1;return True
    def impact(self,task):
        self.commands.append(copy.deepcopy(task));assert task['mode_flag'] in (9,10)
        if task['mode_flag']==9:
            adc=converters.convert_res_to_adc(11,3000,.1,14,5,10,10000)
            return converters.convert_adc_to_res(11,3000,.3,14,5,10,adc),0,adc
        assert task['id']==0 and set(task)=={'mode_flag','vol','wl','id'}
        active=sum(bool(v) for v in task['vol']);assert active<=4
        for i,v in enumerate(task['vol']):task['vol'][i]=converters.convert_volt_to_dac(12,5,v)
        return -111.,0,(14000 if self.fail else (4000 if self.high_basis else 50)*active)


class CalibratedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.entries=runtime.schedule(runtime.SPEC)

    def test_frozen_budget_disjoint_masks_and_dense_retained(self):
        self.assertEqual(len(self.entries)*8+1536,18576)
        masks={role:{tuple(e['mask']) for e in self.entries if e['phase']==role} for role in ('fit','selection','test')}
        self.assertEqual([len(masks[r]) for r in masks],[80,40,81])
        for a,b in (('fit','selection'),('fit','test'),('selection','test')):self.assertFalse(masks[a]&masks[b])
        self.assertIn(tuple(range(32)),masks['test'])
        self.assertEqual([e['vector_id'] for e in self.entries],list(range(1,len(self.entries)+1)))
        self.assertEqual(runtime.schedule(runtime.SPEC),self.entries)
        probes={tuple(e['mask']) for e in self.entries if e['phase']=='input_path_probe'}
        self.assertEqual(len(probes),30)
        for role,values in masks.items():
            self.assertFalse(values&probes)
            self.assertEqual({b for m in values if len(m)<=4 for b in m},set(range(32)))
        self.assertFalse(any('cifar' in e['phase'] for e in self.entries))
        for pattern in runtime.SPEC['synthetic_patterns'].values():
            self.assertEqual(sorted(b for step in pattern for b in step),list(range(8)))
            self.assertTrue(all(len(step)<=2 for step in pattern))

    def test_frontdoor_explicit_only_and_phase_conflict(self):
        with patch('all256_mvm.main',return_value=0) as called,contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(experiment.main([]),0);called.assert_not_called()
            self.assertEqual(experiment.main(['--test-calibrated-mvm','--dry-run']),0)
            called.assert_called_once_with(['--dry-run'])
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            experiment.main(['--test-calibrated-mvm','--run-hardware','--phase','phase1'])

    def test_full_fake_run_skip_ledger_fresh_lists_no_programming(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            fake=FakeConnector();worker=runtime.Acquisition(fake,converters,Path(directory),'fake',runtime.SPEC,'hardware',self.entries)
            with contextlib.redirect_stdout(io.StringIO()):result=worker.run()
            self.assertEqual(result['status'],'completed');self.assertEqual(fake.closed,1)
            self.assertEqual(len(worker.reads),1536);self.assertTrue(worker.skips)
            self.assertEqual(len(worker.vectors)+len(worker.skips),len(self.entries))
            self.assertEqual(result['impact_attempts'],1536+8*len(worker.vectors))
            raw=[json.loads(line) for line in (Path(directory)/'raw.jsonl').read_text().splitlines()]
            requests=[r for r in raw if r['event']=='request']
            self.assertEqual([r['task_id'] for r in requests],list(range(1,len(requests)+1)))
            self.assertTrue(all(len(r['mask'])<=4 for r in requests if r['mode']==10))
            for r in requests:
                if r['mode']==10:
                    self.assertTrue(all(v in (0.,.1) for v in r['command_si']['vol']))
                    self.assertTrue(all(c['dac_code'] in (0,82) for c in r['native_voltage_conversion']))
            folder=Path(directory);runtime.save_json(folder/'protocol.json',runtime.SPEC);runtime.save_json(folder/'schedule.json',self.entries)
            runtime.save_json(folder/'manifest.json',dict(session_id='fake',source='hardware',status='completed',impact_attempts=worker.sequence,
                result_sha256={name:runtime.sha256(folder/name) for name in ('raw.jsonl','observations.json','protocol.json','schedule.json')}))
            observed=json.loads((folder/'observations.json').read_bytes())
            verified=analysis.verify_acquisition(folder,observed);self.assertTrue(verified['passed'])
            observed['vectors'][0]['adc'][0]+=1
            with self.assertRaisesRegex(ValueError,'Compact MVM'):analysis.verify_acquisition(folder,observed)

    def test_observed_overrange_retained_and_close_no_fake_vector(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            fake=FakeConnector(fail=True);worker=runtime.Acquisition(fake,converters,Path(directory),'fake',runtime.SPEC,'hardware',self.entries)
            with contextlib.redirect_stdout(io.StringIO()):result=worker.run()
            raw=[json.loads(line) for line in (Path(directory)/'raw.jsonl').read_text().splitlines()]
            self.assertEqual(result['status'],'failed');self.assertEqual(fake.closed,1)
            self.assertEqual(result['impact_attempts'],769);self.assertEqual(len(worker.vectors),0)
            self.assertEqual(raw[-1]['raw_adc'],14000);self.assertEqual(raw[-1]['status'],'error')

    def test_fresh_basis_overrides_misleading_low_mode9_model(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            worker=runtime.Acquisition(FakeConnector(),converters,Path(directory),'fake',runtime.SPEC,'hardware',[])
            worker.g={(w,b):1e-6 for w in range(8) for b in range(32)}
            worker.basis={0:[(0.,0.)]*8,1:[(4000.,0.)]*8,2:[(4000.,0.)]*8}
            decision=worker.guard(dict(phase='fit',mask=[0,1]));worker.log.close()
            self.assertFalse(decision['allowed'])
            self.assertIn('wl0_fresh_basis_headroom',decision['reasons'])
            self.assertFalse(decision['upper_bound_certified'])

    def test_rank_deficiency_not_fixed_by_ridge(self):
        H=np.ones((8,32));fit={tuple(range(k)):np.tile(np.ones(8)*k,(5,1)) for k in range(2,10)}
        models=analysis.candidate_models(fit,H,np.zeros(8),runtime.SPEC['calibration'])
        full=[c for c in models if c['name'].startswith('full')]
        self.assertTrue(full);self.assertTrue(all(not c['eligible'] for c in full))

    def test_excessive_correction_gain_ineligible(self):
        H=np.tile(np.eye(8),(1,4))*1000
        fit={(b,):np.tile(np.eye(8)[b%8],(5,1)) for b in range(16)}
        models=analysis.candidate_models(fit,H,np.zeros(8),runtime.SPEC['calibration'])
        high=[c for c in models if c['name']!='identity' and c['gain']>10]
        self.assertTrue(high);self.assertTrue(all(not c['eligible'] for c in high))

    def test_final_data_cannot_change_selected_model(self):
        H=np.tile(np.eye(8),(1,4))*30;vectors=[]
        for phase,masks in [('basis',[()]+[(b,) for b in range(32)]),('fit',[(b,) for b in range(16)]),
                            ('selection',[(0,1)]),('selection_anchors',[()]+[(b,) for b in range(32)]),('test',[(2,3)])]:
            for mask in masks:
                for repeat in range(5):vectors.append(dict(phase=phase,mask=list(mask),repeat=repeat,adc=analysis.reference(H,mask).tolist()))
        observations=dict(vectors=vectors,readbacks=[],skips=[])
        first=analysis.analyze(observations,runtime.SPEC)
        for v in vectors:
            if v['phase']=='test':v['adc']=[9999]*8
        second=analysis.analyze(observations,runtime.SPEC)
        self.assertEqual(first['selected_model'],second['selected_model'])
        self.assertEqual(first['C'],second['C']);self.assertEqual(first['d'],second['d'])

    def test_lif_missing_trajectory_and_unexcited_reference_inconclusive(self):
        H=np.zeros((8,32));H[::2]=20
        basis={():np.zeros((5,8)),**{(b,):np.tile(H[:,b],(5,1)) for b in range(32)}}
        chosen=dict(C=np.eye(8),d=np.zeros(8))
        reads=[dict(phase=phase,wl=w,bl=b,conductance_siemens=.0001) for phase in ('pre','post') for w in range(8) for b in range(32) for _ in range(3)]
        vectors=[dict(**e,adc=analysis.reference(H,e['mask']).tolist()) for e in self.entries if e['phase']=='synthetic_lif']
        tests=analysis.lif_tests(vectors,runtime.SPEC,H,np.zeros(8),chosen,basis,reads,[True]*4)
        self.assertTrue(any(r['status']=='pass' and r['pattern']=='coincident' for r in tests))
        self.assertTrue(any(r.get('reason')=='expected_sequence_not_exciting' for r in tests))
        scales={r['scale'] for r in tests if r['group_id']==0 and r['pair']==[0,1]}
        self.assertEqual(len(scales),1)
        vectors=[v for v in vectors if not(v['group_id']==0 and v['pattern']=='coincident' and v['repeat']==0 and v['t']==0)]
        tests=analysis.lif_tests(vectors,runtime.SPEC,H,np.zeros(8),chosen,basis,reads,[True]*4)
        self.assertTrue(any(r.get('reason')=='incomplete_trajectory' for r in tests))
        silent=copy.deepcopy(runtime.SPEC);silent['synthetic_patterns']={'silent':[[] for _ in range(12)]}
        tests=analysis.lif_tests([],silent,H,np.zeros(8),chosen,basis,reads,[True]*4)
        self.assertTrue(all(r['status']=='inconclusive' for r in tests))

    def test_zero_anchor_prevents_fitted_offset_false_input(self):
        H=np.ones((8,32))*20;zero=np.zeros(8);settings=runtime.SPEC['calibration']
        anchors={():np.zeros((5,8)),**{(b,):np.full((5,8),20.) for b in range(32)}}
        selection={(0,1):np.full((5,8),40.)}
        good=dict(name='good',C=np.eye(8),d=np.zeros(8),eligible=True)
        bad=dict(name='bad',C=np.eye(8),d=np.ones(8)*10,eligible=True)
        chosen,scores,_=analysis.select([bad,good],selection,anchors,H,zero,settings)
        self.assertEqual(chosen['name'],'good');self.assertFalse(scores[0]['selection_pass'])
        self.assertFalse(scores[1]['selection_pass'])  # One measured mixed mask is incomplete selection.

    def test_noise_rejects_correct_mean_and_signed_pair_cancellation(self):
        H=np.ones((8,32))*20;candidate=dict(C=np.eye(8),d=np.zeros(8))
        values=np.full((5,8),40.);values[:,0]+=[-10,-5,0,5,10]
        points=analysis.assess(candidate,{(0,1):values},H,np.zeros(8),runtime.SPEC['calibration'])
        self.assertTrue(points[0]['mean_pass']);self.assertFalse(points[0]['repeatability_pass'])
        pairs=analysis.signed_assess(candidate,{(0,1):values},H,np.zeros(8),runtime.SPEC['calibration'])
        self.assertTrue(pairs[0]['mean_pass']);self.assertFalse(pairs[0]['repeatability_pass'])

    def test_partial_basis_preserves_observed_addresses(self):
        rows=analysis.partial_cells({():np.zeros((5,8)),(0,):np.ones((5,8))*20},[])
        self.assertEqual(len(rows),256)
        self.assertEqual(rows[0]['basis_count'],5);self.assertEqual(rows[0]['signal_adc'],20)
        self.assertEqual(rows[1]['status'],'unmeasured')

    def test_bad_selection_row_does_not_hide_other_rows(self):
        H=np.random.default_rng(7).uniform(20,80,(8,32));vectors=[]
        for e in self.entries:
            if e['phase'] not in ('basis','fit','selection','selection_anchors','test','test_anchors') or len(e['mask'])>4:continue
            adc=analysis.reference(H,e['mask']).tolist()
            if e['phase']=='selection':adc[7]+=100
            vectors.append(dict(**e,adc=adc))
        reads=[dict(phase=phase,wl=w,bl=b,conductance_siemens=.0001) for phase in ('pre','post') for w in range(8) for b in range(32) for _ in range(3)]
        result=analysis.analyze(dict(vectors=vectors,readbacks=reads,skips=[]),runtime.SPEC)
        self.assertFalse(result['selection_pass']);self.assertTrue(result['selection_by_wl'][0]['passed'])
        self.assertFalse(result['selection_by_wl'][7]['passed'])
        self.assertTrue(all(c['status']=='single_response_supported' for c in result['cells'] if c['wl']==0))

    def test_path_probe_uses_independent_test_anchors_without_diagnosis(self):
        vectors=[]
        for phase,mask,value in [('test_anchors',[],0),('test_anchors',[8],100),('test_anchors',[16],0),('input_path_probe',[8,16],140)]:
            for repeat in range(5):vectors.append(dict(phase=phase,mask=mask,adc=[value]*8,task_ids=list(range(8))))
        rows=analysis.path_probes(vectors,runtime.SPEC);row=rows[0]
        self.assertEqual(row['increment_adc'],40);self.assertEqual(row['epsilon_adc'],40)
        self.assertTrue(row['increment_distinguishable']);self.assertIsNone(row['physical_cause'])
        self.assertFalse(row['used_for_model_selection'])

    def test_lif_inclusive_negative_state_preserved(self):
        result=analysis.lif([-1.,1.5,0.])
        self.assertEqual(result['post'],[-1.,0.,0.]);self.assertEqual(result['spikes'],[0,1,0])


if __name__=='__main__':unittest.main()
