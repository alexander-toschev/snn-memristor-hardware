"""Small offline fixtures for exploratory singleton analysis."""
import copy
import json
from pathlib import Path
import unittest
import explore_singleton_mvm as review


class SingletonReviewTests(unittest.TestCase):
    def fixture(self):
        vectors=[]
        for phase in ('basis','selection_anchors','test_anchors'):
            for mask in [[]]+[[b] for b in range(32)]:
                for repeat in range(5):vectors.append(dict(phase=phase,mask=mask,repeat=repeat,adc=[20 if mask else 0]*8,task_ids=list(range(8))))
        reads=[dict(phase=phase,wl=w,bl=b,conductance_siemens=.0001) for phase in ('pre','post') for w in range(8) for b in range(32) for _ in range(3)]
        plan=json.loads((Path(__file__).resolve().parents[1]/'configs/calibrated_mvm.json').read_bytes())
        return dict(vectors=vectors,readbacks=reads),plan

    def test_single_bad_anchor_does_not_reject_other_cells(self):
        data,plan=self.fixture()
        for row in data['vectors']:
            if row['phase']=='selection_anchors' and row['mask']==[31]:row['adc'][7]=50
        report=review.evaluate(data,plan)
        self.assertEqual(report['counts']['exploratory_singleton_supported'],255)
        self.assertEqual(report['cells'][-1]['status'],'not_supported')
        self.assertFalse(report['simultaneous_mvm_qualified'])
        self.assertTrue(all(r['status']=='inconclusive' for r in report['recorded_dispersed_lif']))

    def test_zero_noise_floor_and_identity_offset_are_not_relaxed(self):
        basis=review.level([4]*5,[0]*5)
        self.assertFalse(basis['quality']);self.assertEqual(basis['snr'],4)
        strong=review.level([20]*5,[0]*5)
        shifted=review.anchor_test([30]*5,[10]*5,strong)
        self.assertEqual(shifted['own_zero_subtracted_error_adc'],0)
        self.assertFalse(shifted['passes']);self.assertEqual(shifted['identity_error_adc'],10)

    def test_changed_G_and_missing_anchor_preserve_all256_records(self):
        data,plan=self.fixture()
        for row in data['readbacks']:
            if row['wl']==0 and row['bl']==0 and row['phase']=='post':row['conductance_siemens']=.0002
        data['vectors']=[v for v in data['vectors'] if not(v['phase']=='test_anchors' and v['mask']==[1] and v['repeat']==4)]
        report=review.evaluate(data,plan)
        self.assertEqual(len(report['cells']),256)
        self.assertEqual(report['cells'][0]['reasons'],['conductance_consistency'])
        self.assertEqual(report['counts']['unmeasured'],8)
        self.assertEqual(report['counts']['exploratory_singleton_supported'],247)

    def test_negative_signed_signal_is_valid_but_offset_drift_not_hidden(self):
        data,_=self.fixture()
        for row in data['vectors']:
            if row['mask']==[0]:row['adc'][0]=10;row['adc'][1]=30
        evidence=review.signed_anchor_evidence(data['vectors'],0,1,0)
        self.assertTrue(evidence['supported']);self.assertEqual(evidence['phases']['basis']['signal_adc'],-20)
        for row in data['vectors']:
            if row['mask']==[] and row['phase']=='selection_anchors':row['adc'][0]=10
        evidence=review.signed_anchor_evidence(data['vectors'],0,1,0)
        self.assertFalse(evidence['supported'])


if __name__=='__main__':unittest.main()
