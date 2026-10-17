import copy
import json
import tempfile
from pathlib import Path
import unittest
import analyze_fresh_stability as a

class Tests(unittest.TestCase):
    def sample(self,value,task):return dict(voltage_v=value,task_id=task)
    def evidence(self):
        pair=dict(post_feedback=[]);anchors=[];final=[]
        for r in range(5):
            pair['post_feedback'].append(dict(measured_rows=[dict(side=side,
                zero_before=self.sample(1,r*20+j*3),signals=[self.sample(4+j,r*20+j*3+1)],zero_after=self.sample(3,r*20+j*3+2))
                for j,side in enumerate(a.SIDES)]))
            anchors.append([dict(zero=self.sample(2,r*20+10+j),singles=[self.sample(5+j,r*20+12+j)]) for j in range(2)])
            final.append([dict(zero=self.sample(2,r*20+14+j),singles=[self.sample(6+j,r*20+16+j)]) for j in range(2)])
        return pair,anchors,final
    def test_own_bracket_vs_shared_zero_arithmetic_and_ids(self):
        p,b,f=self.evidence();r=a.leg_evidence(p,b,f,0)
        for j,side in enumerate(a.SIDES):
            self.assertEqual(r[side]['own']['mean_v'],2+j)
            self.assertEqual(r[side]['post_all']['mean_v'],3+j)
            self.assertEqual(r[side]['final']['mean_v'],4+j)
            self.assertEqual(r[side]['post_all_minus_own_v'],1)
            self.assertEqual(r[side]['final_minus_post_all_v'],1)
            self.assertEqual(len(r[side]['own_task_ids']),5)
    def test_missing_readback_and_nonfinite_rejected(self):
        p,b,f=self.evidence();p['post_feedback'].pop()
        with self.assertRaises(ValueError):a.leg_evidence(p,b,f,0)
        with self.assertRaises(ValueError):a.moments([1,1,1,1,float('nan')])
    def test_cell_map_includes256_and_excluded_not_dead(self):
        rows=a.all_cell_records({},[8]);self.assertEqual(len(rows),256)
        self.assertEqual(sum(x['observation_status']=='excluded_by_protocol' for x in rows),8)
        self.assertEqual(sum(x['observation_status']=='not_observed' for x in rows),248)
        svg=a.evidence_map(dict(source='test',session_id='fake',cells=rows))
        self.assertIn('Цвет не оценивает точность',svg)
        self.assertEqual(svg.count('<title>'),256)
    def test_excluded_but_observed_rejected(self):
        with self.assertRaises(ValueError):a.all_cell_records({(0,8):dict(probes=[{}])},[8])
    def test_pair_cancellation_does_not_hide_two_leg_changes(self):
        p,b,f=self.evidence();r=a.leg_evidence(p,b,f,0)
        self.assertEqual(r[a.SIDES[0]]['post_all_minus_own_v']-r[a.SIDES[1]]['post_all_minus_own_v'],0)
        self.assertEqual(r[a.SIDES[0]]['post_all_minus_own_v'],1)
    def test_global_nrmse_uses_all_weights(self):
        rows=[dict(weight=1,post_own=2,post_all_raw=1,post_cifar_raw=1,effective=1),dict(weight=3,post_own=3,post_all_raw=3,post_cifar_raw=3,effective=3)]
        self.assertAlmostEqual(a.weight_metrics(rows)['post_own']['nrmse'],(1/10)**.5)
    def test_incomplete_trace_comparison_rejected(self):
        with self.assertRaises(ValueError):a.compare_traces([],[])
    def test_known_auditor_cannot_be_relabelled_other_protocol(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            root=Path(tmp).resolve();(root/'session').mkdir()
            (root/'session/manifest.json').write_text(json.dumps({'protocol_id':'wrong-protocol'}))
            (root/'audit.json').write_text(json.dumps(dict(audit_status='passed',errors=[],auditor_sha256=next(iter(a.AUDITORS)))))
            with self.assertRaisesRegex(ValueError,'Auditor/protocol mismatch'):a.analyze(root,'session','audit.json')

    def test_path_escape_rejected(self):
        with self.assertRaises(ValueError):a.local(Path.cwd(),'../escape')

if __name__=='__main__':unittest.main()
