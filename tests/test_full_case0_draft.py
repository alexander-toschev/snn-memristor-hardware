"""Draft-only structural and lossless storage tests. No model or board."""
import copy
from pathlib import Path
import tempfile
import unittest

from experiments.cifar_full_case0_20261019 import plan, reconstruct, trace


class DraftTests(unittest.TestCase):
    def make_rows(self, image):
        result = []
        for task, meta in enumerate(plan.position_commands(image,0,0,0),1):
            adc = 100 + ((10 if meta['wl'] % 2 == 0 else 5)*len(meta['mask']) if meta['role'] == 'signal' else 0)
            result.append(dict(event='native_mode10', task_id=task, **meta, status='ok',
                command_si=dict(mode_flag=10, vol=[.05 if bl in meta['mask'] else 0. for bl in range(32)],wl=meta['wl'],id=0),
                native_result=[0.,0,adc]))
        return result

    def test_planner_independent_mask_reconstruction(self):
        image = plan.fixture()['image_chw_uint8']
        for tile,y,x in [(0,0,0),(1,27,27),(2,13,5)]:
            actual = [(t,s,m) for t,parts in enumerate(plan.masks(image,tile,y,x)) for s,m in enumerate(parts)]
            self.assertEqual(actual,list(reconstruct.expected_signals(image,tile,y,x)))
            self.assertTrue(all(len(m)<=4 for _,_,m in actual))

    def test_fixed_fixture_bounds(self):
        result = plan.bounds(plan.fixture()['image_chw_uint8'])
        self.assertEqual(result['inference_mode10'],5133568)
        self.assertEqual(result['dense_tensor_bytes'],7225344)
        self.assertEqual(result['all_impacts_upper'],14457648)
        self.assertTrue(plan.SPEC['hardware_qualification_required'])

    def test_fresh_split_terms_sum_before_lif_and_zero_subtraction(self):
        # 32 bright pixels require eight distinct physical vectors at t0.
        image = [255]*3072
        rows = self.make_rows(image)
        matrix = [[float(i==j) for j in range(8)] for i in range(8)]
        result = reconstruct.Replay(rows).position(image,0,0,0,matrix,.125)
        expected = 8*(round(140*5/16383,5)-round(120*5/16383,5))/.125
        self.assertAlmostEqual(result[0][0],expected)
        self.assertTrue(all(v==0 for row in result[1:] for v in row))
        self.assertEqual(len(rows),(8+11)*8+16)

    def test_changed_command_duplicate_id_and_bad_adc_fail(self):
        image=[0]*3072
        matrix=[[float(i==j) for j in range(8)] for i in range(8)]
        for field,value in [('task_id',99),('native_result',[0.,0,16383]),('status','error')]:
            rows=self.make_rows(image)
            rows[10][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                reconstruct.Replay(rows).position(image,0,0,0,matrix,.125)
        rows=self.make_rows(image)
        rows[10]['command_si']['vol'][31]=.1
        with self.assertRaises(ValueError):
            reconstruct.Replay(rows).position(image,0,0,0,matrix,.125)

    def test_truncated_stream_cannot_export_full_tensor(self):
        case=plan.fixture()
        with self.assertRaisesRegex(ValueError,'Incomplete'):
            reconstruct.reconstruct([],case['image_chw_uint8'],case['image_chw_uint8_sha256'],'synthetic_test')

    def test_lossless_chunks_and_corruption(self):
        records=[{'n':n,'data':'abc'*10} for n in range(15)]
        with tempfile.TemporaryDirectory() as temporary:
            target=Path(temporary)/'raw'
            writer=trace.Writer(target,limit=120)
            for row in records:
                writer.write(row)
            index=writer.finish()
            self.assertGreater(len(index['parts']),1)
            self.assertEqual(list(trace.records(target)),records)
            part=target/index['parts'][0]['path']
            part.write_bytes(part.read_bytes()+b'x')
            with self.assertRaises(ValueError):
                list(trace.records(target))


if __name__=='__main__':
    unittest.main()
