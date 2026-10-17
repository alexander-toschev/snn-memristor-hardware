import unittest
import json
import hashlib
import tempfile
from pathlib import Path
from audit_cifar_feedback_diagnostic import (fixed_decode,fixed_effective_weights,fixed_calibration,
    raw_lines,joined_raw,coverage,fidelity,CompactChecks)


class DiagnosticDecodeTests(unittest.TestCase):
    def test_negative_response_remains_negative_without_alpha_sign_flip(self):
        self.assertEqual(fixed_decode([.1,.3],1.),-.19999999999999998)

    def test_backend_gain_is_fixed_not_learned(self):
        self.assertAlmostEqual(fixed_decode([.006,.003],.03),.1)
        with self.assertRaises(ValueError):fixed_decode([.006,.003],.034)

    def test_effective_reference_has_same_fixed_units(self):
        anchors=[[{'zero':{'voltage_v':.01},'singles':[{'voltage_v':.04}]},
                  {'zero':{'voltage_v':.02},'singles':[{'voltage_v':.03}]}]]*5
        self.assertAlmostEqual(fixed_effective_weights([[1.,0.],[0.,1.]],anchors,.03)[0],2/3)

    def test_negative_fitted_alpha_is_diagnostic_not_decoder_or_gate(self):
        result=fixed_calibration([.2],[[ -.1 ]]*5,1.)
        self.assertLess(result['fitted_alpha_diagnostic']['alpha_v_per_weight'],0)
        self.assertFalse(result['fitted_alpha_diagnostic']['decode_usable'])
        self.assertTrue(result['decode_usable'])
        self.assertEqual(result['effective_weights'],[-.1])
        self.assertFalse(fidelity([.2],[[-.1]]*5,1.)['physical_fidelity_pass'])

    def test_zero_fitted_alpha_still_has_finite_fixed_decoder(self):
        result=fixed_calibration([.2],[[0.]]*5,.03)
        self.assertTrue(result['decode_usable']);self.assertEqual(result['effective_weights'],[0.])

    def test_offdiagonal_matrix_is_applied_before_signed_difference(self):
        anchors=[[{'zero':{'voltage_v':.01},'singles':[{'voltage_v':.04}]},
                  {'zero':{'voltage_v':.02},'singles':[{'voltage_v':.03}]}]]*5
        self.assertAlmostEqual(fixed_effective_weights([[2.,1.],[.5,3.]],anchors,1.)[0],.025)

    def test_coverage_does_not_count_unwritten_selected_leg_as_SET(self):
        r=dict(reset_candidates=[dict(cell=[0,8],eligible=True),dict(cell=[1,8],eligible=True),dict(cell=[0,1],eligible=False)],
            effective_route=[dict(positive_cell=[0,8],negative_cell=[1,8])],technical_completed=True,
            pairs=[dict(programming_pulse_attempts=0,active_set_side='positive_cell',positive_cell=[0,8]),
                   dict(programming_pulse_attempts=1,active_set_side='negative_cell',negative_cell=[1,8])])
        c=coverage([r]);self.assertEqual(c['set_attempted']['addresses'],[[1,8]])
        self.assertEqual(c['represented']['count'],2);self.assertEqual(c['probed_but_not_selected'],[[0,1]])
        self.assertTrue(c['coverage_does_not_prove_programmability'])


class RawPartTests(unittest.TestCase):
    def setUp(self):
        base=Path(__file__).resolve().parents[1]/'.tmp';base.mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=base);self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.payloads=[b'{"event":"request"}\n',b'{"event":"result"}\n'];self.index=self.write_parts(self.payloads)

    def write_parts(self,payloads):
        parts=[]
        for i,data in enumerate(payloads):
            name='raw.jsonl' if i==0 else f'raw.{i:05d}.jsonl'
            (self.root/name).write_bytes(data)
            parts.append(dict(path=name,bytes=len(data),records=data.count(b'\n'),sha256=hashlib.sha256(data).hexdigest()))
        joined=b''.join(payloads)
        index=dict(parts=parts,bytes=len(joined),records=joined.count(b'\n'),sha256=hashlib.sha256(joined).hexdigest(),maximum_part_bytes=33554432)
        self.save_index(index);return index

    def save_index(self,index):
        (self.root/'raw_parts.json').write_text(json.dumps(index),encoding='utf-8')

    def test_request_result_pair_can_cross_part_boundary(self):
        with joined_raw(self.root/'raw.jsonl') as stream:
            self.assertEqual(json.loads(stream.readline())['event'],'request')
            self.assertEqual(json.loads(stream.readline())['event'],'result')
            self.assertEqual(stream.readline(),'')

    def test_mutated_part_rejected(self):
        (self.root/'raw.00001.jsonl').write_bytes(b'{"event":"request"}\n')
        with self.assertRaisesRegex(ValueError,'hash/size'):list(raw_lines(self.root/'raw.jsonl'))

    def test_reordered_parts_rejected(self):
        self.index['parts'].reverse();self.save_index(self.index)
        with self.assertRaisesRegex(ValueError,'order/names'):list(raw_lines(self.root/'raw.jsonl'))

    def test_extra_unindexed_part_rejected(self):
        (self.root/'raw.00002.jsonl').write_bytes(b'{}\n')
        with self.assertRaisesRegex(ValueError,'Unindexed'):list(raw_lines(self.root/'raw.jsonl'))

    def test_global_hash_cannot_be_replaced_by_individual_valid_hashes(self):
        self.index['sha256']='0'*64;self.save_index(self.index)
        with self.assertRaisesRegex(ValueError,'stream commitment'):list(raw_lines(self.root/'raw.jsonl'))

    def test_incomplete_final_line_rejected_even_with_correct_hash(self):
        self.write_parts([b'{"event":"request"}',b'{"event":"result"}\n'])
        with self.assertRaisesRegex(ValueError,'complete canonical'):list(raw_lines(self.root/'raw.jsonl'))

    def test_relative_path_escape_rejected_before_read(self):
        self.index['parts'][0]['path']='../raw.jsonl';self.save_index(self.index)
        with self.assertRaisesRegex(ValueError,'order/names'):list(raw_lines(self.root/'raw.jsonl'))


if __name__=='__main__':unittest.main()
