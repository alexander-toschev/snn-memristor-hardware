"""Offline diagnostic boundary/continuation tests: no native Connector."""
import copy
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'.deps')]
from experiments.stable_cell_diagnostic_20261020 import experiment,reselect
from experiments.stable_cell_calibration_20261020 import metrics,plan
from tests.test_stable_cell_calibration import Board,Converters,Sink,Clock


def cells():
    return [dict(wl=w,bl=b,eligible=False,hard_eligible=w<2 and b<8,
                 span_adc=9.,worst_normalized_error=1.,reasons=['original_quality_flag'],
                 hard_reasons=[]) for w,b in plan.CELLS]


def paired(value=3.):
    return [dict(pair=[0,1],bl=0,configuration=c,role=r,status='measured',
                 differences_adc=[-value if c=='minus' else value]*10)
            for c in ('plus','minus','zero','checkerboard') for r in ('immediate','delayed')]


class DiagnosticTests(unittest.TestCase):
    def test_quantization_boundary_original_unchanged(self):
        original=dict(eligible=False,hard_reasons=['insufficient_reproducible_span'],span_adc=3.)
        before=copy.deepcopy(original);r=reselect.diagnostic_cell(original)
        self.assertTrue(r['hard_eligible']);self.assertFalse(r['eligible']);self.assertEqual(original,before)
        self.assertEqual(r['original_hard_reasons'],before['hard_reasons'])
        for span in (2.,1.,float('nan')):
            self.assertFalse(reselect.diagnostic_cell(dict(original,span_adc=span))['hard_eligible'])
        self.assertFalse(reselect.diagnostic_cell(dict(original,hard_reasons=['missing_required_state_checkpoint']))['hard_eligible'])

    def test_b_resolution_sign_noise_and_original_verdict(self):
        selected=dict(pairs=[[0,1]],bls=[0])
        r=reselect.pair_diagnostic(selected,cells(),paired())
        self.assertTrue(r['mandatory_pass']);self.assertFalse(r['qualified'])
        self.assertFalse(r['original_quality_evaluation']['mandatory_pass'])
        for rows in (paired(2.),paired(-3.),paired(0.)):
            self.assertFalse(reselect.pair_diagnostic(selected,cells(),rows)['mandatory_pass'])
        rows=paired();rows[0]['differences_adc']=[-5.,11.]*5
        self.assertFalse(reselect.pair_diagnostic(selected,cells(),rows)['mandatory_pass'])

    def test_b_missing_duplicate_count_nonfinite_fail(self):
        selected=dict(pairs=[[0,1]],bls=[0])
        rows=paired()
        for bad in (rows[:-1],rows+[rows[0]]):
            self.assertFalse(reselect.pair_diagnostic(selected,cells(),bad)['mandatory_pass'])
        for values in ([3.]*9,[float('nan')]*10,[]):
            bad=paired();bad[0]['differences_adc']=values
            self.assertFalse(reselect.pair_diagnostic(selected,cells(),bad)['mandatory_pass'])

    def test_finite_fresh_bc_no_a_and_safe_close(self):
        cs=cells();selection=metrics.select_rectangle(cs)
        candidate=dict(cells=cs,selection=selection,original_verdict='no_stable_rectangle')
        clock=Clock();board=Board();sink=Sink()
        c=experiment.Continuation(board,Converters,sink,'simulator',candidate,monotonic=clock,sleep=clock.sleep)
        r=c.run_continuation()
        self.assertEqual(r['status'],'completed',r['error']);self.assertFalse(r['qualified'])
        self.assertEqual(r['phase_impacts']['A'],0);self.assertEqual(r['checkpoints'],[])
        self.assertEqual(r['pulses'],768);self.assertLessEqual(r['impacts'],c.caps['maximum_impacts'])
        self.assertEqual(r['explicit_wait_requested_s'],240.)
        self.assertEqual(len(r['cells']),256);self.assertEqual(sink.events['matrix_frozen'],1)
        self.assertTrue(board.closed);self.assertFalse(r['phase_A_repeated'])

if __name__=='__main__':unittest.main()
