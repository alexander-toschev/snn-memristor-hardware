"""Focused public-I/O schedule tests; no native simulator opened."""
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'.deps')]
from experiments.cifar_confirmatory_20261020 import native
from tests.test_stable_cell_calibration import Board,Converters,Clock,Sink
import copy


class Tests(unittest.TestCase):
    def runner(self,board=None):
        clock=Clock()
        return native.Collector(board or Board(),Converters,Sink(),kind='characterize',monotonic=clock,sleep=clock.sleep)

    def test_characterization_bound_mapping_no_labels(self):
        c=self.runner();result=c.characterize()
        self.assertEqual(c.calls,1304);self.assertEqual(c.pulses,256);self.assertEqual(c.decisions,256)
        self.assertEqual(len(result['endpoints']),32)
        m=result['mapping']
        self.assertGreater(m['high_g_s'],m['low_g_s'])
        self.assertAlmostEqual(m['span_g_s'],m['high_g_s']-m['low_g_s'])
        self.assertFalse(m['selection_uses_labels'])

    def test_current_guard_and_invalid_read(self):
        board=Board();board.r[0][0]=1.
        c=self.runner(board);c.decision(0,0,1,3.)
        self.assertEqual(c.pulses,0);self.assertEqual(c.calls,4)
        class Bad(Board):
            def impact(self,command):return float('nan'),command['id'],100
        with self.assertRaises(ValueError):self.runner(Bad()).read_g(0,0)

    def test_full_cap_arithmetic(self):
        self.assertEqual(64*75*2*(3+16*5+3),native.SPEC['programming_max_impacts'])
        self.assertEqual(64*75*2*16,native.SPEC['programming_max_pulses'])

    def test_independent_raw_replay_and_corruption(self):
        from experiments.cifar_confirmatory_20261020.hybrid import Replay
        from tests.test_mnist_diagonal import Sink as FullSink
        sink=FullSink();clock=Clock()
        c=native.Collector(Board(),Converters,sink,kind='characterize',monotonic=clock,sleep=clock.sleep)
        expected=c.characterize()
        actual,_=Replay(iter(sink.events)).characterize()
        self.assertEqual(actual,expected)
        bad=copy.deepcopy(sink.events)
        next(r for r in bad if r['event']=='conductance_read')['measurement']['mean_g_s']*=2
        with self.assertRaises(ValueError):Replay(iter(bad)).characterize()
        bad=copy.deepcopy(sink.events)
        next(r for r in bad if r['event']=='request' and r['command_si']['vol']!=0)['command_si']['sign']^=1
        with self.assertRaises(ValueError):Replay(iter(bad)).characterize()


if __name__=='__main__':unittest.main()
