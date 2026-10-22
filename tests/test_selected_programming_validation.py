from __future__ import annotations

from dataclasses import asdict
import unittest
from typing import cast, TYPE_CHECKING

from experiment_common.durable import Writer
from experiment_common.programming_validation import Collector, CELLS, assess_cells, assess_pairs, recipe, pair_states
from experiments.stable_cell_calibration_20261020 import metrics as original
from experiments.stable_cell_calibration_20261020 import plan as original_plan
from tests.test_selective_metrology import Clock, Converters

if TYPE_CHECKING:
    from manager.board import Connector


class Sink:
    def write(self, value):
        pass

    def checkpoint(self, **values):
        pass


class Native:
    def __init__(self, flat=False, resistance=100000.):
        self.state = {cell: 0 for cell in CELLS}
        self.flat, self.resistance = flat, resistance
        self.count = 0
        self.writes = 0

    def impact(self, task):
        self.count += 1
        if task['mode_flag'] == 7:
            if task['vol']:
                self.writes += 1
                cell = task['wl'], task['bl']
                self.state[cell] = 0 if task['sign'] == 0 else self.state[cell] + 10
            return self.resistance, task['id'], 1000
        active = [b for b, v in enumerate(task['vol']) if v]
        return 1., 0, (100 + (0 if self.flat else self.state[task['wl'], active[0]])) if active else 0


def runner(flat=False, resistance=100000.):
    native, clock = Native(flat, resistance), Clock()
    collector = Collector(cast('Connector', native), Converters(), cast(Writer, Sink()), 'simulator',
                          lambda _: None, clock=clock.now, sleep=clock.sleep)
    return collector, native, clock


class ProgrammingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.collector, cls.native, cls.clock = runner()
        cls.result = cls.collector.run()

    def test_complete_budget_and_schedule(self):
        self.assertEqual(self.result['status'], 'completed')
        self.assertTrue(self.result['qualified'])
        self.assertEqual(self.collector.calls, 24320)
        self.assertEqual(self.collector.pulses, 1280)
        self.assertEqual(self.collector.phase_calls, dict(A=17664, B=6656))
        self.assertEqual(self.collector.phase_pulses, dict(A=512, B=768))
        self.assertEqual(len(self.collector.points), 416)
        self.assertEqual(len(self.collector.pairs), 64)
        self.assertEqual(self.clock.value, 600)
        for state in ('R', 'M', 'H'):
            for incremental in (False, True):
                self.assertEqual(recipe(state, incremental), original_plan.recipe(state, incremental))
        for config in ('plus', 'minus', 'zero', 'checkerboard'):
            for i in range(8):
                self.assertEqual(pair_states(config, i), original_plan.pair_states(config, i))

    def test_metrics_equal_original_formulas(self):
        points = self.collector.points
        old_points = [dict(id=i, wl=p.wl, bl=p.bl, cycle=p.cycle, state=p.state, role=p.role,
                           summary=dict(status='measured', signal=dict(mean=p.mean, sd=p.sd),
                                        zero_difference=dict(mean=p.zero_mean), zero_shift_limit=p.zero_limit))
                      for i, p in enumerate(points)]
        updates = [asdict(u) for u in self.collector.updates]
        old = [original.cell_summary(w, b, old_points, updates) for w, b in CELLS]
        new = assess_cells(points, self.collector.updates)
        for a, b in zip(new, old):
            for key, value in a.items():
                self.assertEqual(value, b[key], key)
        pair_rows = [dict(pair=[0, 7], bl=p.bl, configuration=p.configuration, role=p.role,
                          status='measured', differences_adc=[p.mean] * 10)
                     for p in self.collector.pairs]
        old_b = original.qualify_pairs(dict(pairs=[[0, 7]], bls=[2, 3, 6, 10, 11, 12, 14, 15]), old, pair_rows)
        self.assertEqual(assess_pairs(self.collector.pairs, new), old_b)

    def test_flat_stability_is_not_programmability_and_no_automatic_expansion(self):
        collector, native, _ = runner(flat=True)
        result = collector.run()
        self.assertEqual(result['status'], 'no_stable_rectangle')
        self.assertEqual(collector.phase_calls['B'], 0)
        self.assertEqual(collector.calls, 17664)
        self.assertTrue(all('insufficient_reproducible_span' in r['hard_reasons'] for r in collector.cells))
        self.assertEqual(set(native.state), set(CELLS))

    def test_current_guard_skips_set_and_native_invalid_stops(self):
        collector, native, _ = runner(resistance=1000.)
        collector.pulse(0, 2, 1, 3., 0)
        self.assertEqual(collector.calls, 1)
        self.assertEqual(native.writes, 0)
        self.assertFalse(collector.updates[0].admitted)
        collector, native, _ = runner(resistance=0.)
        with self.assertRaises(ValueError):
            collector.pulse(0, 2, 1, 1.2, 0)
        self.assertEqual(native.writes, 0)
        with self.assertRaises(ValueError):
            collector.call(dict(mode_flag=7, wl=1, bl=2, vol=0., sign=0, pulse_width=0.))
        collector.calls = 24320
        with self.assertRaises(ValueError):
            collector.read(0, 2)


if __name__ == '__main__':
    unittest.main()
