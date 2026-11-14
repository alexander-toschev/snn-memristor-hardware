"""Offline scope, scaled tolerance and measured-baseline regression checks."""
from __future__ import annotations
from pathlib import Path
import unittest
from unittest.mock import patch
from experiment_common.runtime import use_local_dependencies
use_local_dependencies(prefer_environment=True)
from manager.service import converters
from experiment_common.block_readout_validation import Leg
from experiment_common.exploratory_transfer import ScopedProgrammer, centered
from experiment_common.selective_metrology import Reading
from experiment_common.tiled_transfer import AdaptiveProgrammer, profiles, schedule


class ExploratoryTests(unittest.TestCase):
    def test_selected_bl31_and_reject_outside_scope_before_io(self) -> None:
        programmer = ScopedProgrammer.__new__(ScopedProgrammer)
        programmer.allowed = frozenset({(0,31)})
        with patch.object(AdaptiveProgrammer, 'call', return_value=(1,1000.,5400)) as native:
            self.assertEqual(programmer.read_r(0,31),1000.)
            with self.assertRaises(ValueError):
                programmer.read_r(0,30)
            native.assert_called_once()

    def test_scaled_tolerance_and_span_restored(self) -> None:
        programmer = ScopedProgrammer.__new__(ScopedProgrammer)
        programmer.span = 1.
        observed: list[float] = []
        def fake(instance: AdaptiveProgrammer, wl: int, bl: int, goal: float, **metadata: object) -> dict[str, object]:
            observed.append(instance.span)
            return {'wl':wl,'bl':bl,'target_g_s':goal}
        with patch.object(AdaptiveProgrammer,'program',autospec=True,side_effect=fake):
            result = programmer.program(0,31,.1,conductance_scale=.2)
        self.assertEqual(observed,[.2])
        self.assertEqual(programmer.span,1.)
        self.assertEqual(result['target_tolerance_s'],.004)

    def test_baseline_formulas_use_identical_native_samples(self) -> None:
        def samples(adc: int) -> tuple[Reading,...]:
            return tuple(Reading(task_id=i,resistance_ohm=1000.,adc=adc) for i in range(4))
        leg = Leg(0,(),samples(100),samples(140),samples(120),tuple([0]*32),tuple([1]*32),tuple([0.]*32),tuple([.05]*32),0.)
        self.assertEqual(centered(leg,converters,'before_only',True),40.)
        self.assertEqual(centered(leg,converters,'bracket_centered',True),30.)
        expected = converters.convert_adc_to_volt(1,14,5,140)-(converters.convert_adc_to_volt(1,14,5,100)+converters.convert_adc_to_volt(1,14,5,120))/2
        self.assertEqual(centered(leg,converters,'bracket_centered'),expected)

    def test_map_covers1600_and_read_budget(self) -> None:
        path=Path(__file__).resolve().parents[1]/'experiments/mnist_expanded_exploratory_20261114/map148.json'
        selected=profiles(path)
        cells={(w,b) for p in selected for w in (p.block.plus_wl,p.block.minus_wl) for b in p.block.bls}
        self.assertEqual(len(cells),148)
        self.assertTrue(any(b==31 for _,b in cells))
        plan=schedule(selected)
        self.assertEqual(len({(a.filter,a.tap) for a in plan}),1600)
        self.assertEqual(max(a.load for a in plan)+1,22)
        calls=sum((1<<len(p.block.bls))-1 for p in selected)*8*2*44
        self.assertEqual(calls,73920)
        self.assertLessEqual(calls+256,74518)


if __name__ == '__main__':
    unittest.main()
