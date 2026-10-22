from pathlib import Path
import unittest
import numpy as np
from experiment_common.tiled_transfer import profiles, schedule, compose


class TiledTransferTest(unittest.TestCase):
    def test_native_voltage_rounding_is_per_sample(self) -> None:
        from experiment_common.runtime import use_local_dependencies
        use_local_dependencies(prefer_environment=True)
        from manager.service import converters
        from experiment_common.block_readout_validation import Leg
        from experiment_common.tiled_transfer import voltage_centered
        from experiment_common.selective_metrology import Reading
        before = tuple(Reading(task_id=i, resistance_ohm=1000., adc=1001) for i in range(4))
        signal = tuple(Reading(task_id=i, resistance_ohm=1000., adc=1103) for i in range(4))
        leg = Leg(0, (), before, signal, before, (), (), (), (), 0.)
        actual = voltage_centered(leg, converters)
        expected = converters.convert_adc_to_volt(1, 14, 5, 1103) - converters.convert_adc_to_volt(1, 14, 5, 1001)
        self.assertAlmostEqual(actual, expected)
        self.assertGreater(abs(actual - 102 * converters.convert_adc_to_volt(1, 14, 5, 1)), .0001)

    def test_all_coefficients_once_and_physical_slots_unique(self) -> None:
        selected = profiles(Path('reports/2026-10-22/block-readout-validation/STABLE_BLOCKS.json'))
        plan = schedule(selected)
        self.assertEqual(len(plan), 1600)
        self.assertEqual({(a.filter, a.tap) for a in plan}, {(f, t) for f in range(64) for t in range(25)})
        self.assertEqual(len({(a.load, a.block, a.slot) for a in plan}), 1600)
        self.assertEqual(max(a.load for a in plan), 88)

    def test_cross_filter_block_does_not_mix_outputs(self) -> None:
        from experiment_common.tiled_transfer import Assignment
        assignments = (Assignment(0, 'test', 0, 0, 24), Assignment(0, 'test', 1, 1, 0), Assignment(0, 'test', 2, 1, 1))
        keys = np.asarray([0, 1, 2, 3, (1 << 24) | 3], dtype=np.uint32)
        # Deliberately NON-additive empirical response. The code must index
        # measured subset6=123, not synthesize singleton2+singleton4=60.
        operators = [dict(block='test', lut_difference_g_s=[0., 10., 20., 30., 40., 50., 123., 999.])]
        target = np.zeros((len(keys), 64), dtype=np.float32)
        compose(keys, assignments, operators, 1., target)
        np.testing.assert_array_equal(target[:, 0], [0, 0, 0, 0, 10])
        np.testing.assert_array_equal(target[:, 1], [0, 20, 40, 123, 123])
        self.assertFalse(np.any(target[:, 2:]))


if __name__ == '__main__':
    unittest.main()
