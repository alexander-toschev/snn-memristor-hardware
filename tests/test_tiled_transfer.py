from pathlib import Path
import unittest
import numpy as np
from experiment_common.tiled_transfer import profiles, schedule, compose


class TiledTransferTest(unittest.TestCase):
    def test_expanded_encoding_preserves_signed_weights_without_clipping(self) -> None:
        import json
        selected = profiles(Path('experiments/mnist_tiled_transfer_20261022/map44.json'))
        source = json.loads(Path('experiments/mnist_tiled_transfer_20261022/assets/learned_conductances.json').read_text())
        goals = np.asarray(source['conductances_s'])
        raw = json.loads(Path('experiments/mnist_tiled_transfer_20261022/map44.json').read_text())
        self.assertEqual(sum(2*len(p.block.bls) for p in selected), 44)
        plan = schedule(selected)
        self.assertEqual(max(a.load for a in plan)+1, 73)
        self.assertEqual(len({(a.filter,a.tap) for a in plan}), 1600)
        for p, row in zip(selected, raw['selected_blocks']):
            physical = goals*p.conductance_scale+p.conductance_offset
            np.testing.assert_allclose((physical[:,:,0]-physical[:,:,1])/p.conductance_scale,
                                      goals[:,:,0]-goals[:,:,1], atol=1e-18)
            if row['new']:
                self.assertTrue(p.global_bias)
                self.assertFalse(p.require_resolved_legs)
                self.assertGreaterEqual(float(physical.min()), row['calibrated_range'][0]-1e-18)
                self.assertLessEqual(float(physical.max()), row['calibrated_range'][1]+1e-18)

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
