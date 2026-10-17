"""Offline checks against actual MemriBoard converters; never imports a driver."""

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
for relative in (".deps", "vendor/MemriBoard"):
    path = ROOT / relative
    if path.is_dir():
        sys.path.insert(0, str(path))

from manager.service import converters
import native_conversions as conversion

BOARD = {"dac_bit": 12, "vol_ref_dac": 5, "gain": 11, "res_load": 3000,
         "adc_bit": 14, "vol_ref_adc": 5, "res_switches": 10, "sum_gain": 330}


class NativeConversionTests(unittest.TestCase):
    def test_voltage_preserves_native_quantization(self):
        for voltage, code, decoded in ((0, 0, 0), (.1, 82, .1), (.3, 246, .3), (3, 2457, 3)):
            result = conversion.voltage(voltage, BOARD, converters)
            self.assertEqual(result["dac_code"], code)
            self.assertEqual(result["native_decoded_v"], decoded)

    def test_voltage_rejects_invalid_and_out_of_dac_range(self):
        for value in (-1, float("nan"), float("inf"), True, 100):
            with self.subTest(value=value), self.assertRaises(ValueError):
                conversion.voltage(value, BOARD, converters)

    def test_read_voltage_is_explicit_and_native_rounding_preserved(self):
        at_point_one = conversion.readback(686, BOARD, converters, read_voltage=.1)
        at_point_three = conversion.readback(686, BOARD, converters, read_voltage=.3)
        self.assertEqual(at_point_one["resistance_ohm"], 12753.03)
        self.assertEqual(at_point_three["resistance_ohm"], 44279.1)
        self.assertEqual(at_point_one["q"], 330 / 12753.03)
        self.assertEqual(at_point_one["conductance_siemens"], 1 / 12753.03)
        self.assertNotEqual(at_point_one["q"], converters.convert_res_to_weight(3000, 12753.03))

    def test_adc_invalid_clamp_and_sentinel_not_used_as_measurement(self):
        for value in (0, 1, 19, -1, 16383, 16384, True, 1.5):
            with self.subTest(adc=value), self.assertRaises(ValueError):
                conversion.readback(value, BOARD, converters, read_voltage=.3)
        with self.assertRaisesRegex(ValueError, "replacement"):
            conversion.readback(10000, BOARD, converters, read_voltage=.1)

    def test_mvm_zero_allowed_and_gain_one(self):
        self.assertEqual(conversion.mvm_output(0, BOARD, converters), 0)
        self.assertEqual(conversion.mvm_output(1000, BOARD, converters), .30519)
        self.assertEqual(conversion.mvm_output(1000, BOARD, converters),
                         converters.convert_adc_to_volt(1, 14, 5, 1000))
        self.assertNotEqual(conversion.mvm_output(1000, BOARD, converters),
                            converters.convert_adc_to_volt(11, 14, 5, 1000))

    def test_mvm_saturation_and_wrong_types_rejected(self):
        for value in (-1, 16383, 16384, True, 1.1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                conversion.mvm_output(value, BOARD, converters)


if __name__ == "__main__":
    unittest.main()
