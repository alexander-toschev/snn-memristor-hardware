"""Experiment-side conversions using the unmodified MemriBoard converters.

The caller supplies manager.service.converters after ordinary import and source
fingerprint checks. No hardware/driver imports or calls occur in this module.
"""

import math


def voltage(value, board, converters):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("Voltage must be finite and nonnegative; polarity is separate")
    code = converters.convert_volt_to_dac(board["dac_bit"], board["vol_ref_dac"], value)
    if type(code) is not int or not 0 <= code < 2**board["dac_bit"]:
        raise ValueError("Native DAC code is outside the configured DAC range")
    decoded = converters.convert_dac_to_volt(board["dac_bit"], board["vol_ref_dac"], code)
    return {"requested_v": value, "dac_code": code, "native_decoded_v": decoded}


def readback(raw_adc, board, converters, *, read_voltage):
    if type(raw_adc) is not int or not 1 <= raw_adc <= 2**board["adc_bit"]-2:
        raise ValueError("Invalid readback ADC; retain the original record")
    if raw_adc < 20:
        raise ValueError("Native a2r would clamp ADC below 20; retain it without derived R")
    if type(read_voltage) not in (float, int) or not math.isfinite(read_voltage) or read_voltage <= 0:
        raise ValueError("Read voltage must be explicit and positive")
    resistance = converters.convert_adc_to_res(
        board["gain"], board["res_load"], read_voltage, board["adc_bit"],
        board["vol_ref_adc"], board["res_switches"], raw_adc)
    if not math.isfinite(resistance) or resistance <= 1e-8:
        raise ValueError("Native a2r returned its nonpositive-R replacement; no valid R")
    return {"resistance_ohm": resistance, "conductance_siemens": 1/resistance,
            "q": board["sum_gain"]/resistance, "read_voltage_parameter_v": read_voltage,
            "conversion": "MemriBoard convert_adc_to_res; native rounding retained"}


def mvm_output(raw_adc, board, converters):
    if type(raw_adc) is not int or not 0 <= raw_adc <= 2**board["adc_bit"]-2:
        raise ValueError("Invalid/saturated MVM ADC; zero is valid")
    # GUI math.py uses gain=1 here: no WR gain=11 and no additional sum_gain.
    return converters.convert_adc_to_volt(1, board["adc_bit"], board["vol_ref_adc"], raw_adc)
