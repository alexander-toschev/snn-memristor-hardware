"""Pure readback conversions and statistics; no driver or simulator imports."""

import math
from statistics import fmean, stdev

from config import Board, positive_number


def adc_to_resistance(raw_adc: int, board: Board) -> float:
    """Return ohms. Rail codes are invalid; never clamp or silently repair ADC."""
    if type(raw_adc) is not int or not 1 <= raw_adc <= (2**board.adc_bit - 2):
        raise ValueError("raw_adc must be an integer in [1, 16382]")
    resistance = (
        board.gain * board.res_load * board.vol_read * 2**board.adc_bit
        / (raw_adc * board.vol_ref_adc)
        - board.res_switches - board.res_load
    )
    return positive_number(resistance, "derived resistance_ohm")


def resistance_to_conductance(resistance_ohm: float) -> float:
    """Physical conductance G=1/R, in siemens."""
    return 1.0 / positive_number(resistance_ohm, "resistance_ohm")


def resistance_to_mvm_gain(resistance_ohm: float, board: Board) -> float:
    """MVM effective gain q=sum_gain/R; NOT the WR-mode divider weight."""
    return board.sum_gain / positive_number(resistance_ohm, "resistance_ohm")


def sample_stats(values) -> dict:
    """Signed samples, ddof=1. One sample has undefined std (JSON null)."""
    samples = list(values)
    if not samples or any(type(x) not in (int, float) or not math.isfinite(x) for x in samples):
        raise ValueError("Expected at least one finite numeric sample")
    return {
        "count": len(samples),
        "mean": fmean(samples),
        "std_ddof1": stdev(samples) if len(samples) > 1 else None,
        "min": min(samples),
        "max": max(samples),
    }


def write_error_stats(cycle_conductances_s, target_conductance_s: float) -> dict:
    """One readback estimate per programming cycle, same cell and target.

    This measures observed write/readback error, not isolated write noise.
    Averaging reads does not remove programming error or prove independence.
    """
    target = positive_number(target_conductance_s, "target_conductance_s")
    errors = [positive_number(value, "cycle conductance_s") - target for value in cycle_conductances_s]
    stats = sample_stats(errors)
    return {
        "cycle_count": stats["count"],
        "bias_s": stats["mean"],
        "error_std_s_ddof1": stats["std_ddof1"],
        "rmse_s": math.hypot(*errors) / math.sqrt(len(errors)),
    }
