"""Raw readback records and hashes for the root experiment.py; no entry point."""

from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import time

from .measurements import adc_to_resistance, resistance_to_conductance

def utc_now():
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_record(session, protocol, *, session_id, array_id, phase, task_id, wl, bl):
    start = time.monotonic()
    row = {
        "timestamp_utc": utc_now(), "monotonic_time_s": start,
        "session_id": session_id, "array_id": array_id, "source": "simulator",
        "wl": wl, "bl": bl, "task_id": task_id, "raw_adc": None,
        "status": "error", "latency_s": None, "phase": phase,
        "cycle": None, "target_conductance_siemens": None,
    }
    try:
        result = session.read_cell(wl, bl, task_id)
        row["native_result_repr"] = repr(result)
        if not isinstance(result, (tuple, list)) or len(result) != 3:
            raise ValueError("Expected native (resistance, request_id, raw_adc) result")
        _, returned_id, raw_adc = result
        # Preserve even invalid ADC values in valid JSON. Exact native repr remains
        # available for malformed types, NaN, and infinity without fabricating R.
        if raw_adc is None or type(raw_adc) in (str, bool, int):
            row["raw_adc"] = raw_adc
        elif type(raw_adc) is float and math.isfinite(raw_adc):
            row["raw_adc"] = raw_adc
        else:
            row["raw_adc"] = repr(raw_adc)
        if type(returned_id) is not int or returned_id != task_id:
            raise ValueError(f"Request ID mismatch: expected {task_id}, received {returned_id!r}")
        resistance = adc_to_resistance(raw_adc, protocol.board)
        row["resistance_ohm"] = resistance
        row["conductance_siemens"] = resistance_to_conductance(resistance)
        row["status"] = "ok"
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["latency_s"] = time.monotonic() - start
    return row



