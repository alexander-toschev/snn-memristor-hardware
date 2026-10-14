"""Analyze an existing JSONL file offline. Does not acquire any measurements.

Required row fields match protocol.json: timestamp_utc, monotonic_time_s,
session_id, array_id, source (hardware or simulator), wl, bl, task_id,
raw_adc, status (ok), latency_s and phase. For write/readback observations,
also supply cycle (integer >=0) and target_conductance_siemens (positive).
task_id identifies one request; session_id identifies the acquisition session.
Never reuse a cycle number for a new programming event within one phase.
An invalid row is retained in the JSON error report and blocks all statistics.

CLI (requires user approval to run):
  python analyze_readback.py existing.jsonl --protocol protocol.json --output report.json
"""

import argparse
from collections import defaultdict
from datetime import datetime
import json
import math
from pathlib import Path

from config import load_protocol, positive_number
from measurements import (
    adc_to_resistance,
    resistance_to_conductance,
    resistance_to_mvm_gain,
    sample_stats,
    write_error_stats,
)


def validate_row(row, protocol):
    if not isinstance(row, dict):
        raise ValueError("Each JSONL row must be an object")
    for field in ("session_id", "array_id", "phase"):
        if not isinstance(row[field], str) or not row[field].strip():
            raise ValueError(f"{field} must be a nonempty string")
    task_id = row["task_id"]
    if not ((type(task_id) is int and task_id >= 0)
            or (isinstance(task_id, str) and task_id.strip())):
        raise ValueError("task_id must be a nonempty string or nonnegative integer")
    if row["source"] not in ("hardware", "simulator"):
        raise ValueError("source must be hardware or simulator")
    if not isinstance(row["timestamp_utc"], str):
        raise ValueError("timestamp_utc must be an ISO8601 string")
    timestamp = datetime.fromisoformat(row["timestamp_utc"].replace("Z", "+00:00"))
    if timestamp.utcoffset() is None or timestamp.utcoffset().total_seconds() != 0:
        raise ValueError("timestamp_utc must include the UTC timezone")
    for field in ("monotonic_time_s", "latency_s"):
        value = row[field]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{field} must be finite and nonnegative")
    if row["status"] != "ok":
        raise ValueError(f"Measurement status is not ok: {row['status']!r}")
    protocol.check_cell(row["wl"], row["bl"])
    has_cycle = row.get("cycle") is not None
    if has_cycle != (row.get("target_conductance_siemens") is not None):
        raise ValueError("cycle and target_conductance_siemens must be supplied together")
    if has_cycle:
        if type(row["cycle"]) is not int or row["cycle"] < 0:
            raise ValueError("cycle must be an integer >=0")
        positive_number(row["target_conductance_siemens"], "target_conductance_siemens")


def analyze(input_path: Path, protocol) -> dict:
    groups = defaultdict(list)
    cycle_targets = {}
    seen_requests = set()
    errors = []
    count = 0
    with Path(input_path).open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                validate_row(row, protocol)
                request = (row["session_id"], row["source"], row["array_id"], row["task_id"])
                if request in seen_requests:
                    raise ValueError("Duplicate task_id within the session/source/array")
                seen_requests.add(request)
                resistance = adc_to_resistance(row["raw_adc"], protocol.board)
                conductance = resistance_to_conductance(resistance)
                mvm_gain = resistance_to_mvm_gain(resistance, protocol.board)
                identity = (row["session_id"], row["source"], row["array_id"],
                            row["wl"], row["bl"], row["phase"])
                target, cycle = row.get("target_conductance_siemens"), row.get("cycle")
                if cycle is not None:
                    cycle_key = identity + (cycle,)
                    if cycle_targets.setdefault(cycle_key, target) != target:
                        raise ValueError("A programming cycle cannot have multiple targets")
                groups[identity + (target, cycle)].append(
                    (row["raw_adc"], resistance, conductance, mvm_gain)
                )
                count += 1
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                errors.append({"line_number": line_number, "raw_line": line.rstrip("\r\n"),
                               "error": str(exc)})
    if errors:
        return {"valid": False, "valid_record_count": count, "invalid_record_count": len(errors),
                "errors": errors, "readback_groups": [], "write_readback_error_groups": [],
                "interpretation": "Invalid records retained; all statistics blocked for review."}
    if not count:
        raise ValueError("Input contains no measurements")

    read_groups = []
    write_groups = defaultdict(list)
    observed_cells = defaultdict(set)
    for (session, source, array, wl, bl, phase, target, cycle), readings in groups.items():
        observed_cells[(session, source, array, phase)].add((wl, bl))
        adc, resistance, conductance, mvm_gain = zip(*readings)
        conductance_stats = sample_stats(conductance)
        read_groups.append({
            "session_id": session, "source": source, "array_id": array,
            "wl": wl, "bl": bl, "phase": phase,
            "target_conductance_siemens": target, "cycle": cycle,
            "raw_adc": sample_stats(adc),
            "resistance_ohm": sample_stats(resistance),
            "conductance_s": conductance_stats,
            "mvm_effective_gain": sample_stats(mvm_gain),
        })
        if cycle is not None:
            write_groups[(session, source, array, wl, bl, phase, target)].append(conductance_stats["mean"])
    return {
        "valid": True,
        "measurement_count": count,
        "cell_coverage": [
            {"session_id": session, "source": source, "array_id": array, "phase": phase,
             "missing_initial_cells": [
                 {"wl": wl, "bl": bl} for wl, bl in protocol.initial_cells if (wl, bl) not in cells
             ]}
            for (session, source, array, phase), cells in observed_cells.items()
        ],
        "readback_groups": read_groups,
        "write_readback_error_groups": [
            {"session_id": session, "source": source, "array_id": array,
             "wl": wl, "bl": bl, "phase": phase,
             "target_conductance_siemens": target, **write_error_stats(values, target)}
            for (session, source, array, wl, bl, phase, target), values in write_groups.items()
        ],
        "interpretation": (
            "Read std uses ddof=1 within each cell/target/cycle. Write error uses "
            "one mean conductance per cycle and includes readout uncertainty. "
            "Repeated reads do not remove write noise. No hardware was accessed."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Existing raw ADC JSONL file")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Create a new JSON report; refuse overwrite")
    args = parser.parse_args()
    try:
        result = analyze(args.input, load_protocol(args.protocol))
        report = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(report)
        else:
            print(report, end="")
    except (OSError, KeyError, TypeError, ValueError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    if not result["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
