"""Simulator experiments through the native Connector, without a second driver.

settings is a small dict: seed (default 20261014), input_voltage (default 0.1),
filters (default 14 fixed filters), session_id (default a new UUID). Programming uses
the documented fixed 0.3..3.0 V / 0.01 V / 125 us simulator reference staircase.
The caller owns the Connector and closes it. Suite.close() closes only the log.
"""

import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
from statistics import fmean
import time
import uuid

from .measurements import adc_to_resistance, sample_stats, write_error_stats


FILTERS = (63, 41, 32, 30, 38, 43, 60, 47, 2, 9, 46, 12, 0, 1)
TARGETS_Q = (0.01, 0.02, 0.03, 0.04, 0.05)
TILES = ((0, 32), (32, 64), (64, 75))


def flatten(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (tuple, list)):
        return [number for item in value for number in flatten(item)]
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Weights must be finite")
    return [number]


def comparison(actual, expected):
    errors = [a - e for a, e in zip(actual, expected)]
    if len(actual) != len(expected) or not errors:
        raise ValueError("Aligned, nonempty vectors required")
    ma, me = fmean(actual), fmean(expected)
    da, de = [a - ma for a in actual], [e - me for e in expected]
    denominator = math.sqrt(sum(x*x for x in da) * sum(x*x for x in de))
    rmse = math.hypot(*errors) / math.sqrt(len(errors))
    expected_sd = math.sqrt(fmean(x*x for x in de))
    corr = sum(a*e for a, e in zip(da, de)) / denominator if denominator else None
    return {"count": len(errors), "bias": fmean(errors), "rmse": rmse,
            "expected_std_ddof0": expected_sd, "pearson": corr,
            "historical_gate_pass": bool(corr is not None and corr >= 0.9
                                          and expected_sd > 0 and rmse <= 0.6*expected_sd)}


class SimulatorSuite:
    def __init__(self, connectors, protocol, output_dir, settings=None):
        if set(connectors) != {"A", "B"} or any(c.cb_type != "simulator" for c in connectors.values()):
            raise ValueError("SimulatorSuite only accepts the native simulator backend")
        if connectors["A"] is connectors["B"]:
            raise ValueError("Two distinct native simulator connections are required")
        self.connectors, self.protocol = connectors, protocol
        self.active_array = "A"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.settings = dict(settings or {})
        self.session_id = self.settings.get("session_id", str(uuid.uuid4()))
        self.input_voltage = self.settings.get("input_voltage", 0.1)
        if not math.isfinite(self.input_voltage) or not 0 < self.input_voltage <= 0.1:
            raise ValueError("input_voltage must be positive and <=0.1 V")
        self.rng = random.Random(self.settings.get("seed", 20261014))
        self.next_id = 0
        self.log = (self.output_dir / "raw_tasks.jsonl").open("x", encoding="utf-8")
        self.summary = {"source": "simulator", "hardware_qualification": False,
                        "array_ids": ["A", "B"], "stages": {}, "raw_task_count": 0}

    def close(self):
        self.log.close()

    def save(self, name, value):
        with (self.output_dir / name).open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")

    def table(self, name, rows):
        if not rows:
            return
        keys = list(dict.fromkeys(key for row in rows for key in row))
        with (self.output_dir / name).open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                                 for key, value in row.items()})

    def check_cell(self, wl, bl):
        if type(wl) is not int or type(bl) is not int or not (0 <= wl < 8 and 0 <= bl < 32):
            raise ValueError("Invalid WL/BL")
        if wl in self.protocol.excluded_wl:
            raise ValueError(f"Excluded WL={wl}")

    def impact(self, task, phase, **metadata):
        self.check_cell(task["wl"], task.get("bl", 0))
        self.next_id += 1
        task = dict(task, id=self.next_id)
        # Native Connector mutates list elements during SI->DAC conversion.
        sent = dict(task, vol=list(task["vol"])) if isinstance(task["vol"], list) else task.copy()
        start = time.monotonic()
        row = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "monotonic_time_s": start,
               "session_id": self.session_id, "array_id": self.active_array, "source": "simulator",
               "phase": phase, "task_id": self.next_id, "wl": task["wl"], "bl": task.get("bl"),
               "task": task, "raw_adc": None, "status": "error", **metadata}
        error = None
        try:
            result = self.connectors[self.active_array].impact(sent)
            row["native_result_repr"] = repr(result)
            if not isinstance(result, (tuple, list)) or len(result) != 3:
                raise ValueError("Expected native (resistance, id, raw_adc)")
            _, echoed, adc = result
            row["raw_adc"] = adc if type(adc) is int else repr(adc)
            if type(echoed) is not int or echoed != self.next_id:
                raise ValueError("Returned task ID mismatch")
            if task["mode_flag"] == 10:
                if type(adc) is not int or not 0 <= adc <= 2**self.protocol.board.adc_bit - 2:
                    raise ValueError("MVM ADC out of range (zero is valid)")
                row["output_voltage"] = adc * self.protocol.board.vol_ref_adc / 2**self.protocol.board.adc_bit
            else:
                resistance = adc_to_resistance(adc, self.protocol.board)
                row.update(resistance_ohm=resistance, conductance_siemens=1/resistance,
                           q=self.protocol.board.sum_gain/resistance)
            row["status"] = "ok"
        except Exception as exc:
            error = exc
            row["error"] = f"{type(exc).__name__}: {exc}"
        row["latency_s"] = time.monotonic() - start
        self.log.write(json.dumps(row, allow_nan=False) + "\n")
        self.log.flush()
        self.summary["raw_task_count"] += 1
        if error is not None:
            raise RuntimeError(f"Task {self.next_id} failed; raw result saved") from error
        return row

    def read(self, wl, bl, phase, **metadata):
        return self.impact({"mode_flag": 7, "vol": 0.0, "pulse_width": 0.0,
                            "sign": 0, "wl": wl, "bl": bl}, phase, **metadata)

    def pulse(self, wl, bl, voltage, sign, phase, **metadata):
        return self.impact({"mode_flag": 7, "vol": voltage, "pulse_width": 125e-6,
                            "sign": sign, "wl": wl, "bl": bl}, phase, **metadata)

    def program(self, wl, bl, desired_q, phase, **metadata):
        """Bounded reset to q<=.005, then SET until first target crossing."""
        last = self.read(wl, bl, phase, operation="before_reset", **metadata)
        reset_pulses = set_pulses = 0
        for step in range(271):
            if last["q"] <= 0.005:
                break
            last = self.pulse(wl, bl, round(0.3 + step*0.01, 2), 0, phase,
                              operation="reset", **metadata)
            reset_pulses += 1
        reset_ok = last["q"] <= 0.005
        if reset_ok:
            for step in range(271):
                if last["q"] >= desired_q:
                    break
                last = self.pulse(wl, bl, round(0.3 + step*0.01, 2), 1, phase,
                                  operation="set", **metadata)
                set_pulses += 1
        return {"reset_ok": reset_ok, "target_reached": reset_ok and last["q"] >= desired_q,
                "reset_pulses": reset_pulses, "set_pulses": set_pulses, "last_q": last["q"]}

    def run_baseline(self):
        self.active_array = "A"
        rows = []
        for wl, bl in self.protocol.initial_cells:
            first = self.read(wl, bl, "first_check")
            repeated = [self.read(wl, bl, "repeatability") for _ in range(20)]
            rows.append({"wl": wl, "bl": bl, "first_q": first["q"],
                         **sample_stats([r["conductance_siemens"] for r in repeated])})
        self.table("baseline.csv", rows)
        result = {"cell_count": len(rows), "read_count": 21*len(rows), "cells": rows}
        self.save("baseline.json", result)
        self.summary["stages"]["baseline"] = result
        return result

    def run_characterization(self):
        self.active_array = "A"
        attempts, stats = [], []
        for wl, bl in self.protocol.initial_cells:
            for target_q in TARGETS_Q:
                target_g = target_q/self.protocol.board.sum_gain
                cycle_means = []
                successful = 0
                for cycle in range(5):
                    metadata = {"cycle": cycle, "target_q": target_q, "target_conductance_siemens": target_g}
                    program = self.program(wl, bl, target_q, "characterization", **metadata)
                    readings = [self.read(wl, bl, "characterization_readback", **metadata) for _ in range(5)]
                    values = [row["conductance_siemens"] for row in readings]
                    observed = sample_stats(values)
                    if program["reset_ok"]:
                        cycle_means.append(observed["mean"])
                    successful += int(program["target_reached"])
                    attempts.append({"wl": wl, "bl": bl, **metadata, **program,
                                     "mean_conductance_siemens": observed["mean"],
                                     "read_std_s_ddof1": observed["std_ddof1"]})
                record = {"wl": wl, "bl": bl, "target_q": target_q, "target_conductance_siemens": target_g,
                          "attempt_count": 5, "reset_success_count": len(cycle_means),
                          "target_reached_count": successful}
                if cycle_means:
                    record.update(write_error_stats(cycle_means, target_g))
                stats.append(record)
                print(f"Characterization WL={wl} BL={bl} q={target_q:.3f}: {successful}/5 reached", flush=True)
        self.table("characterization_attempts.csv", attempts)
        self.table("characterization_stats.csv", stats)
        result = {"attempt_count": len(attempts), "post_read_count": len(attempts)*5,
                  "target_reached_count": sum(int(a["target_reached"]) for a in attempts), "groups": stats,
                  "interpretation": "Cycle error includes readout noise. Failed reset cycles counted but excluded from independent-cycle SD."}
        self.save("characterization.json", result)
        self.summary["stages"]["characterization"] = result
        return result

    def decoded_voltage(self, voltage):
        board = self.protocol.board
        code = int(round(voltage*(2**board.dac_bit-1)/board.vol_ref_dac))
        return round(code*board.vol_ref_dac/(2**board.dac_bit-1), 3)

    def mvm(self, wl, voltages, **metadata):
        if len(voltages) != 32 or any(not math.isfinite(v) or not 0 <= v <= 0.1 for v in voltages):
            raise ValueError("MVM requires 32 finite voltages in [0,.1]")
        return self.impact({"mode_flag": 10, "vol": list(voltages), "pulse_width": 0.0,
                            "sign": 0, "wl": wl, "bl": 0}, "mvm", **metadata)["output_voltage"]

    def run_mapping(self, weights, filters=None, label="phase2", repetitions=1):
        if not label or not label.replace("_", "").isalnum():
            raise ValueError("label must contain only letters, digits or underscores")
        if type(repetitions) is not int or not 1 <= repetitions <= 3:
            raise ValueError("repetitions must be an integer in [1,3]")
        filters = list(self.settings.get("filters", FILTERS) if filters is None else filters)
        if not filters or len(set(filters)) != len(filters):
            raise ValueError("Distinct, nonempty filters required")
        for fid in filters:
            if type(fid) is not int or not 0 <= fid < 64:
                raise ValueError("Filter IDs must be integers in [0,63]")
        inputs = [[0.0]*75]
        for index in range(75):
            inputs.append([float(i == index) for i in range(75)])
        for _ in range(16):
            active = set(self.rng.sample(range(75), 8))
            inputs.append([float(i in active) for i in range(75)])
        self.save(f"{label}_mvm_inputs.json", {"kind": "zero_75_basis_16_sparse", "inputs": inputs,
                                      "not_dataset_accuracy": True})
        summaries, programming, outputs, reconstructed, aggregate = [], [], [], {}, []
        voltage = self.decoded_voltage(self.input_voltage)
        if voltage <= 0:
            raise ValueError("input_voltage quantizes to zero")
        for fid in filters:
            w = flatten(weights[fid])
            if len(w) != 75 or not (scale := max(map(abs, w))):
                raise ValueError(f"Filter {fid} must contain 75 nonzero-scale taps")
            ideal = [sum(x*y for x, y in zip(vector, w)) for vector in inputs]
            weight_replicates, output_replicates = [], []
            for repetition in range(repetitions):
                actual = [0.0]*len(inputs)
                programmed = [0.0]*75
                reached = reset_success = 0
                for tile, (lo, hi) in enumerate(TILES):
                    self.active_array = "B" if tile == 1 else "A"
                    for bl, tap in enumerate(range(lo, hi)):
                        pair = []
                        for wl, sign in ((0, 1), (2, -1)):
                            target_q = 0.03 + sign*0.02*w[tap]/scale
                            metadata = {"stage": label, "filter_id": fid, "repetition": repetition,
                                        "tile": tile, "tap": tap, "target_q": target_q}
                            record = self.program(wl, bl, target_q, "mapping", **metadata)
                            readings = [self.read(wl, bl, "mapping_readback", **metadata) for _ in range(3)]
                            q = fmean(row["q"] for row in readings)
                            pair.append(q)
                            reached += int(record["target_reached"])
                            reset_success += int(record["reset_ok"])
                            programming.append({**metadata, "array_id": self.active_array,
                                                "wl": wl, "bl": bl, **record, "observed_q": q})
                        programmed[tap] = (pair[0]-pair[1])*scale/0.04
                    for input_id, vector in enumerate(inputs):
                        volts = [self.input_voltage*x for x in vector[lo:hi]] + [0.0]*(32-(hi-lo))
                        metadata = {"stage": label, "filter_id": fid, "repetition": repetition,
                                    "tile": tile, "input_id": input_id}
                        plus = self.mvm(0, volts, **metadata)
                        minus = self.mvm(2, volts, **metadata)
                        actual[input_id] += (plus-minus)*scale/(0.04*voltage)
                prediction = [sum(x*y for x, y in zip(vector, programmed)) for vector in inputs]
                result = {"filter_id": fid, "repetition": repetition, "programmed_cell_count": 150,
                          "target_reached_count": reached, "reset_success_count": reset_success,
                          "input_count": len(inputs), "weights": comparison(programmed, w),
                          "mvm_vs_ideal": comparison(actual, ideal),
                          "mvm_vs_programmed": comparison(actual, prediction)}
                summaries.append(result)
                weight_replicates.append(programmed)
                output_replicates.append(actual)
                for input_id, (observed, expected, predicted) in enumerate(zip(actual, ideal, prediction)):
                    outputs.append({"filter_id": fid, "repetition": repetition, "input_id": input_id,
                                    "measured": observed, "ideal": expected, "programmed_prediction": predicted})
                self.save(f"{label}_filter_{fid:02d}_rep_{repetition}.json",
                          {**result, "ideal_weights": w, "readback_weights": programmed})
                print(f"{label} filter {fid} rep {repetition}: {reached}/150 targets; "
                      f"MVM RMSE {result['mvm_vs_ideal']['rmse']:.6g}", flush=True)
            reconstructed[str(fid)] = weight_replicates
            for k in range(1, repetitions+1):
                mean_w = [fmean(values) for values in zip(*weight_replicates[:k])]
                mean_y = [fmean(values) for values in zip(*output_replicates[:k])]
                aggregate.append({"filter_id": fid, "K": k, "readback_weights": mean_w,
                                  "weights": comparison(mean_w, w), "mvm_vs_ideal": comparison(mean_y, ideal),
                                  "all_resets_succeeded": all(r["reset_success_count"] == 150
                                      for r in summaries if r["filter_id"] == fid and r["repetition"] < k)})
        self.table(f"{label}_programming.csv", programming)
        self.table(f"{label}_mvm_outputs.csv", outputs)
        result = {"array_count": 2, "arrays_reused": True, "tile_arrays": ["A", "B", "A"],
                  "wl_pair": [0, 2], "excluded_wl": [1], "filter_count": len(filters),
                  "repetitions": repetitions, "programmed_cell_count": 150*len(summaries),
                  "inputs_per_filter": len(inputs), "results": summaries,
                  "reconstructed_weights": reconstructed, "prefix_means": aggregate,
                  "interpretation": "Deterministic simulator basis/sparse verification; not CIFAR accuracy or hardware qualification. K means independent reprogramming attempts, not repeated reads; failed resets are flagged. No guaranteed improvement."}
        self.save(f"{label}.json", result)
        self.summary["stages"][label] = result
        return result
