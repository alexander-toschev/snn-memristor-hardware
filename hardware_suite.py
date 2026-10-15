"""Four physical Conv1 phases on ONE sequentially reused array.

Only public Connector.impact is used here. The entry point owns open/close.
Programming uses native mode-7 timing, whose physical pulse width is unknown.
"""

import json
import math
from pathlib import Path
import random
from statistics import fmean, stdev
import time

from read_all import now, json_value, save_json
import native_conversions as conversion

FILTERS = (63, 41, 32, 30, 38, 43, 60, 47, 2, 9, 46, 12, 0, 1)
TARGETS = (.01, .02, .03, .04, .05)
TILES = ((0, 32), (32, 64), (64, 75))


class ScientificGateError(RuntimeError):
    pass


def stats(values):
    return {"count": len(values), "mean": fmean(values),
            "std_ddof1": stdev(values) if len(values) > 1 else None,
            "min": min(values), "max": max(values)}


def compare(observed, expected):
    if len(observed) != len(expected) or not observed:
        raise ValueError("Aligned nonempty vectors required")
    errors = [a-b for a, b in zip(observed, expected)]
    aa = [a-fmean(observed) for a in observed]
    bb = [b-fmean(expected) for b in expected]
    denom = math.sqrt(sum(a*a for a in aa)*sum(b*b for b in bb))
    corr = sum(a*b for a, b in zip(aa, bb))/denom if denom else None
    rmse = math.sqrt(fmean(e*e for e in errors))
    sd = math.sqrt(fmean(b*b for b in bb))
    return {"count": len(errors), "bias": fmean(errors), "rmse": rmse,
            "pearson": corr, "expected_std_ddof0": sd,
            "historical_gate_pass": corr is not None and corr >= .9 and sd > 0 and rmse <= .6*sd}


class HardwareSuite:
    def __init__(self, connector, board, output_dir, session_id, converters, *, max_pulses=6625950):
        if connector.cb_type != "real":
            raise ValueError("HardwareSuite requires the native real Connector")
        if type(max_pulses) is not int or not 1 <= max_pulses <= 6625950:
            raise ValueError("Pulse budget must be in 1..6625950")
        self.connector, self.board, self.converters = connector, board, converters
        self.output = Path(output_dir)
        self.session_id, self.max_pulses = session_id, max_pulses
        self.next_id = self.pulses = self.invalid_readings = 0
        self.log = (self.output / "raw.jsonl").open("x", encoding="utf-8", newline="\n")
        self.results = {"source": "hardware", "physical_array_id": "single_board",
                        "pulse_timing": "native_unknown", "phases": {}}

    def close(self):
        self.log.close()

    def append(self, filename, row):
        with (self.output / filename).open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
            stream.flush()

    def emit(self, row):
        self.log.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
        self.log.flush()

    def impact(self, mode, wl, bl, voltage, sign, phase, **metadata):
        if type(wl) is not int or wl not in (0, 2) or type(bl) is not int or not 0 <= bl < 32:
            raise ValueError("Programming/read plan permits WL0/WL2 and BL0..31")
        if mode == 10:
            if not isinstance(voltage, list) or len(voltage) != 32:
                raise ValueError("MVM requires 32 inputs")
            if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= .1 for v in voltage):
                raise ValueError("MVM input outside 0..0.1 V")
            if sum(v != 0 for v in voltage) > 8:
                raise ValueError("MVM permits at most eight active BL")
            codes = [conversion.voltage(v, self.board, self.converters) for v in voltage]
        elif mode == 7:
            if type(voltage) not in (int, float) or not math.isfinite(voltage) or not 0 <= voltage <= 3:
                raise ValueError("Mode7 amplitude outside 0..3 V")
            if sign not in (0, 1):
                raise ValueError("Invalid polarity")
            codes = conversion.voltage(voltage, self.board, self.converters)
            if voltage:
                if self.pulses >= self.max_pulses:
                    raise ScientificGateError("Whole-session pulse budget exhausted")
                self.pulses += 1
        else:
            raise ValueError("Only native modes 7 and 10 belong to this plan")
        self.next_id += 1
        task = {"mode_flag": mode, "vol": voltage.copy() if isinstance(voltage, list) else voltage,
                "pulse_width": 0.0, "sign": sign, "id": self.next_id, "wl": wl, "bl": bl}
        row = {"session_id": self.session_id, "source": "hardware", "physical_array_id": "single_board",
               "task_id": self.next_id, "timestamp_utc": now(), "monotonic_time_s": time.monotonic(),
               "phase": phase, "wl": wl, "bl": bl, "command": task, "native_voltage_conversion": codes,
               "native_adc_average_count": 10 if mode == 7 else 1,
               "pulse_timing": "native_unknown" if mode == 7 and voltage else None,
               "historical_reference_pulse_width_s": .000125 if mode == 7 and voltage else None,
               "physical_pulse_width_s": None, "raw_adc": None, "status": "pending", **metadata}
        self.emit({**row, "event": "request"})
        start = time.monotonic()
        try:
            # Connector's SI conversion mutates nested lists: never share log input.
            sent = {**task, "vol": task["vol"].copy()} if isinstance(task["vol"], list) else task.copy()
            response = self.connector.impact(sent)
            row["native_result"] = json_value(response)
            if not isinstance(response, (tuple, list)) or len(response) != 3:
                raise ValueError("Expected native (R, id, ADC)")
            native_r, echoed, adc = response
            row.update(raw_adc=json_value(adc), returned_id=json_value(echoed), native_resistance=json_value(native_r))
            if type(echoed) is not int or echoed != self.next_id:
                raise ValueError("Native response ID mismatch")
            if mode == 10:
                row["output_voltage"] = conversion.mvm_output(adc, self.board, self.converters)
                row["native_resistance_interpretation"] = "not an MVM resistance"
            else:
                row.update(conversion.readback(adc, self.board, self.converters,
                                               read_voltage=self.board["vol_read"]))
            row["status"] = "ok"
        except (Exception, KeyboardInterrupt) as exc:
            self.invalid_readings += 1
            row.update(status="error", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            row["latency_s"] = time.monotonic()-start
            self.emit({**row, "event": "result"})
        return row

    def read(self, wl, bl, phase, **metadata):
        return self.impact(7, wl, bl, 0.0, 0, phase, **metadata)

    def program(self, wl, bl, target_q, phase, **metadata):
        attempt = {"wl": wl, "bl": bl, "target_q": target_q, "phase": phase, **metadata,
                   "reset_pulses": 0, "set_pulses": 0, "reset_ok": False, "target_reached": False,
                   "pulse_count_semantics": "attempted nonzero mode7; physical delivery on exception is unknown"}
        def pulse(voltage, sign, operation):
            before = self.pulses
            try:
                return self.impact(7, wl, bl, voltage, sign, phase,
                                   operation=operation, target_q=target_q, **metadata)
            finally:
                attempt[operation+"_pulses"] += self.pulses-before
                attempt["last_command_id"] = self.next_id
        try:
            last = self.read(wl, bl, phase, operation="before_reset", **metadata)
            for step in range(271):
                if last["q"] <= .005:
                    break
                last = pulse(round(.3+.01*step, 2), 0, "reset")
            attempt.update(reset_ok=last["q"] <= .005, reset_last_q=last["q"])
            if not attempt["reset_ok"]:
                raise ScientificGateError("RESET endpoint not reached within 271 native pulses")
            for step in range(271):
                if last["q"] >= target_q:
                    break
                last = pulse(round(.3+.01*step, 2), 1, "set")
            attempt.update(target_reached=last["q"] >= target_q, last_q=last["q"],
                           signed_overshoot_q=last["q"]-target_q)
            if not attempt["target_reached"]:
                raise ScientificGateError("SET target not reached within 271 native pulses")
        except (Exception, KeyboardInterrupt) as exc:
            attempt["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self.append("programming_attempts.jsonl", attempt)
        return attempt

    def phase1(self):
        result = {"completion_conditions_pass": False, "precision_qualified": False,
                  "baseline": [], "attempts": [], "groups": []}
        self.results["phases"]["phase1"] = result
        cells = ((0, 2), (2, 6), (2, 4))
        for wl, bl in cells:
            first = self.read(wl, bl, "baseline", read_index=0)
            repeated = [self.read(wl, bl, "baseline", read_index=i+1) for i in range(20)]
            row = {"wl": wl, "bl": bl, "first": first["conductance_siemens"],
                   **stats([r["conductance_siemens"] for r in repeated]), "native_samples_per_impact": 10}
            result["baseline"].append(row)
            self.append("baseline.jsonl", row)
        for wl, bl in cells:
            for target in TARGETS:
                cycle_means = []
                for cycle in range(5):
                    programmed = self.program(wl, bl, target, "phase1", cycle=cycle)
                    reads = [self.read(wl, bl, "phase1_readback", target_q=target, cycle=cycle,
                                       read_index=i) for i in range(5)]
                    observed = stats([r["conductance_siemens"] for r in reads])
                    row = {**programmed, **observed, "mean_q": observed["mean"]*self.board["sum_gain"],
                           "signed_error_siemens": observed["mean"]-target/self.board["sum_gain"]}
                    row["post_read_target_reached"] = row["mean_q"] >= target
                    result["attempts"].append(row)
                    self.append("phase1_attempts.jsonl", row)
                    cycle_means.append(observed["mean"])
                errors = [mean-target/self.board["sum_gain"] for mean in cycle_means]
                result["groups"].append({"wl": wl, "bl": bl, "target_q": target,
                                         "signed_error": stats(errors), "rmse_siemens": math.sqrt(fmean(e*e for e in errors))})
                print(f"phase1 WL{wl}/BL{bl} q={target}: 5/5 attempts passed", flush=True)
        result["completion_conditions_pass"] = len(result["attempts"]) == 75 and self.invalid_readings == 0
        result["gate_scope"] = "RESET/SET reachability on three cells; not full-array calibration or independence proof"
        save_json(self.output / "phase1.json", result)
        return result

    def mvm(self, wl, volts, conductances, phase, **metadata):
        if len(conductances) != 32 or any(not math.isfinite(g) or g < 0 for g in conductances):
            raise ValueError("Expected 32 finite nonnegative readback conductances")
        current = sum(v*g for v, g in zip(volts, conductances))
        vout = current*self.board["sum_gain"]
        prediction = {"predicted_current_a": current, "predicted_output_v": vout,
                      "current_limit_a": .005, "adc_headroom_fraction": .8,
                      "prediction_is_uncalibrated": True}
        if current > .005 or vout > .8*self.board["vol_ref_adc"]:
            self.append("rejected_commands.jsonl", {"phase": phase, "wl": wl, "vol": volts,
                                                    **metadata, **prediction, "reason": "provisional MVM current/headroom limit"})
            raise ScientificGateError("MVM current/headroom estimate exceeded; amplitude remains unchanged")
        observed = self.impact(10, wl, 0, volts.copy(), 0, phase, **prediction, **metadata)
        if observed["output_voltage"] > .8*self.board["vol_ref_adc"]:
            raise ScientificGateError("Observed MVM output exceeded the 80% ADC headroom bound")
        return observed

    def mapping(self, weights, filters, phase, repetitions=1):
        rng = random.Random(20261014)
        inputs = [[0.0]*75] + [[float(i == j) for i in range(75)] for j in range(75)]
        for _ in range(16):
            active = set(rng.sample(range(75), 8))
            inputs.append([float(i in active) for i in range(75)])
        save_json(self.output / f"{phase}_inputs.json", {"inputs": inputs, "seed": 20261014,
                                                       "seed_scope": "reset_each_mapping_phase"})
        vin = conversion.voltage(.1, self.board, self.converters)["native_decoded_v"]
        result = {"filter_runs": [], "prefix_means": [], "programming_gate_pass": False,
                  "zero_offset_correction": False, "scale_fitted_to_ideal": False}
        self.results["phases"][phase] = result
        for fid in filters:
            values = weights[fid].reshape(-1).tolist()
            if len(values) != 75 or not all(math.isfinite(w) for w in values) or not (scale := max(map(abs, values))):
                raise ValueError("Expected 75 finite nonzero-scale Conv1 weights")
            ideal = [sum(x*w for x, w in zip(vector, values)) for vector in inputs]
            replicas_w, replicas_y = [], []
            for repetition in range(repetitions):
                reconstructed, outputs = [0.0]*75, [0.0]*len(inputs)
                for tile, (lo, hi) in enumerate(TILES):
                    conductances = {0: [0.0]*32, 2: [0.0]*32}
                    # ONE physical array: measure every input for this tile before overwriting it.
                    for bl, tap in enumerate(range(lo, hi)):
                        qs = []
                        for wl, sign in ((0, 1), (2, -1)):
                            target = .03 + .02*sign*values[tap]/scale
                            metadata = {"filter_id": fid, "repetition": repetition, "tile": tile, "tap": tap}
                            self.program(wl, bl, target, phase, **metadata)
                            reads = [self.read(wl, bl, phase+"_readback", read_index=i, **metadata) for i in range(3)]
                            mean_q = fmean(row["q"] for row in reads)
                            self.append("mapping_cells.jsonl", {"phase": phase, **metadata, "wl": wl, "bl": bl,
                                                                "target_q": target, "mean_q": mean_q,
                                                                "signed_error_q": mean_q-target,
                                                                "post_read_target_reached": mean_q >= target})
                            qs.append(mean_q)
                    # Re-read the completed tile: later programming can disturb earlier cells.
                    for bl, tap in enumerate(range(lo, hi)):
                        final_q = []
                        for wl in (0, 2):
                            metadata = {"filter_id": fid, "repetition": repetition, "tile": tile, "tap": tap}
                            reads = [self.read(wl, bl, phase+"_pre_mvm_tile", read_index=i, **metadata) for i in range(3)]
                            conductances[wl][bl] = fmean(row["conductance_siemens"] for row in reads)
                            final_q.append(conductances[wl][bl]*self.board["sum_gain"])
                            self.append("pre_mvm_tile_readback.jsonl", {"phase": phase, **metadata, "wl": wl, "bl": bl,
                                                                       "mean_conductance_siemens": conductances[wl][bl]})
                        reconstructed[tap] = (final_q[0]-final_q[1])*scale/.04
                    for input_id, vector in enumerate(inputs):
                        volts = [vin*x for x in vector[lo:hi]] + [0.0]*(32-(hi-lo))
                        metadata = {"filter_id": fid, "repetition": repetition, "tile": tile, "input_id": input_id}
                        plus = self.mvm(0, volts.copy(), conductances[0], phase+"_mvm", zero_input=input_id == 0, **metadata)
                        minus = self.mvm(2, volts.copy(), conductances[2], phase+"_mvm", zero_input=input_id == 0, **metadata)
                        contribution = (plus["output_voltage"]-minus["output_voltage"])*scale/(.04*vin)
                        outputs[input_id] += contribution
                        self.append("tile_outputs.jsonl", {"phase": phase, **metadata, "contribution": contribution,
                                                           "plus_adc": plus["raw_adc"], "minus_adc": minus["raw_adc"]})
                predicted = [sum(x*w for x, w in zip(vector, reconstructed)) for vector in inputs]
                row = {"filter_id": fid, "repetition": repetition, "weights": compare(reconstructed, values),
                       "mvm_vs_ideal": compare(outputs, ideal), "mvm_vs_programmed": compare(outputs, predicted),
                       "reconstructed_weights": reconstructed, "outputs": outputs, "ideal_outputs": ideal,
                       "programmed_prediction": predicted}
                result["filter_runs"].append(row)
                self.append(f"{phase}_filters.jsonl", row)
                replicas_w.append(reconstructed)
                replicas_y.append(outputs)
                print(f"{phase} filter={fid} repetition={repetition}: completed", flush=True)
            for k in range(1, repetitions+1):
                mean_w = [fmean(items) for items in zip(*replicas_w[:k])]
                mean_y = [fmean(items) for items in zip(*replicas_y[:k])]
                aggregate = {"filter_id": fid, "K": k, "weights": compare(mean_w, values),
                             "mvm_vs_ideal": compare(mean_y, ideal),
                             "independence_established": False}
                result["prefix_means"].append(aggregate)
                self.append(f"{phase}_prefix_means.jsonl", aggregate)
        result["programming_gate_pass"] = self.invalid_readings == 0
        save_json(self.output / f"{phase}.json", result)
        return result
