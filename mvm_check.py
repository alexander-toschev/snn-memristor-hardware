"""Small existing-conductance diagnostic: only native Connector modes 9 and 10."""
import argparse
import json
import statistics
import time
import uuid

from read_all import ROOT, INI, EXPECTED_BOARD, local_path, preflight, native_connector, now, sha256, save_json, json_value
from native_conversions import voltage, readback, mvm_output
from experiment_hardware import launch_provenance, READ_ALL_SHA256

ROWS = (0, 2)
COLS = (2, 3)
VECTORS = {"zero": (), "A": (2,), "B": (3,), "AB": (2, 3)}
LOCK = "results/read_all/.hardware.lock"


def stats(values):
    return {"count": len(values), "mean": statistics.mean(values),
            "sample_std": statistics.stdev(values) if len(values) > 1 else None}


def collect(connector, converters, output, session_id, *, gui_ordered=False):
    """No driver/interface access. Preserve partial records even after an exception."""
    board = EXPECTED_BOARD
    rows = tuple(range(8)) if gui_ordered else ROWS
    cols = tuple(range(32)) if gui_ordered else COLS
    vectors = {**VECTORS, "ALL32": tuple(range(32))} if gui_ordered else VECTORS
    summary = {"status": "failed", "impact_attempts": 0, "valid_responses": 0,
               "readbacks": [], "mvm": [], "row_statistics": [], "conductance_comparison": [],
               "calibration_verified": False, "acceptance_threshold": None,
               "zero_offset_correction_applied": False, "gui_ordered": gui_ordered}
    decoded = voltage(.1, board, converters)
    try:
        with (output / "raw.jsonl").open("x", encoding="utf-8") as stream:
            def emit(row):
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()

            def impact(mode, wl, bl, value, **metadata):
                task_id = summary["impact_attempts"] + 1
                task = {"mode_flag": mode, "vol": value, "pulse_width": 0.0,
                        "sign": 0, "id": task_id, "wl": wl, "bl": bl}
                if gui_ordered and mode == 10:
                    # Exact public task fields used by GUI MatMul; log task_id stays unique.
                    task = {"mode_flag": 10, "vol": value, "id": 0, "wl": wl}
                dac = ([voltage(v, board, converters)["dac_code"] for v in value]
                       if mode == 10 else decoded["dac_code"])
                row = {"session_id": session_id, "source": "hardware", "task_id": task_id,
                       "timestamp_utc": now(), "monotonic_time_s": time.monotonic(),
                       "command_si": task, "vol_dac": dac, **metadata}
                emit({**row, "event": "request"})
                summary["impact_attempts"] += 1
                start = time.monotonic()
                try:
                    submitted = {**task, "vol": list(value) if mode == 10 else value}
                    result = connector.impact(submitted)
                    row["native_result"] = json_value(result)
                    if not isinstance(result, (list, tuple)) or len(result) != 3:
                        raise ValueError("Expected native (R, id, ADC)")
                    native_r, returned_id, adc = result
                    row.update(native_r=json_value(native_r), returned_id=json_value(returned_id), raw_adc=json_value(adc))
                    expected_id = 0 if mode == 9 or gui_ordered else task_id
                    if type(returned_id) is not int or returned_id != expected_id:
                        raise ValueError(f"Unexpected native ID; expected {expected_id}")
                    if mode == 9:
                        row.update(readback(adc, board, converters, read_voltage=.1))
                        row["native_r_interpretation"] = "INI vol_read=.3; use separate native a2r with read_voltage=.1"
                    else:
                        row["output_voltage_v"] = mvm_output(adc, board, converters)
                        row["native_r_interpretation"] = "not interpreted for MVM"
                        if row["output_voltage_v"] > .8 * board["vol_ref_adc"]:
                            raise ValueError("Observed MVM output exceeds 80% ADC reference")
                    row["status"] = "ok"
                    summary["valid_responses"] += 1
                    return row
                except (Exception, KeyboardInterrupt) as exc:
                    row.update(status="error", error=f"{type(exc).__name__}: {exc}")
                    raise
                finally:
                    emit({**row, "event": "result", "latency_s": time.monotonic()-start})

            if not connector.open_port():
                raise ConnectionError("Connector.open_port returned false")
            for stage in ("before", "after"):
                for wl in rows:
                    for bl in cols:
                        for repeat in range(3):
                            summary["readbacks"].append(impact(9, wl, bl, .1, stage=stage, repeat=repeat))
                if stage == "after":
                    break
                means = {(wl, bl): statistics.mean(r["conductance_siemens"] for r in summary["readbacks"]
                         if r["command_si"]["wl"] == wl and r["command_si"]["bl"] == bl)
                         for wl in rows for bl in cols}
                predictions = {}
                for wl in rows:
                    for label, active in vectors.items():
                        current = sum(decoded["native_decoded_v"] * means[wl, bl] for bl in active)
                        predicted_output = board["sum_gain"] * current
                        gui_prediction = sum(.1 * round(board["sum_gain"] * means[wl, bl], 4) for bl in active)
                        predictions[wl, label] = (current, predicted_output, gui_prediction)
                if gui_ordered:
                    summary["pre_mvm_guard_predictions"] = [
                        {"wl": wl, "vector": label, "current_a": values[0], "output_v": values[1],
                         "gui_prediction_v": values[2]} for (wl, label), values in predictions.items()]
                    if any(i > .005 or v > .8*board["vol_ref_adc"] or gui >= 3.3
                           for i, v, gui in predictions.values()):
                        raise ValueError("Pre-read model exceeds 5 mA / 80% ADC or GUI >=3.3 V skip condition; no MVM sent")
                order = ((repeat, label, active, wl) for repeat in range(5)
                         for label, active in vectors.items() for wl in rows) if gui_ordered else (
                         (repeat, label, active, wl) for wl in rows for repeat in range(5)
                         for label, active in vectors.items())
                for repeat, label, active, wl in order:
                    vector = [.1 if bl in active else 0.0 for bl in range(32)]
                    current, predicted_output, gui_prediction = predictions[wl, label]
                    if current > .005 or predicted_output > .8 * board["vol_ref_adc"]:
                        raise ValueError("Pre-read model exceeds 5 mA / 80% ADC reference bound")
                    summary["mvm"].append(impact(10, wl, 0, vector, stage="mvm", vector=label,
                        repeat=repeat, predicted_current_a=current, predicted_output_v=predicted_output,
                        gui_prediction_from_preread_v=gui_prediction,
                        gui_skip_eligibility_from_preread=gui_prediction >= 3.3))
            for wl in rows:
                groups = {label: [r for r in summary["mvm"] if r["command_si"]["wl"] == wl and r["vector"] == label]
                          for label in vectors}
                values = {label: {"adc": stats([r["raw_adc"] for r in rows]),
                                 "output_v": stats([r["output_voltage_v"] for r in rows])}
                          for label, rows in groups.items()}
                epsilon = {unit: values["AB"][unit]["mean"] - values["A"][unit]["mean"]
                           - values["B"][unit]["mean"] + values["zero"][unit]["mean"] for unit in ("adc", "output_v")}
                summary["row_statistics"].append({"wl": wl, "vectors": values,
                    "offset_adjusted_superposition_epsilon": epsilon,
                    "epsilon_definition": "mean(AB)-mean(A)-mean(B)+mean(zero); descriptive, no pass threshold"})
                for bl in cols:
                    stages = {stage: stats([r["conductance_siemens"] for r in summary["readbacks"]
                              if r["stage"] == stage and r["command_si"]["wl"] == wl and r["command_si"]["bl"] == bl])
                              for stage in ("before", "after")}
                    delta = stages["after"]["mean"] - stages["before"]["mean"]
                    summary["conductance_comparison"].append({"wl": wl, "bl": bl, **stages,
                        "mean_delta_siemens": delta, "fractional_delta": delta / stages["before"]["mean"],
                        "observed_mean_difference_nonzero": delta != 0,
                        "interpretation": "descriptive readback difference; not evidence of changed weights or calibrated drift"})
            summary["status"] = "completed"
    except (Exception, KeyboardInterrupt) as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            if not connector.close_port():
                raise ConnectionError("Connector.close_port returned false")
        except Exception as exc:
            summary.update(status="failed", close_error=f"{type(exc).__name__}: {exc}")
        summary["finished_utc"] = now()
        save_json(output / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run-hardware", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    parser.add_argument("--gui-ordered", action="store_true", help="GUI MatMul order: vectors outer, WL0..7 inner; native ID0")
    args = parser.parse_args(argv)
    if not (args.run_hardware or args.dry_run):
        parser.print_help()
        return 0
    if sha256(local_path("read_all.py")) != READ_ALL_SHA256:
        raise ValueError("Unreviewed read_all source")
    config, snapshot, hashes = preflight(INI)
    for filename in ("experiment.py", "mvm_check.py", "native_conversions.py", "experiment_hardware.py"):
        hashes[filename] = sha256(local_path(filename))
    rows = tuple(range(8)) if args.gui_ordered else ROWS
    cols = tuple(range(32)) if args.gui_ordered else COLS
    read_count = len(rows)*len(cols)*6
    mvm_count = len(rows)*5*(5 if args.gui_ordered else 4)
    description = {"modes": [9, 10], "wl": list(rows), "bl": list(cols), "nominal_v": .1,
                   "mode9_count": read_count, "mode10_count": mvm_count, "total_impacts": read_count+mvm_count,
                   "gui_ordered": args.gui_ordered,
                   "weights_programmed": False, "source_sha256": hashes}
    if args.gui_ordered:
        gui_source = "vendor/MemriBoard/gui/windows/math.py"
        hashes[gui_source] = sha256(local_path(gui_source))
        description["gui_comparison"] = {
            "source": gui_source, "native_mode10_id": 0, "mask": "all_ones_32x8",
            "order": "repeat -> vector(zero,A,B,AB,ALL32) -> WL0..7",
            "preprocessing": "none", "postprocessing": "none", "extra_quantization": False,
            "gui_input_ceiling_v": .3, "diagnostic_max_input_v": .1,
            "gui_skip_predicted_v": 3.3,
            "gui_prediction_source": "GUI caches round(sum_gain/all_resistances,4); no fresh read in MatMul",
            "diagnostic_prediction_source": "fresh mode9 pre-read; GUI cache is not available",
            "guard_difference": "Before any MVM, stop if any fresh model exceeds 5mA/80% ADC or GUI prediction >=3.3V; observed 80% also stops. GUI uses cached prediction and substitutes ADC0 for skipped calls; no response is fabricated here."}
    if args.dry_run:
        print(json.dumps({"hardware_access": False, **description}))
        return 0
    provenance = launch_provenance(argv)
    lock = local_path(LOCK)
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(f"MVM diagnostic {now()}\n")
    try:
        session_id = str(uuid.uuid4())
        output = local_path("results/mvm_checks") / session_id
        output.mkdir(parents=True, exist_ok=False)
        (output / "board.ini").write_bytes(snapshot)
        manifest = {"session_id": session_id, "source": "hardware", "started_utc": now(),
                    "invocation": provenance, "status": "failed", **description}
        save_json(output / "start_manifest.json", manifest)
        try:
            connector = native_connector(config)
            from manager.service import converters
            result = collect(connector, converters, output, session_id, gui_ordered=args.gui_ordered)
            manifest.update(status=result["status"], impact_attempts=result["impact_attempts"],
                            valid_responses=result["valid_responses"], error=result.get("error"))
        except (Exception, KeyboardInterrupt) as exc:
            manifest["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            manifest["finished_utc"] = now()
            manifest["result_sha256"] = {p.name: sha256(p) for p in output.iterdir() if p.is_file()}
            save_json(output / "manifest.json", manifest)
        print(json.dumps({"session": output.relative_to(ROOT).as_posix(), "status": manifest["status"]}))
        return 0 if manifest["status"] == "completed" else 2
    finally:
        lock.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
