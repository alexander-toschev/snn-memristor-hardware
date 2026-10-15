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


def collect(connector, converters, output, session_id):
    """No driver/interface access. Preserve partial records even after an exception."""
    board = EXPECTED_BOARD
    summary = {"status": "failed", "impact_attempts": 0, "valid_responses": 0,
               "readbacks": [], "mvm": [], "row_statistics": [], "conductance_comparison": [],
               "calibration_verified": False, "acceptance_threshold": None,
               "zero_offset_correction_applied": False}
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
                    expected_id = 0 if mode == 9 else task_id
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
                for wl in ROWS:
                    for bl in COLS:
                        for repeat in range(3):
                            summary["readbacks"].append(impact(9, wl, bl, .1, stage=stage, repeat=repeat))
                if stage == "after":
                    break
                means = {(wl, bl): statistics.mean(r["conductance_siemens"] for r in summary["readbacks"]
                         if r["command_si"]["wl"] == wl and r["command_si"]["bl"] == bl)
                         for wl in ROWS for bl in COLS}
                for wl in ROWS:
                    for repeat in range(5):
                        for label, active in VECTORS.items():
                            vector = [.1 if bl in active else 0.0 for bl in range(32)]
                            current = sum(decoded["native_decoded_v"] * means[wl, bl] for bl in active)
                            predicted_output = board["sum_gain"] * current
                            if current > .005 or predicted_output > .8 * board["vol_ref_adc"]:
                                raise ValueError("Pre-read model exceeds 5 mA / 80% ADC reference bound")
                            summary["mvm"].append(impact(10, wl, 0, vector, stage="mvm", vector=label,
                                repeat=repeat, predicted_current_a=current, predicted_output_v=predicted_output))
            for wl in ROWS:
                groups = {label: [r for r in summary["mvm"] if r["command_si"]["wl"] == wl and r["vector"] == label]
                          for label in VECTORS}
                values = {label: {"adc": stats([r["raw_adc"] for r in rows]),
                                 "output_v": stats([r["output_voltage_v"] for r in rows])}
                          for label, rows in groups.items()}
                epsilon = {unit: values["AB"][unit]["mean"] - values["A"][unit]["mean"]
                           - values["B"][unit]["mean"] + values["zero"][unit]["mean"] for unit in ("adc", "output_v")}
                summary["row_statistics"].append({"wl": wl, "vectors": values,
                    "offset_adjusted_superposition_epsilon": epsilon,
                    "epsilon_definition": "mean(AB)-mean(A)-mean(B)+mean(zero); descriptive, no pass threshold"})
                for bl in COLS:
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
    args = parser.parse_args(argv)
    if not (args.run_hardware or args.dry_run):
        parser.print_help()
        return 0
    if sha256(local_path("read_all.py")) != READ_ALL_SHA256:
        raise ValueError("Unreviewed read_all source")
    config, snapshot, hashes = preflight(INI)
    for filename in ("experiment.py", "mvm_check.py", "native_conversions.py", "experiment_hardware.py"):
        hashes[filename] = sha256(local_path(filename))
    description = {"modes": [9, 10], "wl": list(ROWS), "bl": list(COLS), "nominal_v": .1,
                   "mode9_count": 24, "mode10_count": 40, "total_impacts": 64,
                   "weights_programmed": False, "source_sha256": hashes}
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
            result = collect(connector, converters, output, session_id)
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
