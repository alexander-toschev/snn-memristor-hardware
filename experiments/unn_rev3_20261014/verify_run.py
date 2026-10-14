"""Audit saved simulator artifacts with the standard library; never run a model.

Usage: python -m experiments.unn_rev3_20261014.verify_run results/simulator/SESSION
Creates audit.json and metrics.json exclusively. An incomplete or invalid run
produces an audit with passed=false and exits 2. A passing integrity audit is
not evidence of target accuracy, classifier accuracy, or hardware qualification.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import zlib

if __package__:
    from .paths import repo_path, relative_path
    from .trace_storage import load_storage, raw_parts, raw_lines
else:
    from paths import repo_path, relative_path
    from trace_storage import load_storage, raw_parts, raw_lines


PHASE2_FILTERS = [63, 41, 32, 30, 38, 43, 60, 47, 2, 9, 46, 12, 0, 1]
CELLS = {(0, 2), (2, 6), (2, 4)}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def audit_run(directory):
    problems, problem_counts = [], Counter()

    def issue(check, detail):
        problem_counts[check] += 1
        if len(problems) < 100:
            problems.append({"check": check, "detail": detail})

    def require(condition, check, detail):
        if not condition:
            issue(check, detail)

    def read_json(name):
        try:
            with (directory/name).open(encoding="utf-8-sig") as stream:
                result = json.load(stream)
            if not isinstance(result, dict):
                raise ValueError("JSON object required")
            return result
        except (OSError, ValueError) as exc:
            issue("artifact", f"Cannot read {name}: {type(exc).__name__}")
            return {}

    manifest, summary, protocol = read_json("manifest.json"), read_json("summary.json"), read_json("protocol.json")
    complete = manifest.get("status") == "completed"
    require(complete, "completion", "Manifest must report completed")
    require(manifest.get("source") == "simulator", "backend", "Manifest source must be simulator")
    require(manifest.get("session_id") == protocol.get("suite", {}).get("session_id")
            and bool(manifest.get("session_id")), "session", "Manifest/protocol session IDs must agree")
    require(manifest.get("completed_phases") == ["baseline", "phase1", "phase2", "phase3", "phase4"],
            "coverage", "All five ordered stages must be complete")

    checked_hashes = 0

    def check_hash(path_string, expected, label):
        nonlocal checked_hashes
        try:
            path = repo_path(path_string)
            actual = sha256(path)
        except (OSError, ValueError, TypeError):
            issue("hash", f"Cannot read repository path for {label}")
            return
        checked_hashes += 1
        require(isinstance(expected, str) and actual == expected, "hash", f"SHA256 mismatch: {label}")

    check_hash(relative_path(directory/"protocol.json"), manifest.get("protocol_sha256"), "protocol.json")
    for key in ("source_sha256", "array_sha256"):
        hashes = manifest.get(key, {})
        require(isinstance(hashes, dict) and bool(hashes), "hash", f"Missing {key}")
        if isinstance(hashes, dict):
            for path, digest in hashes.items():
                check_hash(path, digest, path)
    suite = protocol.get("suite", {})
    for kind in ("weights", "checkpoint"):
        name, digest = suite.get(f"{kind}_file"), manifest.get(f"{kind}_sha256")
        require(digest == suite.get(f"{kind}_sha256") and bool(digest), "hash", f"{kind} hash metadata mismatch")
        check_hash(name, digest, f"{kind} file")
    arrays = sorted(relative_path(path) for path in (directory/"arrays").glob("*.cb"))
    require(len(arrays) == 2 and arrays == sorted(manifest.get("array_sha256", {})),
            "arrays", "Exactly two array files must match the manifest")

    modes, phases, stages, statuses, arrays_seen = Counter(), Counter(), Counter(), Counter(), set()
    baseline_cells, char_reads, mapping_counts, pulse_budgets = Counter(), Counter(), Counter(), Counter()
    filter_counts, program_attempts = Counter(), Counter()
    raw_count = 0
    raw_file = directory/"raw_tasks.jsonl"
    storage = None
    storage_verified = False
    trace_identity = {}
    try:
        storage = load_storage(directory) if (directory / "trace_storage.json").exists() else None
        parts = raw_parts(directory, storage=storage)
        initial_sizes = [p.stat().st_size for p in parts]
        from contextlib import closing
        with closing(raw_lines(parts, storage=storage, verification=trace_identity)) as stream:
            for line_number, line in enumerate(stream, 1):
                raw_count += 1
                where = f"raw_tasks.jsonl:{line_number}"
                try:
                    row = json.loads(line)
                    task = row["task"]
                    mode, wl, bl = task["mode_flag"], task["wl"], task["bl"]
                    phase, stage = row["phase"], row.get("stage")
                    require(type(row["task_id"]) is int and row["task_id"] == line_number
                            and type(task["id"]) is int and task["id"] == row["task_id"], "task_id", where)
                    require(row["session_id"] == manifest.get("session_id"), "session", where)
                    require(row["source"] == "simulator", "backend", where)
                    require(row["array_id"] in ("A", "B"), "arrays", where)
                    if isinstance(row["array_id"], str):
                        arrays_seen.add(row["array_id"])
                    require(type(wl) is int and 0 <= wl < 8 and wl != 1
                            and row["wl"] == wl, "wl", where)
                    require(type(bl) is int and 0 <= bl < 32 and row["bl"] == bl, "bl", where)
                    require(type(mode) is int and mode in (7, 10), "mode", where)
                    require(row["status"] == "ok", "status", where)
                    require(finite(row["latency_s"]) and row["latency_s"] >= 0,
                            "timing", where)
                    adc = row["raw_adc"]
                    require(type(adc) is int and (0 if mode == 10 else 1) <= adc <= 16382,
                            "adc", where)
                    if mode == 7:
                        voltage = task["vol"]
                        require(finite(voltage) and 0 <= voltage <= 3.0, "write_voltage", where)
                        require(type(task["sign"]) is int and task["sign"] in (0, 1), "sign", where)
                        require(task["pulse_width"] == (125e-6 if voltage else 0), "pulse_duration", where)
                        require(finite(row.get("resistance_ohm")) and row["resistance_ohm"] > 0,
                                "resistance", where)
                        if voltage:
                            require(row.get("operation") in ("reset", "set")
                                    and task["sign"] == (0 if row.get("operation") == "reset" else 1),
                                    "pulse_operation", where)
                            key = (stage, phase, row.get("filter_id"), row.get("repetition"),
                                   row.get("tile"), row.get("tap"), row.get("target_q"),
                                   row.get("cycle"), row["array_id"], wl, bl, row.get("operation"))
                            pulse_budgets[key] += 1
                    elif mode == 10:
                        volts = task["vol"]
                        require(isinstance(volts, list) and len(volts) == 32
                                and all(finite(v) and 0 <= v <= .1 for v in volts), "mvm_voltage", where)
                        require(task["pulse_width"] == 0 and task["sign"] == 0, "mvm_task", where)
                    if phase in ("first_check", "repeatability"):
                        baseline_cells[(phase, wl, bl)] += 1
                        require(row["array_id"] == "A", "arrays", where)
                    if phase == "characterization_readback":
                        char_reads[(wl, bl, row["target_q"], row["cycle"])] += 1
                    if phase in ("mapping", "mapping_readback", "mvm"):
                        mapping_counts[(stage, phase)] += 1
                        filter_counts[(stage, phase, row["filter_id"], row["repetition"])] += 1
                        if phase == "mapping" and row.get("operation") == "before_reset":
                            program_attempts[(stage, row["filter_id"], row["repetition"])] += 1
                        expected_array = "B" if row["tile"] == 1 else "A"
                        require(row["array_id"] == expected_array and wl in (0, 2), "tile_array", where)
                    modes[mode] += 1
                    phases[phase] += 1
                    if stage:
                        stages[stage] += 1
                    statuses[row["status"]] += 1
                except (KeyError, TypeError, ValueError) as exc:
                    issue("raw_record", f"{where}: {type(exc).__name__}")
        require([p.stat().st_size for p in parts] == initial_sizes, "completion", "Raw files changed during audit")
        storage_verified = storage is not None
    except (OSError, ValueError, EOFError, zlib.error, KeyError, TypeError):
        issue("artifact", "Cannot read or validate raw trace storage")
    require(raw_count == manifest.get("raw_task_count") and raw_count > 0, "raw_count", "Raw/manifest counts disagree")
    require(arrays_seen == {"A", "B"}, "arrays", "Raw log must include A and B")
    require(all(count <= 271 for count in pulse_budgets.values()), "pulse_budget", "A reset or SET exceeds 271 pulses")
    expected_baseline = {(phase, wl, bl): count for wl, bl in CELLS
                         for phase, count in (("first_check", 1), ("repeatability", 20))}
    require(dict(baseline_cells) == expected_baseline, "baseline", "Expected 3 first and 60 repeat reads")
    expected_char = {(wl, bl, target, cycle): 5 for wl, bl in CELLS
                     for target in (.01, .02, .03, .04, .05) for cycle in range(5)}
    require(dict(char_reads) == expected_char, "characterization", "Expected 75 programming cycles and 375 post-reads")

    baseline, char = summary.get("baseline", {}), summary.get("phase1", {})
    require(baseline.get("read_count") == 63 and baseline.get("cell_count") == 3,
            "baseline", "Summary baseline count mismatch")
    require(char.get("attempt_count") == 75 and char.get("post_read_count") == 375
            and len(char.get("groups", [])) == 15, "characterization", "Summary characterization count mismatch")
    mapping_metrics = {}
    for label, filters, reps in (("phase2", PHASE2_FILTERS, 1), ("phase3", list(range(64)), 1), ("phase4", [41], 3)):
        stage = summary.get(label, {})
        records = stage.get("results", [])
        expected = {(fid, rep) for fid in filters for rep in range(reps)}
        observed = {(row.get("filter_id"), row.get("repetition")) for row in records}
        require(observed == expected and len(records) == len(expected), "coverage", f"{label}: filter/repetition coverage")
        require(stage.get("filter_count") == len(filters) and stage.get("repetitions") == reps,
                "coverage", f"{label}: declared counts")
        require(stage.get("array_count") == 2 and stage.get("tile_arrays") == ["A", "B", "A"],
                "arrays", f"{label}: declared array layout")
        for phase, expected_count in (("mapping_readback", 450*len(expected)), ("mvm", 552*len(expected))):
            require(mapping_counts[(label, phase)] == expected_count, "coverage", f"{label}: {phase} task count")
        for fid, rep in expected:
            require(filter_counts[(label, "mapping_readback", fid, rep)] == 450
                    and filter_counts[(label, "mvm", fid, rep)] == 552
                    and program_attempts[(label, fid, rep)] == 150,
                    "coverage", f"{label}: filter {fid} repetition {rep} raw counts")
        reconstructed = stage.get("reconstructed_weights", {})
        require(set(reconstructed) == {str(fid) for fid in filters}, "weights", f"{label}: reconstructed filter keys")
        for values in reconstructed.values():
            require(isinstance(values, list) and len(values) == reps
                    and all(isinstance(w, list) and len(w) == 75 and all(finite(v) for v in w) for w in values),
                    "weights", f"{label}: reconstructed weights shape/finite values")
        mapping_metrics[label] = {
            "filters": len(filters), "replicate_count": len(records),
            "target_crossings": sum(row.get("target_reached_count", 0) for row in records),
            "programming_attempts": 150*len(records),
            "reset_successes": sum(row.get("reset_success_count", 0) for row in records),
            "weight_historical_gate_passes": sum(row.get("weights", {}).get("historical_gate_pass") is True for row in records),
            "mvm_historical_gate_passes": sum(row.get("mvm_vs_ideal", {}).get("historical_gate_pass") is True for row in records),
        }
    means = summary.get("phase4", {}).get("prefix_means", [])
    require(len(means) == 3 and {(row.get("filter_id"), row.get("K")) for row in means} == {(41, 1), (41, 2), (41, 3)},
            "coverage", "Phase4 must contain filter41 prefix means K=1,2,3")
    metrics = {
        "session": relative_path(directory), "source": "simulator", "hardware_qualification": False,
        "baseline": baseline,
        "characterization": {
            "attempt_count": char.get("attempt_count"), "target_crossings": char.get("target_reached_count"),
            "groups": [{"wl": g.get("wl"), "bl": g.get("bl"), "target_q": g.get("target_q"),
                        "cycle_count": g.get("cycle_count"),
                        **{name: g[field]*1e6 for name, field in (("bias_uS", "bias_s"), ("std_uS_ddof1", "error_std_s_ddof1"), ("rmse_uS", "rmse_s")) if finite(g.get(field))}}
                       for g in char.get("groups", [])],
        },
        "mapping": mapping_metrics,
        "phase4_prefix_means": [{"filter_id": row.get("filter_id"), "K": row.get("K"),
                                 "weight_rmse": row.get("weights", {}).get("rmse"),
                                 "mvm_rmse": row.get("mvm_vs_ideal", {}).get("rmse"),
                                 "all_resets_succeeded": row.get("all_resets_succeeded")}
                                for row in means],
        "interpretation": "Target crossing is not target accuracy. Gates are historical simulator comparators. Synthetic MVM is not classifier accuracy. No hardware was measured.",
    }
    audit = {"session": relative_path(directory), "audited_utc": datetime.now(timezone.utc).isoformat(),
             "passed": not problem_counts, "run_complete": complete, "hardware_qualification": False,
             "raw_records": raw_count, "mode_counts": dict(modes), "phase_counts": dict(phases),
             "stage_counts": dict(stages), "status_counts": dict(statuses), "arrays": sorted(arrays_seen),
             "verified_hash_count": checked_hashes, "issue_counts": dict(problem_counts), "issues": problems,
             "trace_storage_verified": storage_verified,
             "trace_uncompressed_sha256": trace_identity.get("sha256"),
             "trace_uncompressed_bytes": trace_identity.get("bytes"),
             "interpretation": "passed checks artifact integrity and coverage only, not scientific target quality."}
    return audit, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", help="Session directory relative to the repository")
    args = parser.parse_args()
    directory = repo_path(args.session)
    if not directory.is_dir():
        parser.exit(2, "Session directory does not exist\n")
    if any((directory/name).exists() for name in ("audit.json", "metrics.json")):
        parser.exit(2, "audit.json or metrics.json already exists; refusing overwrite\n")
    audit, metrics = audit_run(directory)
    for name, value in (("audit.json", audit), ("metrics.json", metrics)):
        with (directory/name).open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps({"session": args.session, "passed": audit["passed"],
                      "run_complete": audit["run_complete"], "raw_records": audit["raw_records"],
                      "issue_counts": audit["issue_counts"]}))
    if not audit["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
