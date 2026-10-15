"""CIFAR Conv1 physical four-phase experiment through native MemriBoard.

One physical array is reused sequentially; native write duration is unknown.
Default phase1; no connection without --run-hardware. --all requires phase1
completion conditions before mapping. No accuracy of a full SNN is claimed.
"""

import argparse
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import uuid

from read_all import ROOT, INI, EXPECTED_BOARD, local_path, sha256, now, save_json, preflight, native_connector
from hardware_suite import HardwareSuite, ScientificGateError, FILTERS

CONFIG = "configs/hardware_four_phase.json"
LOCK_FILE = "results/read_all/.hardware.lock"
READ_ALL_SHA256 = "b30ae2f4dd3cd166dbc5cefb07b558127de2bbdba1f6674c40b1b45f4dd74d5f"


def launch_provenance(argv):
    """Read-only Git identity before creating session files; no network or checkout."""
    git = {"commit": None, "dirty_tracked": None, "untracked": None, "status_porcelain": None}
    try:
        def read_git(*arguments):
            return subprocess.run(["git", "-C", str(ROOT), *arguments], check=True,
                                  capture_output=True, text=True, timeout=10).stdout.rstrip("\r\n")
        git["commit"] = read_git("rev-parse", "--verify", "HEAD")
        git["status_porcelain"] = read_git("status", "--porcelain=v1", "--untracked-files=normal")
        lines = git["status_porcelain"].splitlines()
        git["dirty_tracked"] = any(not line.startswith("?? ") for line in lines)
        git["untracked"] = [line[3:] for line in lines if line.startswith("?? ")]
    except (OSError, subprocess.SubprocessError) as exc:
        git["error"] = f"{type(exc).__name__}: {exc}"
    return {"git": git, "process_argv": list(sys.argv),
            "hardware_argv": list(sys.argv[1:] if argv is None else argv)}


def validate_plan(plan):
    if plan["backend"] != "real" or plan["board_type"] != "rp5_fpga_python":
        raise ValueError("Expected the reviewed native real rp5_fpga_python backend")
    if plan["programming"]["timing_mode"] != "native_unknown":
        raise ValueError("Installed mode7 cannot implement a requested pulse width")
    if plan["geometry"]["physical_array_policy"] != "single_array_sequential_tiles":
        raise ValueError("No native physical array selector exists; use the resolved one-array protocol")
    if not plan["execution"]["user_requested_hardware_plan"] or not plan["execution"]["protocol_choices_resolved"]:
        raise ValueError("The agreed protocol choices must be recorded")
    # Algorithm constants are explicit: reject config drift instead of silently ignoring it.
    required = {"start_v": .3, "step_v": .01, "stop_v": 3.0,
                "max_pulses_per_reset_or_set": 271, "reset_sign": 0, "set_sign": 1,
                "reset_endpoint_q": .005, "target_q": [.01, .02, .03, .04, .05],
                "mapping_q_window": [.01, .05]}
    if any(plan["programming"][key] != value for key, value in required.items()):
        raise ValueError("Programming parameters differ from the reviewed staircase")
    if plan["mvm"]["input_voltage_v"] != .1 or plan["mvm"]["input_count"] != 92:
        raise ValueError("Unexpected MVM voltage/input count")
    if plan["geometry"]["cells"] != [[0, 2], [2, 6], [2, 4]] or plan["geometry"]["differential_wl_pair"] != [0, 2]:
        raise ValueError("Unexpected cells or WL pair")
    geometry = plan["geometry"]
    if (geometry["wl_count"] != 8 or geometry["bl_count"] != 32
            or geometry["programming_excluded_wl"] != [1]
            or geometry["tiles"] != [[0, 32], [32, 64], [64, 75]]
            or geometry["logical_tile_arrays"] != ["single_board"]*3):
        raise ValueError("Unexpected array geometry, exclusion, or tile mapping")
    if plan["phases"] != {
        "phase1": {"cell_count": 3, "target_count": 5, "cycles": 5, "post_reads": 5},
        "phase2": {"filters": list(FILTERS), "repetitions": 1},
        "phase3": {"filters": "all_64", "repetitions": 1},
        "phase4": {"filters": [41], "repetitions": 3, "report_prefix_mean_K": [1, 2, 3]},
    }:
        raise ValueError("Phase counts/filter lists differ from the implemented protocol")
    if plan["readback"] != {
        "mode": 7, "write_voltage_v": 0, "native_read_dac": 246,
        "native_adc_average_count": 10, "initial_reads_per_cell": 21,
        "mapping_immediate_reads_per_cell": 3, "pre_mvm_tile_reads_per_cell": 3,
    }:
        raise ValueError("Readback settings differ from the implemented protocol")
    if (plan["programming"]["submitted_pulse_width_s"] != 0
            or plan["programming"]["physical_pulse_width_s"] is not None
            or plan["programming"]["whole_session_pulse_budget"] != 6625950
            or plan["mvm"]["predicted_current_limit_a"] != .005
            or plan["mvm"]["adc_headroom_fraction"] != .8
            or plan["mvm"]["zero_offset_correction"] is not False):
        raise ValueError("Timing/budget or MVM bounds differ from the implemented protocol")


def qualification(path, hashes, config_hash, host):
    session = local_path(path)
    manifest = json.loads((session / "manifest.json").read_text(encoding="utf-8"))
    phase1 = json.loads((session / "phase1.json").read_text(encoding="utf-8"))
    if manifest["status"] != "completed" or manifest["source"] != "hardware" or manifest["host"] != host:
        raise ValueError("Phase1 dependency must be a completed physical session on this host")
    if manifest["plan_sha256"] != config_hash:
        raise ValueError("Phase1 was collected under a different plan")
    if manifest["result_sha256"].get("phase1.json") != sha256(session / "phase1.json"):
        raise ValueError("Phase1 result hash mismatch")
    if not phase1["completion_conditions_pass"] or len(phase1["attempts"]) != 75:
        raise ScientificGateError("Prior phase1 did not meet all 75 reachability conditions")
    if any(manifest["source_sha256"].get(name) != digest for name, digest in hashes.items()):
        raise ValueError("Phase1 source/INI fingerprints differ")
    return {"session": path, "manifest_sha256": sha256(session / "manifest.json"),
            "phase1_sha256": sha256(session / "phase1.json"), "precision_qualified": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    launch = parser.add_mutually_exclusive_group()
    launch.add_argument("--run-hardware", action="store_true")
    launch.add_argument("--dry-run", action="store_true")
    stage = parser.add_mutually_exclusive_group()
    stage.add_argument("--phase", choices=("phase1", "phase2", "phase3", "phase4"))
    stage.add_argument("--all", action="store_true")
    parser.add_argument("--phase1-session", help="Completed phase1 session, relative to repository")
    parser.add_argument("--max-pulses", type=int, default=6625950, help="Whole-session hard ceiling; may be reduced")
    args = parser.parse_args(argv)
    if not (args.run_hardware or args.dry_run):
        parser.print_help()
        return 0
    phases = ["phase1", "phase2", "phase3", "phase4"] if args.all else [args.phase or "phase1"]
    if not 1 <= args.max_pulses <= 6625950:
        raise ValueError("Pulse budget must be in 1..6625950")
    plan_bytes = local_path(CONFIG).read_bytes()
    plan = json.loads(plan_bytes.decode("utf-8"))
    validate_plan(plan)
    if sha256(local_path("read_all.py")) != READ_ALL_SHA256:
        raise ValueError("Unreviewed read_all helper source")
    ini, snapshot, hashes = preflight(plan["ini"])
    for filename in ("experiment.py", "experiment_hardware.py", "hardware_suite.py", "native_conversions.py"):
        hashes[filename] = sha256(local_path(filename))
    plan_hash = hashlib.sha256(plan_bytes).hexdigest()
    if sha256(local_path(plan["weights"])) != plan["weights_sha256"]:
        raise ValueError("Conv1 weight hash mismatch")
    host = socket.gethostname()
    dependency = None
    if "phase1" not in phases:
        if not args.phase1_session:
            raise ValueError("Mapping requires --phase1-session or --all")
        dependency = qualification(args.phase1_session, hashes, plan_hash, host)
    elif args.phase1_session:
        raise ValueError("A new phase1 does not use --phase1-session")
    if args.dry_run:
        print(json.dumps({"hardware_access": False, "phases": phases, "physical_array_count": 1,
                          "pulse_timing": "native_unknown", "max_pulses": args.max_pulses,
                          "phase1_dependency": dependency, "source_sha256": hashes}))
        return 0
    invocation = launch_provenance(argv)
    # Factory import only; hardware initialization occurs in open_port below.
    connector = native_connector(ini)
    from manager.service import converters
    import numpy as np
    weights = np.load(local_path(plan["weights"]), allow_pickle=False)
    if weights.shape != (64, 3, 5, 5) or weights.dtype != np.float32 or not np.isfinite(weights).all():
        raise ValueError("Expected finite float32 Conv1 weights [64,3,5,5]")
    lock = local_path(LOCK_FILE)
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(f"CIFAR four-phase {now()}\n")
    try:
        session_id = str(uuid.uuid4())
        output = local_path("results/hardware_four_phase") / session_id
        output.mkdir(parents=True, exist_ok=False)
        (output / "board.ini").write_bytes(snapshot)
        (output / "plan.json").write_bytes(plan_bytes)
        manifest = {"session_id": session_id, "source": "hardware", "host": host, "started_utc": now(),
                    "physical_array_id": "single_board", "physical_array_count": 1,
                    "pulse_timing": "native_unknown", "physical_pulse_width_s": None,
                    "phases_requested": phases, "phases_completed": [], "phase1_dependency": dependency,
                    "source_sha256": hashes, "plan_sha256": plan_hash, "weights_sha256": plan["weights_sha256"],
                    "max_pulses": args.max_pulses, "python": sys.version, "numpy": np.__version__,
                    "invocation": invocation, "status": "failed"}
        save_json(output / "start_manifest.json", manifest)
        suite = None
        try:
            suite = HardwareSuite(connector, EXPECTED_BOARD, output, session_id, converters, max_pulses=args.max_pulses)
            if not connector.open_port():
                raise ConnectionError("Connector.open_port returned false")
            for phase in phases:
                if phase == "phase1":
                    result = suite.phase1()
                    if not result["completion_conditions_pass"]:
                        raise ScientificGateError("Phase1 completion conditions failed")
                elif phase == "phase2":
                    suite.mapping(weights, FILTERS, phase)
                elif phase == "phase3":
                    suite.mapping(weights, range(64), phase)
                else:
                    suite.mapping(weights, (41,), phase, repetitions=3)
                manifest["phases_completed"].append(phase)
            manifest["status"] = "completed"
        except (Exception, KeyboardInterrupt) as exc:
            manifest.update(status="scientific_gate_failed" if isinstance(exc, ScientificGateError) else "failed",
                            error=f"{type(exc).__name__}: {exc}")
        finally:
            try:
                if not connector.close_port():
                    raise ConnectionError("Connector.close_port returned false")
            except Exception as exc:
                manifest.update(status="failed", close_error=f"{type(exc).__name__}: {exc}")
            if suite is not None:
                suite.close()
                manifest.update(impact_count=suite.next_id, programming_pulses=suite.pulses,
                                invalid_readings=suite.invalid_readings)
                save_json(output / "summary.json", suite.results)
            manifest["finished_utc"] = now()
            manifest["result_sha256"] = {p.name: sha256(p) for p in output.iterdir() if p.is_file()}
            save_json(output / "manifest.json", manifest)
        print(json.dumps({"session": output.relative_to(ROOT).as_posix(), "status": manifest["status"],
                          "phases_completed": manifest["phases_completed"], "error": manifest.get("error")}))
        return 0 if manifest["status"] == "completed" else 2
    finally:
        lock.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
