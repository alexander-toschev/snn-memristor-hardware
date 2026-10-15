"""Offline four-phase CIFAR Conv1 hardware plan; no Connector or driver imports.

This is an offline report tool. Physical execution belongs to the separate
experiment_hardware.py entry point and its native Connector-only suite.
"""

import argparse
import json
from pathlib import Path, PureWindowsPath

ROOT = Path(__file__).resolve().parent


def path_in_repo(value):
    path, windows = Path(value), PureWindowsPath(value)
    if not value or path.is_absolute() or windows.drive or windows.root or ".." in path.parts or ".." in windows.parts:
        raise ValueError("Expected a repository-relative path")
    resolved = (ROOT / path).resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError("Path escapes the repository")
    return resolved


def describe(plan):
    programming, geometry = plan["programming"], plan["geometry"]
    blockers = []
    if programming["timing_mode"] == "required_125us" and not programming["installed_driver_uses_tms_tus"]:
        blockers.append("unsupported_timing: installed mode_7 ignores tms/tus; requested 125 us cannot be delivered")
    if geometry["physical_array_policy"] == "unresolved":
        blockers.append("array_mapping: A/B/A are logical simulator names; no physical array selector established")
    if not plan["execution"]["protocol_choices_resolved"]:
        blockers.append("methodological_clarification: user requested hardware execution; timing and array choices remain unresolved")
    first = plan["phases"]["phase1"]
    attempts = first["cell_count"] * first["target_count"] * first["cycles"]
    filter_runs = len(plan["phases"]["phase2"]["filters"]) + 64 + len(plan["phases"]["phase4"]["filters"]) * 3
    program_cycles = attempts + filter_runs * 150
    return {"plan_id": plan["plan_id"], "hardware_access": False,
            "protocol_choices_resolved": plan["execution"]["protocol_choices_resolved"],
            "physical_entry_point": "experiment_hardware.py",
            "blockers": blockers, "weights": plan["weights"], "weights_shape": plan["weights_shape"],
            "phase1_programming_attempts": attempts, "phase1_post_read_impacts": attempts * first["post_reads"],
            "phase2_filter_runs": 14, "phase3_filter_runs": 64, "phase4_filter_runs": 3,
            "mapping_programmed_cell_cycles": filter_runs * 150,
            "maximum_programming_pulses": program_cycles * 2 * programming["max_pulses_per_reset_or_set"],
            "mapping_mvm_impacts": filter_runs * plan["mvm"]["input_count"] * 3 * 2,
            "readback_adc_semantics": "mode_7 returns native ten-sample aggregate, not individual ADC samples",
            "conversions": {"voltage": "native convert_volt_to_dac + convert_dac_to_volt",
                            "readback": "native convert_adc_to_res at explicit read level; preserve clamp flags/raw ADC",
                            "mvm": plan["mvm"]["output_converter"],
                            "q": "sum_gain/R as in MemriBoard GUI; r2w/w2r are WR divider and must not be used"},
            "scope": plan["dataset_scope"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hardware_four_phase.json")
    parser.add_argument("--output", help="Optional NEW repository-relative JSON report")
    parser.add_argument("--run-hardware", action="store_true", help="Always blocked in this preparation tool")
    args = parser.parse_args(argv)
    plan = json.loads(path_in_repo(args.config).read_text(encoding="utf-8"))
    report = describe(plan)
    if args.output:
        destination = path_in_repo(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
    print(json.dumps(report, indent=2))
    if args.run_hardware:
        raise SystemExit("Physical execution blocked before hardware: " + "; ".join(report["blockers"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
