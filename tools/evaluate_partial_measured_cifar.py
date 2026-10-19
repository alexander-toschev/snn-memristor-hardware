"""Offline mixed model: replace exactly 300 Conv1 weights by saved W_eff.

This performs no acquisition, native SIM or hardware classification. Historical
effective weights are a linear approximation, not measured full-image outputs.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for dependency_dir in (ROOT / ".deps", ROOT / ".deps-classifier"):
    if dependency_dir.is_dir():
        sys.path.insert(0, str(dependency_dir))
SOURCES = {
    "modeled_saved_hardware_weights": {
        "path": "reports/2026-10-16/hardware-stability-comparison/SESSION2_WEIGHTS.csv",
        "sha256": "39fb9039fef3db8fb700feba2ffb58f151ce3cada6673e248bb14b6b4d763f7c",
        "provenance": "reports/2026-10-16/hardware-stability-comparison/PROVENANCE.json",
        "provenance_sha256": "bbb817611307e6aae09398788131be5bab75c7c64643dd3509928747889a6dc7",
        "session": "8c59eb76-c4e1-4d6f-8238-2b6622e226f6",
    },
    "modeled_saved_native_sim_weights": {
        "path": "reports/2026-10-16/no-bl8-simulator/WEIGHTS.csv",
        "sha256": "1945f9a4c445172e5a941220d554902248928ddccf634dc843616c4d26950ca9",
        "provenance": "reports/2026-10-16/no-bl8-simulator/PROVENANCE.json",
        "provenance_sha256": "ddfe1c20bc903c0021c21780a2b7e8c48ece845c8c67257c657a35b2eb786210",
        "session": "28aa7248-72db-428a-b0eb-92b1712a25af",
    },
}
EXPECTED = {(f, t) for f in range(4) for t in range(75)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_rows(rows, trained):
    """No missing, duplicate, extra or nonfinite coefficient is tolerated."""
    values = {}
    for row in rows:
        key = (int(row["filter"]), int(row["tap"]))
        if key not in EXPECTED or key in values:
            raise ValueError("Duplicate or out-of-scope coefficient: " + str(key))
        target, effective = float(row["weight"]), float(row["effective"])
        if not math.isfinite(target) or not math.isfinite(effective):
            raise ValueError("Nonfinite weight: " + str(key))
        if target != float(trained[key[0]][key[1]]):
            raise ValueError("Saved target differs from pinned checkpoint: " + str(key))
        values[key] = effective
    if set(values) != EXPECTED:
        raise ValueError("Incomplete measured coefficients: require exactly 300")
    return values


def read_values(repo, source, trained):
    for path_key, hash_key in (("path", "sha256"), ("provenance", "provenance_sha256")):
        if sha(repo / source[path_key]) != source[hash_key]:
            raise ValueError("Source SHA mismatch: " + source[path_key])
    with (repo / source["path"]).open(newline="", encoding="utf-8") as stream:
        return validate_rows(csv.DictReader(stream), trained)


def replace_weights(original, values):
    if set(values) != EXPECTED or any(not math.isfinite(v) for v in values.values()):
        raise ValueError("Exactly 300 finite coefficients required")
    if list(original.shape) != [64, 3, 5, 5]:
        raise ValueError("Unexpected Conv1 shape")
    modified = original.clone()
    flat = modified.reshape(64, 75)
    for (f, tap), value in values.items():
        flat[f, tap] = value
    return modified


def first_layer_trace(conv, state, api):
    import torch
    voltage = torch.zeros_like(conv[0])
    currents, voltages, spikes = [], [], []
    for t in range(12):
        current = api.bn(conv[t].unsqueeze(0), state, f"bn_layers.0.{t}").squeeze(0)
        spk, voltage = api.lif(current, voltage)
        currents.append(current)
        voltages.append(voltage)
        spikes.append(spk)
    return tuple(torch.stack(v) for v in (currents, voltages, spikes))


def errors(actual, reference):
    import torch
    delta = actual - reference
    return {"rmse": float(torch.sqrt((delta.double() ** 2).mean())),
            "max_abs": float(delta.abs().max()), "elements": actual.numel()}


def compare_trace(actual, reference):
    return {"bn_current": errors(actual[0], reference[0]),
            "lif_membrane_post_reset": errors(actual[1], reference[1]),
            "lif_spike_mismatches": int((actual[2] != reference[2]).sum()),
            "lif_spike_elements": actual[2].numel(),
            "actual_spikes": int(actual[2].sum()), "baseline_spikes": int(reference[2].sum())}


def evaluate(repo):
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, str(repo))
    import full_cifar_classifier as api
    torch.set_num_threads(2)
    state, model, _, _ = api.assets(repo)
    weight = state["conv_layers.0.weight"]
    weights = {name: replace_weights(weight, read_values(repo, source, weight.reshape(64, 75)))
               for name, source in SOURCES.items()}
    cases = []
    with torch.no_grad():
        for index in range(10):
            # Reuses the pinned fixture, image hashes and class-remap checks.
            _, _, case, image = api.assets(repo, index)
            x = api.encode(image)
            baseline_conv = F.conv2d(x, weight)
            baseline_output = api.downstream(baseline_conv, state)
            original = model(x[:, None])
            if any(not torch.equal(a, b) for a, b in zip(baseline_output, original)):
                raise ValueError("Digital baseline differs from original model forward")
            baseline_trace = first_layer_trace(baseline_conv, state, api)
            entry = {"fixture_case_index": index,
                     "case": {k: case[k] for k in ("original_test_index", "fine_label", "remapped_label", "image_chw_uint8_sha256")},
                     "cpu_baseline": api.describe(baseline_output), "models": {}}
            for name, modified in weights.items():
                conv = F.conv2d(x, modified)
                if not torch.equal(conv[:, 4:], baseline_conv[:, 4:]):
                    raise ValueError("Unmeasured filters changed")
                output = api.downstream(conv, state)
                trace = first_layer_trace(conv, state, api)
                entry["models"][name] = {
                    "result": api.describe(output),
                    "conv1_affected_filters": errors(conv[:, :4], baseline_conv[:, :4]),
                    "first_layer_affected_filters": compare_trace(tuple(v[:, :4] for v in trace), tuple(v[:, :4] for v in baseline_trace)),
                    "output_lif_membrane": errors(output[0], baseline_output[0]),
                    "output_lif_spike_mismatches": int((output[1] != baseline_output[1]).sum()),
                    "output_lif_spike_elements": output[1].numel(),
                    "twd_score": errors(output[2], baseline_output[2]),
                }
            cases.append(entry)
    summary = {}
    for name in ["cpu_baseline", *SOURCES]:
        counts = {}
        for decoder in ("first_spike", "membrane", "twd"):
            results = [c["cpu_baseline"] if name == "cpu_baseline" else c["models"][name]["result"] for c in cases]
            correct = sum(r["predicted_subset_indices"][decoder] == c["case"]["remapped_label"] for c, r in zip(cases, results))
            agreement = sum(r["predicted_subset_indices"][decoder] == c["cpu_baseline"]["predicted_subset_indices"][decoder] for c, r in zip(cases, results))
            counts[decoder] = {"correct": correct, "total": 10, "fixture_accuracy": correct / 10,
                               "prediction_agreement_with_cpu": agreement / 10}
        summary[name] = counts
    return {"schema": "partial-measured-weights-cifar-model-v1", "source": "offline_mixed_digital_model",
            "hardware_acquisition_performed": False, "native_simulator_executed": False,
            "hardware_classifier_accuracy_measured": False, "full_dataset_accuracy_measured": False,
            "replaced_conv1_coefficients": 300, "total_conv1_coefficients": 4800,
            "unchanged_conv1_coefficients": 4500, "affected_filters": [0, 1, 2, 3],
            "weight_field": "effective", "additional_weight_fitting": False,
            "assumptions": ["Saved W_eff is treated as a stationary linear weight across every spatial position and T12.",
                            "This does not reproduce fresh hardware ADC, drift, nonlinear interactions or repeat noise.",
                            "All computations and all downstream layers are digital; no missing hardware outputs are filled.",
                            "Ten preselected fixtures are diagnostic, not a full or randomly sampled test-set estimate."],
            "lif": {"tau": .5, "threshold": 1., "hard_reset": 0., "negative_voltage_preserved": True,
                    "physical_substeps_do_not_advance_lif": True},
            "sources": SOURCES, "classifier_pins": api.PINS,
            "classifier_sha256": sha(repo / "full_cifar_classifier.py"),
            "evaluator_sha256": sha(Path(__file__)), "torch_version": torch.__version__,
            "summary": summary, "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="New repo-relative JSON path")
    args = parser.parse_args()
    from pathlib import PureWindowsPath
    requested = Path(args.output)
    output = (ROOT / requested).resolve()
    if requested.is_absolute() or PureWindowsPath(args.output).drive or PureWindowsPath(args.output).root or ROOT not in output.parents:
        raise ValueError("Output must remain repo-relative")
    if output.exists():
        raise ValueError("Preserve previous output")
    report = evaluate(ROOT)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": args.output, "source": report["source"], "summary": report["summary"]}))


if __name__ == "__main__":
    main()
