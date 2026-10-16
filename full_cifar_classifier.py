"""Offline full CIFAR100-subset classifier; no driver imports or acquisition.

Hardware boundary is a complete, audited pre-BN Conv1 partial tensor. CPU
reference generation is a separate operation and cannot satisfy that boundary.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PureWindowsPath

import torch
import torch.nn.functional as F

LABELS = [0, 1, 2, 3, 5, 8, 13, 14, 17, 19]
TILES = [[0, 32], [32, 64], [64, 75]]
SHAPE = [12, 64, 28, 28, 3]
PINS = {
    "weights/best_fs.pt": "ee4fc4f58a3d010218116e9ad4c7ba808787959f9ed7687cd58cbb32d90f62fb",
}
PINS["weights/cfg.json"] = "94ea7d5f57c5fdf6e5f4e319c38b4921783cdc04f5f082fdca59c2e86a14c645"
PINS["weights/cifar100_conv1_fixture.json"] = "8cb9ec0fe30242328527073bb076f8019295d73a748343f58fc2a8671bf7c154"
PINS.update({
    "full_cifar_reference/__init__.py": "761a7fc9d64f62e2a87e4a2776df356b926a137a2c8105713639d7b2644c55c6",
    "full_cifar_reference/encoders.py": "16a532134eaf78055db355f531214daa2af888a7fe6f30ba905f0c01db96f431",
    "full_cifar_reference/lif.py": "61762a092c7c7ea1b21823ec01cda9fa24355ca2f6d3f1f65e535ec12fdda91d",
    "full_cifar_reference/temporal_csnn.py": "73ee710edc86666ba6fceab46644724793834d6b57e1f121d63098976dac9d89",
})
# No collector-specific full Conv1 export auditor exists yet. Structural checks
# below are preparation, not authentication of a self-declared hardware source.
TRUSTED_EXPORT_AUDITORS = frozenset()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def local(repo, value):
    if Path(value).is_absolute() or PureWindowsPath(value).drive or PureWindowsPath(value).root:
        raise ValueError("Only repo-relative paths are accepted")
    path = (repo / value).resolve()
    if not path.is_relative_to(repo.resolve()):
        raise ValueError("Path escapes repository")
    return path


def assets(repo, case_index=0):
    for name, digest in PINS.items():
        if sha(repo / name) != digest:
            raise ValueError("Artifact SHA mismatch: " + name)
    cfg = json.loads((repo / "weights/cfg.json").read_text())
    state = torch.load(repo / "weights/best_fs.pt", weights_only=True, map_location="cpu")
    from full_cifar_reference.temporal_csnn import TemporalCSNN
    model = TemporalCSNN(**{k: cfg[k] for k in (
        "c1_out", "c2_out", "c3_out", "fc1_out", "tau", "v_threshold",
        "surrogate_alpha", "dropout_fc", "n_conv_layers", "use_twd", "use_bntt",
        "use_gap", "v_threshold_out", "ettfs_init", "n_classes")}, T_for_init=cfg["T"])
    model.load_state_dict(state, strict=True)
    model.eval()
    cases = json.loads((repo / "weights/cifar100_conv1_fixture.json").read_text())["cases"]
    case = cases[case_index]
    image = torch.tensor(case["image_chw_uint8"], dtype=torch.uint8).reshape(3, 32, 32)
    digest = hashlib.sha256(bytes(image.flatten().tolist())).hexdigest()
    if digest != case["image_chw_uint8_sha256"]:
        raise ValueError("Image byte SHA mismatch")
    if LABELS[case["remapped_label"]] != case["fine_label"]:
        raise ValueError("Class remap mismatch")
    return state, model, case, image


def encode(image):
    x = image.to(torch.float32) / 255.0
    times = torch.floor((1.0 - x) * 11).long().clamp(0, 11)
    return ((torch.arange(12)[:, None, None, None] == times) & (x >= .1)).float()


def cpu_tiles(spikes, state):
    """CPU reference only: [T,F,Y,X,K], CHW flatten order, no BN."""
    patches = F.unfold(spikes, kernel_size=5)  # [12,75,784]
    w = state["conv_layers.0.weight"].reshape(64, 75)
    parts = [torch.matmul(w[:, a:b], patches[:, a:b]) for a, b in TILES]
    return torch.stack(parts, -1).reshape(SHAPE)


def lif(current, previous):
    v = .5 * previous + current
    s = (v >= 1.0).float()
    return s, v - s * v


def bn(x, state, prefix):
    return F.batch_norm(x, state[prefix + ".running_mean"], state[prefix + ".running_var"],
                        state[prefix + ".weight"], state[prefix + ".bias"],
                        training=False, eps=1e-5)


@torch.no_grad()
def downstream(conv1, state):
    """Complete pre-BN Conv1 only; sum all 75 taps before calling this function."""
    if list(conv1.shape) != SHAPE[:-1] or not torch.isfinite(conv1).all():
        raise ValueError("Complete finite Conv1 [12,64,28,28] required")
    vc = [None] * 3
    vf = vo = None
    membranes, spikes = [], []
    for t in range(12):
        h = conv1[t].unsqueeze(0)
        for layer in range(3):
            if layer:
                h = F.conv2d(h, state[f"conv_layers.{layer}.weight"], padding=1)
            h = bn(h, state, f"bn_layers.{layer}.{t}")
            if vc[layer] is None:
                vc[layer] = torch.zeros_like(h)
            h, vc[layer] = lif(h, vc[layer])
            if layer < 2:
                h = F.avg_pool2d(h, 2)
        h = F.linear(h.flatten(1), state["fc1.weight"])
        h = bn(h, state, f"bn_fc1.{t}")  # dropout disabled in eval
        if vf is None:
            vf = torch.zeros_like(h)
        h, vf = lif(h, vf)
        h = F.linear(h, state["fc2.weight"], state["fc2.bias"])
        if vo is None:
            vo = torch.zeros_like(h)
        h, vo = lif(h, vo)
        membranes.append(vo)
        spikes.append(h)
    mem, spk = torch.stack(membranes), torch.stack(spikes)
    twd = (F.softplus(state["twd.alpha"])[:, None, None] * spk).sum(0)
    return mem, spk, twd


def validate_payload(payload, manifest, image_sha):
    expected = {"schema": "full-conv1-measured-v1", "source": "hardware",
                "stage": "pre_bn", "units": "checkpoint_conv1", "shape": SHAPE,
                "tile_tap_ranges_chw": TILES, "bn_applied": False,
                "image_sha256": image_sha, "checkpoint_sha256": PINS["weights/best_fs.pt"],
                "calibration_frozen_before_acquisition": True,
                "cpu_replacement_used": False}
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError("Measured contract mismatch: " + key)
    if set(payload) != {"partials", "coverage"}:
        raise ValueError("Payload must contain only partials and coverage")
    p, coverage = payload["partials"], payload["coverage"]
    if not isinstance(p, torch.Tensor) or list(p.shape) != SHAPE or p.dtype != torch.float32:
        raise ValueError("Wrong partial shape/dtype")
    if not isinstance(coverage, torch.Tensor) or list(coverage.shape) != SHAPE or coverage.dtype != torch.bool:
        raise ValueError("Wrong coverage shape/dtype")
    if not coverage.all() or not torch.isfinite(p).all():
        raise ValueError("Missing/nonfinite measured partial; no CPU or zero fill allowed")
    return p.sum(-1)


def load_measured(repo, manifest_path, image_sha):
    if not TRUSTED_EXPORT_AUDITORS:
        raise NotImplementedError("Hardware import unavailable: no reviewed full-Conv1 collector/export auditor")
    manifest_file = local(repo, manifest_path)
    m = json.loads(manifest_file.read_text())
    # A collector-specific independent auditor must attest the dense export and
    # its raw provenance. This consumer does not pretend to audit ADC acquisition.
    for key in ("payload", "raw_manifest", "independent_audit"):
        item = m[key]
        if sha(local(repo, item["path"])) != item["sha256"]:
            raise ValueError("Evidence SHA mismatch: " + key)
    audit = json.loads(local(repo, m["independent_audit"]["path"]).read_text())
    if (audit.get("status") != "passed" or audit.get("errors") != [] or
        audit.get("auditor_sha256") not in TRUSTED_EXPORT_AUDITORS or
        audit.get("raw_manifest_sha256") != m["raw_manifest"]["sha256"] or
        audit.get("export_payload_sha256") != m["payload"]["sha256"] or
        audit.get("validated_contract") != "full-conv1-measured-v1"):
        raise ValueError("Independent full-coverage export audit required")
    payload = torch.load(local(repo, m["payload"]["path"]), weights_only=True, map_location="cpu")
    return validate_payload(payload, m, image_sha)


def describe(output):
    mem, spikes, score = output
    first = torch.where(spikes > .5, torch.arange(12)[:, None, None], 12).min(0).values
    predictions = {"first_spike": int(first.argmin(1)), "membrane": int(mem.sum(0).argmax(1)),
                   "twd": int(score.argmax(1))}
    return {"predicted_subset_indices": predictions,
            "predicted_cifar100_fine_labels": {k: LABELS[v] for k, v in predictions.items()},
            "all_output_classes_silent": bool(not spikes.any()),
            "first_spike_winner_count": int((first == first.min()).sum()),
            "output_membranes_post_reset": mem.squeeze(1).tolist(),
            "output_spikes": spikes.squeeze(1).tolist(), "twd_scores": score[0].tolist()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--cpu-reference", action="store_true")
    mode.add_argument("--measured-manifest")
    parser.add_argument("--case-index", type=int, choices=range(10), default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parent
    output = local(repo, args.output)
    if output.exists():
        raise ValueError("Output already exists; preserve previous report")
    torch.set_num_threads(2)
    state, model, case, image = assets(repo, args.case_index)
    x = encode(image)
    if args.cpu_reference:
        with torch.no_grad():
            original = model(x[:, None])
            direct = F.conv2d(x, state["conv_layers.0.weight"])
            partials = cpu_tiles(x, state)
            result = downstream(partials.sum(-1), state)
            direct_result = downstream(direct, state)
        differences = {name: float((a - b).abs().max()) for name, a, b in
                       zip(("membrane", "spikes", "twd"), result, original)}
        direct_diff = max(float((a-b).abs().max()) for a,b in zip(direct_result, original))
        if direct_diff > 1e-6 or differences["spikes"] != 0 or differences["membrane"] > 1e-5:
            raise ValueError("CPU golden mismatch: " + str(differences))
        validation = {"original_forward_max_abs_differences": differences,
                      "direct_downstream_max_abs_difference": direct_diff,
                      "tiled_vs_direct_conv1_max_abs": float((partials.sum(-1)-direct).abs().max())}
        source = "cpu_reference_only"
    else:
        result = downstream(load_measured(repo, args.measured_manifest, case["image_chw_uint8_sha256"]), state)
        validation, source = {"measured_manifest": args.measured_manifest}, "hardware_conv1_digital_remaining_network"
    report = {"schema": "full-cifar100-subset-classifier-v1", "source": source,
              "classifier_sha256": sha(Path(__file__)),
              "hardware_acquisition_performed": False, "full_dataset_accuracy_measured": False,
              "class_fine_labels_in_output_order": LABELS,
              "case": {k: case[k] for k in ("original_test_index", "fine_label", "remapped_label", "image_chw_uint8_sha256")},
              "artifact_sha256": PINS, "logical_partial_count_per_image": 1806336,
              "hardware_native_call_estimate": "not specified: depends on repeats, paired zeros, microtiles, programming",
              "validation": validation, "result": describe(result)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": args.output, "source": source, "validation": validation,
                      "predictions": report["result"]["predicted_subset_indices"]}))


if __name__ == "__main__":
    main()
