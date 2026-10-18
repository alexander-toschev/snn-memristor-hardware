from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict
import time

import torch

# project imports
ROOT = Path(__file__).resolve().parent.parent
AGENT_DIR = Path(__file__).resolve().parent
for p in (ROOT, AGENT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from agent_common import (
    append_jsonl,
    best_readout,
    config_hash,
    ensure_dir,
    merge_dicts,
    parse_overrides,
    read_json,
    slugify,
    utc_now_iso,
    write_json,
)

from snn_mnist_net import Cfg, run_experiment, save_snn  # noqa: E402
from csnn_mnist_net import (
    CSCfg,
    build_csnn,
    build_encoder_from_cfg as build_csnn_encoder,
    _spikes_flat_to_hw,
    load_csnn_weights_into,
)  # noqa: E402
from label_map import build_label_map, save_label_map  # noqa: E402
from evaluation import eval_readouts_from_net  # noqa: E402


SAFE_SUMMARY_METRICS = [
    "spikes_per_sample",
    "synops_per_sample",
    "v_updates_per_sample",
    "energy_proxy_per_sample",
    "winners_unique",
    "winner_HHI",
]


def build_cfg(overrides: Dict[str, Any]) -> Cfg:
    valid = {name for name in Cfg.__dataclass_fields__.keys()}
    unknown = sorted(set(overrides) - valid)
    if unknown:
        raise ValueError(f"Unknown Cfg fields: {unknown}")
    return Cfg(**overrides)


def build_cs_cfg(overrides: Dict[str, Any]) -> CSCfg:
    valid = {name for name in CSCfg.__dataclass_fields__.keys()}
    unknown = sorted(set(overrides) - valid)
    if unknown:
        raise ValueError(f"Unknown CSCfg fields: {unknown}")
    return CSCfg(**overrides)



def run_single_experiment(
    overrides: Dict[str, Any],
    outdir: str | Path,
    *,
    n_calib: int = 2000,
    n_train_counts: int = 60000,
    n_test_counts: int = 10000,
    activity_log_every: int = 1000,
    skip_label_map: bool = False,
    skip_eval: bool = False,
    verbose: bool = True,
    progress: bool = True,
) -> Dict[str, Any]:
    arch = str(overrides.get("arch", "fc")).lower()
    if arch == "csnn":
        cfg = build_cs_cfg({k: v for k, v in overrides.items() if k != "arch"})
    else:
        cfg = build_cfg({k: v for k, v in overrides.items() if k != "arch"})
    outdir = ensure_dir(outdir)
    cfg_dict = dict(overrides)
    cfg_digest = config_hash(cfg_dict)
    run_id = f"{utc_now_iso().replace(':', '').replace('-', '')}_{cfg_digest}"
    run_dir = ensure_dir(outdir / run_id)

    status_path = run_dir / "status.json"
    registry_path = outdir / "registry.jsonl"

    base_payload: Dict[str, Any] = {
        "run_id": run_id,
        "created_at": utc_now_iso(),
        "status": "running",
        "config_hash": cfg_digest,
        "cfg": cfg_dict,
        "run_dir": str(run_dir),
    }
    write_json(status_path, base_payload)

    live_log_path = run_dir / "live.log"
    def _status_update(**fields):
        payload = {**read_json(status_path), **fields}
        payload["updated_at"] = utc_now_iso()
        write_json(status_path, payload)
        try:
            with open(live_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({k: payload.get(k) for k in ("status","stage","i","n","pct","eta_seconds") if k in payload}, ensure_ascii=False) + "\n")
        except BrokenPipeError:
            pass

    try:
        if arch == "csnn":
            # Minimal CSNN training loop (v0): 1 conv layer + STDP.
            from bindsnet.datasets import MNIST
            from torchvision import transforms
            from bindsnet.network.monitors import Monitor

            net, input_layer, lif_layer, connection = build_csnn(cfg)
            encoder = build_csnn_encoder(cfg)
            device = cfg.torch_device()
            # Ensure the conv input adapter is available even if label_map is skipped.
            from csnn_mnist_net import _spikes_flat_to_hw

            transform = transforms.Compose([transforms.ToTensor()])
            ds = MNIST(root="./data", train=True, download=True, transform=transform)
            n_train = min(int(getattr(cfg, "N", 12000)), len(ds))
            T = int(cfg.time)

            mon_X = Monitor(input_layer, state_vars=("s",), time=T)
            mon_H = Monitor(lif_layer, state_vars=("s",), time=T)
            net.add_monitor(mon_X, name="mon_X")
            net.add_monitor(mon_H, name="mon_H")

            S_out = 0
            last_report = time.time()
            # Resume: skip training if resume_checkpoint provided.
            if getattr(cfg, "resume_checkpoint", None):
                _status_update(stage="resume_checkpoint")
                load_csnn_weights_into(net, connection, lif_layer, str(cfg.resume_checkpoint))
                n_train = 0
            else:
                _status_update(stage="train", i=0, n=int(n_train), pct=0.0)

            activity_log_path = run_dir / "activity.jsonl"

            for i in range(n_train):
                x = ds[i]["image"].to(device)
                spikes = encoder(x)
                # spikes should already be [T,1,1,28,28] when encoder_out_format=TBNCHW
                if hasattr(spikes, "dim") and spikes.dim() == 3:
                    spikes_hw = _spikes_flat_to_hw(spikes, device)
                else:
                    spikes_hw = spikes
                net.run(inputs={"Input": spikes_hw}, time=T)

                # Per-sample spike count: sum over time from monitor, then reset monitor buffers.
                ssum = None
                try:
                    sH = mon_H.get("s")  # expected [T,B,C,H,W]
                    if torch.is_tensor(sH):
                        ssum = int(sH.sum().item())
                        mon_H.reset_state_variables()
                except Exception:
                    ssum = None

                if ssum is None:
                    # Fallback: last-step spikes only.
                    s_layer = getattr(lif_layer, "s", None)
                    ssum = int(s_layer.sum().item()) if torch.is_tensor(s_layer) else 0
                S_out += ssum

                if activity_log_every and ((i + 1) % int(activity_log_every) == 0):
                    try:
                        in_spikes = None
                        if torch.is_tensor(spikes_hw):
                            in_spikes = int(spikes_hw.sum().item())

                        # monitor cross-check (same as ssum); keep None to avoid confusion
                        ssum_mon = None

                        v_mean = None
                        v_max = None
                        try:
                            v = getattr(lif_layer, "v", None)
                            if torch.is_tensor(v) and v.numel() > 0:
                                v_mean = float(v.mean().item())
                                v_max = float(v.max().item())
                        except Exception:
                            pass

                        wsum = None
                        wmin = None
                        wmax = None
                        if hasattr(connection, "w") and torch.is_tensor(connection.w):
                            wsum = float(connection.w.detach().abs().sum().item())
                            wmin = float(connection.w.detach().min().item())
                            wmax = float(connection.w.detach().max().item())

                        rec_extra = {
                            "in_spikes": in_spikes,
                            "w_abs_sum": wsum,
                            "w_min": wmin,
                            "w_max": wmax,
                            "v_mean": v_mean,
                            "v_max": v_max,
                            "spikes_mon": ssum_mon,
                        }
                    except Exception:
                        rec_extra = {}
                    try:
                        theta_mean = None
                        theta = getattr(lif_layer, "theta", None)
                        if theta is not None and hasattr(theta, "mean"):
                            theta_mean = float(theta.mean().item())
                        rec = {
                            "i": int(i + 1),
                            "spikes": int(ssum),
                            "spikes_per_sample": float(ssum),
                            "theta_mean": theta_mean,
                            "t": utc_now_iso(),
                            **rec_extra,
                        }
                        with open(activity_log_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        _status_update(stage="train", activity_spikes=float(ssum), activity_theta_mean=theta_mean)
                    except Exception:
                        pass

                net.reset_state_variables()

                now = time.time()
                if now - last_report >= 30:
                    pct = 100.0 * (i + 1) / max(1, n_train)
                    it_s = (i + 1) / max(1e-6, (now - (last_report - 30)))  # rough
                    eta = int((n_train - (i + 1)) / max(1e-6, it_s))
                    _status_update(stage="train", i=int(i + 1), n=int(n_train), pct=float(pct), eta_seconds=int(eta))
                    last_report = now

            spikes_per_sample = float(S_out) / max(1, n_train)
            # propagate arch tag for evaluation pipeline
            cfg.arch = "csnn"

            train_summary = {
                "arch": "csnn",
                "N": int(n_train),
                "device": str(device),
                "time": int(cfg.time),
                "encoder": str(cfg.encoder),
                "spikes_per_sample": spikes_per_sample,
                "energy_proxy_per_sample": spikes_per_sample,
                "winners_unique": None,
                "winner_HHI": None,
            }
        else:
            train_summary, connection, lif_layer, net, encoder = run_experiment(
                cfg, verbose=verbose, progress=progress
            )

        # Save a checkpoint right after training so we can resume eval/readouts later.
        if arch == "csnn" and not getattr(cfg, "resume_checkpoint", None):
            _status_update(stage="checkpoint_after_train")
            early_ckpt = run_dir / "model_after_train.pt"
            save_snn(
                str(early_ckpt),
                cfg,
                connection,
                lif_layer,
                train_summary=train_summary,
                eval_summary=None,
                notes={"run_id": run_id, "config_hash": cfg_digest, "stage": "after_train"},
            )

        label_map_path = None
        label_map_summary = None
        label_map = None
        if not skip_label_map:
            _status_update(stage="label_map")
            if arch == "csnn":
                from csnn_mnist_net import _spikes_flat_to_hw

                def _enc_hw(x):
                    return _spikes_flat_to_hw(encoder(x), cfg.torch_device())

                label_map = build_label_map(
                    net,
                    None,
                    lif_layer,
                    _enc_hw,
                    n_calib=n_calib,
                    T=cfg.time,
                    top_k=max(1, int(getattr(cfg, "top_k", 0)) if int(getattr(cfg, "top_k", 0)) > 0 else 3),
                    seed=cfg.seed,
                )
            else:
                label_map = build_label_map(
                    net,
                    None,
                    lif_layer,
                    encoder,
                    n_calib=n_calib,
                    T=cfg.time,
                    top_k=max(1, int(cfg.top_k) if int(cfg.top_k) > 0 else 3),
                    seed=cfg.seed,
                )
            label_map_path = run_dir / "label_map.pt"
            save_label_map(
                str(label_map_path),
                label_map,
                meta={
                    "run_id": run_id,
                    "n_calib": n_calib,
                    "time": cfg.time,
                },
            )
            label_map_summary = {
                "assigned_neurons": int((label_map >= 0).sum().item()),
                "total_neurons": int(label_map.numel()),
            }

        eval_summary = None
        if not skip_eval:
            _status_update(stage="eval")
            def _eval_status_cb(stage: str, **kw):
                # kw: step, write_pos, batch_size, n_samples
                i = int(kw.get("write_pos", 0) + kw.get("batch_size", 0))
                n = int(kw.get("n_samples", 0))
                pct = 100.0 * i / max(1, n) if n else None
                _status_update(stage=stage, i=i, n=n, pct=pct)

            eval_summary = eval_readouts_from_net(
                net,
                lif_layer,
                encoder,
                cfg,
                label_map=label_map,
                n_train_counts=n_train_counts,
                n_test_counts=n_test_counts,
                status_cb=_eval_status_cb,
            )

        best_name, best_acc = best_readout(eval_summary)
        ckpt_path = run_dir / "model.pt"
        save_snn(
            str(ckpt_path),
            cfg,
            connection,
            lif_layer,
            train_summary=train_summary,
            eval_summary=eval_summary,
            notes={
                "run_id": run_id,
                "config_hash": cfg_digest,
                "label_map_summary": label_map_summary,
            },
        )

        payload: Dict[str, Any] = {
            **base_payload,
            "status": "ok",
            "finished_at": utc_now_iso(),
            "checkpoint_path": str(ckpt_path),
            "label_map_path": str(label_map_path) if label_map_path else None,
            "label_map_summary": label_map_summary,
            "best_readout_name": best_name,
            "best_readout_acc": best_acc,
            "train_summary": train_summary,
            "eval_summary": eval_summary,
            "summary": {
                k: train_summary.get(k)
                for k in SAFE_SUMMARY_METRICS
                if isinstance(train_summary, dict) and k in train_summary
            },
        }
        write_json(run_dir / "summary.json", payload)
        write_json(status_path, payload)
        append_jsonl(registry_path, payload)
        return payload
    except Exception as exc:  # pragma: no cover - defensive runtime wrapper
        err_payload = {
            **base_payload,
            "status": "failed",
            "finished_at": utc_now_iso(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(run_dir / "summary.json", err_payload)
        write_json(status_path, err_payload)
        append_jsonl(registry_path, err_payload)
        raise



def main() -> None:
    parser = argparse.ArgumentParser(description="Run one SNN experiment and persist artifacts.")
    parser.add_argument("--outdir", default="runs_agent", help="Root directory for run folders and registry.jsonl")
    parser.add_argument("--config-json", help="Path to JSON file with Cfg overrides")
    parser.add_argument("--set", action="append", default=[], help="Cfg override, e.g. --set time=300")
    parser.add_argument(
        "--resume-from",
        help=(
            "Path to an existing run directory to resume from. "
            "Loads <run_dir>/model_after_train.pt and skips training (N is forced to 0)."
        ),
    )
    parser.add_argument("--n-calib", type=int, default=2000)
    parser.add_argument("--n-train-counts", type=int, default=60000)
    parser.add_argument("--n-test-counts", type=int, default=10000)
    parser.add_argument("--activity-log-every", type=int, default=1000)
    parser.add_argument("--skip-label-map", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    file_overrides = read_json(args.config_json) if args.config_json else {}
    cli_overrides = parse_overrides(args.set)
    overrides = merge_dicts(file_overrides, cli_overrides)

    if args.resume_from:
        resume_dir = Path(args.resume_from)
        ckpt = resume_dir / "model_after_train.pt"
        if not ckpt.exists():
            raise SystemExit(f"--resume-from: missing checkpoint: {ckpt}")
        # Do not force N=0: label_map/eval use cfg.N for dataset sizing.
        # Training will be skipped inside run_single_experiment when resume_checkpoint is provided.
        overrides = merge_dicts(overrides, {
            "resume_checkpoint": str(ckpt),
        })

    if not overrides:
        raise SystemExit("No config overrides provided. Pass --config-json or --set key=value.")

    payload = run_single_experiment(
        overrides,
        outdir=args.outdir,
        n_calib=args.n_calib,
        n_train_counts=args.n_train_counts,
        n_test_counts=args.n_test_counts,
        activity_log_every=args.activity_log_every,
        skip_label_map=args.skip_label_map,
        skip_eval=args.skip_eval,
        verbose=not args.quiet,
        progress=not args.no_progress,
    )
    print(json.dumps({
        "run_id": payload["run_id"],
        "status": payload["status"],
        "best_readout_name": payload.get("best_readout_name"),
        "best_readout_acc": payload.get("best_readout_acc"),
        "run_dir": payload["run_dir"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
