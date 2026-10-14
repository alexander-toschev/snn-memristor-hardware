"""Phase 3 — hybrid forward pass (14-of-64 Conv1 mapped to Phase 2 readback weights).

Loads the frozen checkpoint, replaces ONLY the 14 mapped filters' conv_layers.0.weight
with their reconstructed 75-tap readback weights from Phase 2 (readback_weight.npy),
keeps the other 50 filters + BN stats/biases/FC/TWD decoder clean, and runs single-shot
inference over the official CIFAR-100-subset test set + val sanity.

Reports clean FS vs hybrid FS, delta, and prediction-change fraction. Read-only w.r.t.
checkpoint / test set.
"""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/mnt/d/projects/snn-mnist")
sys.path.insert(0, str(REPO))

from temporal_snn.config import TemporalSNNCfg
from temporal_snn.dataset import (
    CIFAR100Subset,
    LatencyEncoder,
    make_test_dataloader,
)
from temporal_snn.model import TemporalCSNN
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

RUN_DIR = REPO / "runs_csnn/confirmatory_wide_bntt_20261006/twd1_sigma0p0_seed0"
P2_DIR = REPO / ".tmp/rev2_rehearsal_20261013/results"
OUT_DIR = REPO / ".tmp/rev2_rehearsal_20261013"
DEVICE = "cpu"

MAPPED = [63, 41, 32, 30, 38, 43, 60, 47, 2, 9, 46, 12, 0, 1]  # 14-of-64


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build_model(cfg: TemporalSNNCfg) -> TemporalCSNN:
    return TemporalCSNN(
        in_channels=3,
        n_classes=cfg.n_classes,
        c1_out=cfg.c1_out,
        c2_out=cfg.c2_out,
        fc1_out=cfg.fc1_out,
        tau=cfg.tau,
        v_threshold=cfg.v_threshold,
        surrogate_alpha=cfg.surrogate_alpha,
        dropout_fc=cfg.dropout_fc,
        n_conv_layers=getattr(cfg, "n_conv_layers", 2),
        c3_out=getattr(cfg, "c3_out", 128),
        wta_enable=getattr(cfg, "wta_enable", False),
        wta_layers=getattr(cfg, "wta_layers", "conv1"),
        v_threshold_out=getattr(cfg, "v_threshold_out", -1.0),
        ettfs_init=getattr(cfg, "ettfs_init", False),
        T_for_init=cfg.T,
        use_twd=getattr(cfg, "use_twd", False),
        use_bntt=getattr(cfg, "use_bntt", False),
        use_gap=getattr(cfg, "use_gap", False),
    )


def build_val_loader(cfg: TemporalSNNCfg, val_indices, batch_size=32):
    val_tf = transforms.Compose([transforms.ToTensor()])
    ds = CIFAR100Subset(cfg.data_dir, train=True, fine_labels=_parse_labels(cfg.dataset),
                        transform=val_tf, download=False)
    va_ds = Subset(ds, val_indices)
    return DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)


def _parse_labels(spec):
    return [int(x) for x in spec.split(":", 1)[1].split(",")]


@torch.no_grad()
def get_preds(model, loader, encoder, device):
    model.eval()
    preds, labels = [], []
    for imgs, lab in loader:
        imgs = imgs.to(device)
        lab = lab.to(device)
        spikes_in = encoder(imgs)
        fwd = model(spikes_in)
        spikes_out = fwd[1]
        pred = TemporalCSNN.first_spike_pred(spikes_out)
        preds.append(pred.cpu())
        labels.append(lab.cpu())
    return torch.cat(preds), torch.cat(labels)


def main():
    cfg_dict = json.loads((RUN_DIR / "cfg.json").read_text())
    cfg = TemporalSNNCfg(**cfg_dict)
    encoder = LatencyEncoder(T=cfg.T, x_min=cfg.x_min)
    device = torch.device(DEVICE)

    ckpt = torch.load(RUN_DIR / "best_fs.pt", map_location="cpu", weights_only=False)
    clean_conv1 = ckpt["conv_layers.0.weight"].clone()

    manifest = json.loads((RUN_DIR / "split_manifest.json").read_text())
    val_indices = manifest["val_indices"]
    val_loader = build_val_loader(cfg, val_indices, batch_size=32)
    test_loader = make_test_dataloader(cfg.dataset, cfg.data_dir, batch_size=32,
                                       num_workers=0, pin_memory=False, download=False)
    n_val = len(val_loader.dataset)
    n_test = len(test_loader.dataset)
    print(f"[loader] val={n_val} test={n_test} split_sha={manifest['index_sha256'][:12]}...")

    model = build_model(cfg).to(device)
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    assert not missing and not unexpected, f"state_dict mismatch: missing={missing} unexpected={unexpected}"
    model.eval()

    # --- clean baseline ---
    preds_test_clean, labels_test = get_preds(model, test_loader, encoder, device)
    preds_val_clean, labels_val = get_preds(model, val_loader, encoder, device)
    acc_test_clean = (preds_test_clean == labels_test).float().mean().item()
    acc_val_clean = (preds_val_clean == labels_val).float().mean().item()
    print(f"[clean] test FS={acc_test_clean:.4f}  val FS={acc_val_clean:.4f}")

    # --- verify Phase 2 readback weights match checkpoint scale ---
    maxabs = {}
    for f in MAPPED:
        w = clean_conv1[f].reshape(-1).numpy().astype(np.float64)
        maxabs[f] = float(np.max(np.abs(w)))
    assert np.allclose(acc_test_clean, 0.725, atol=1e-3) and np.allclose(acc_val_clean, 0.756, atol=1e-3), \
        f"clean baseline mismatch: test={acc_test_clean} val={acc_val_clean}"

    # --- hybrid: replace only the 14 mapped filters ---
    new_conv1 = clean_conv1.clone()
    rb_meta = {}
    for f in MAPPED:
        p = P2_DIR / f"filter_{f:02d}" / "readback_weight.npy"
        w = np.load(p)  # (75,) weight space
        assert w.shape == (75,), f"unexpected shape {w.shape} for filter {f}"
        assert np.all(np.isfinite(w)), f"nonfinite weight filter {f}"
        new_conv1[f] = torch.from_numpy(w.astype(np.float32)).reshape(3, 5, 5)
        # verify reconstruction scale against ideal (weight-space RMSE guard, informational)
        ideal = clean_conv1[f].reshape(-1).numpy().astype(np.float64)
        rb_meta[f] = dict(maxabs_w=maxabs[f],
                          w_rmse=float(np.sqrt(np.mean((w - ideal) ** 2))),
                          w_corr=float(np.corrcoef(w, ideal)[0, 1]))

    sd = dict(ckpt)
    sd["conv_layers.0.weight"] = new_conv1
    m_hyb = build_model(cfg).to(device)
    m_hyb.load_state_dict(sd)
    m_hyb.eval()
    preds_test_h, _ = get_preds(m_hyb, test_loader, encoder, device)
    preds_val_h, _ = get_preds(m_hyb, val_loader, encoder, device)
    acc_test_h = (preds_test_h == labels_test).float().mean().item()
    acc_val_h = (preds_val_h == labels_val).float().mean().item()
    changed_test = (preds_test_h != preds_test_clean).float().mean().item()
    changed_val = (preds_val_h != preds_val_clean).float().mean().item()
    print(f"[hybrid 14/64] test FS={acc_test_h:.4f} val FS={acc_val_h:.4f} "
          f"test_changed={changed_test:.4f} val_changed={changed_val:.4f}")

    result = {
        "checkpoint_sha256": sha256_file(RUN_DIR / "best_fs.pt"),
        "cfg_sha256": sha256_file(RUN_DIR / "cfg.json"),
        "split_index_sha256": manifest["index_sha256"],
        "n_test": n_test,
        "n_val": n_val,
        "clean": {"test_fs": acc_test_clean, "val_fs": acc_val_clean},
        "hybrid_14of64": {
            "test_fs": acc_test_h,
            "val_fs": acc_val_h,
            "test_fs_delta": acc_test_h - acc_test_clean,
            "val_fs_delta": acc_val_h - acc_val_clean,
            "test_prediction_change_fraction": changed_test,
            "val_prediction_change_fraction": changed_val,
        },
        "mapped_filters": MAPPED,
        "n_mapped": len(MAPPED),
        "n_clean": 64 - len(MAPPED),
        "readback_weight_meta": rb_meta,
        "scope_note": "software representation of MemriBoard write-noise (simulator programmed-readback "
                      "weights, K=1, erase-before-write); 14 characterized filters mapped, 50 filters clean",
    }
    out_path = OUT_DIR / "phase3_result.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\n[summary] clean test={acc_test_clean:.4f} val={acc_val_clean:.4f}  "
          f"hybrid test={acc_test_h:.4f} val={acc_val_h:.4f}  "
          f"delta={acc_test_h - acc_test_clean:+.4f}  changed={changed_test:.4f}")
    print(f"[written] {out_path}")


if __name__ == "__main__":
    main()
