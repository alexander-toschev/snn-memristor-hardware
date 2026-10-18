# evaluation.py
from __future__ import annotations
import torch
from bindsnet.network.monitors import Monitor
from bindsnet.datasets import MNIST
from torchvision import transforms

from snn_mnist_net import SNNMeter

# ===== extra evaluation utilities =====
from typing import Dict, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from counts_readout import collect_counts_plus_fast, make_mnist_datasets
from readout_models import tfidf_from_counts, train_mlp_readout

__all__ = ["probe_readouts_counts"]

@torch.no_grad()
def evaluate_on_mnist(net, input_layer, lif_layer, encoder, label_map, T: int = 200, top_k: int = 3, n_test: int = 1000, seed: int = 999):
    for c in net.connections.values():
        if hasattr(c, "update_rule"):
            c.update_rule.nu = (torch.as_tensor(0.0), torch.as_tensor(0.0))

    lif_mon = Monitor(lif_layer, state_vars=("s",), time=T); net.add_monitor(lif_mon, name="lif_test_tmp")
    transform = transforms.Compose([transforms.ToTensor()])
    ds_test = MNIST(root="./data", train=False, download=True, transform=transform)
    idxs = list(range(min(n_test, len(ds_test))))

    correct = 0
    meter = SNNMeter()
    lm_t = torch.as_tensor(label_map, dtype=torch.long, device=lif_layer.s.device if hasattr(lif_layer,'s') else None)

    for i in idxs:
        torch.manual_seed(seed + i)
        x = ds_test[i]["image"]; y = int(ds_test[i]["label"])
        spikes_in = encoder(x)
        net.run(inputs={"Input": spikes_in}, time=T)

        s_full = lif_mon.get("s")              # [T,1,N]
        s2 = s_full[:,0,:].to(torch.float32)   # [T,N]
        valid_mask = (lm_t >= 0)

        if not valid_mask.any():
            pred = -1
        else:
            s2m = s2.clone(); s2m[:, ~valid_mask] = -1e9
            active_t = (s2[:, valid_mask].sum(dim=1) > 0)
            if active_t.any():
                winners_t = s2m[active_t].argmax(dim=1)
                class_seq = lm_t[winners_t].clamp_min(0)
                votes = torch.bincount(class_seq, minlength=10)
                pred = int(votes.argmax().item())
            else:
                pred = -1

        if pred == y: correct += 1

        k_meter = min(3, lif_layer.n)
        counts = s2.sum(0)
        winners_for_meter = torch.topk(counts, k=k_meter).indices.tolist() if counts.sum() > 0 else None
        meter.log_sample(s_full, spikes_in, lif_layer.n, T, winners=winners_for_meter)

        net.reset_state_variables(); lif_mon.reset_state_variables()

    acc = correct / len(idxs); rpt = meter.report()
    net.monitors.pop("lif_test_tmp", None)
    print(f"TEST accuracy: {acc:.3f}  | spikes/sample={rpt['spikes_per_sample']:.2f}  energy≈{rpt['energy_proxy_per_sample']:.1f}")
    return {"accuracy": acc, **rpt}


def _train_linear_readout(
    Xtr: torch.Tensor, ytr: torch.Tensor,
    Xte: torch.Tensor, yte: torch.Tensor,
    epochs: int = 25, batch_size: int = 512,
    lr: float = 1e-3, weight_decay: float = 1e-4,
    device: str | torch.device = "cpu",
) -> Tuple[nn.Module, float]:
    """Простая линейная голова (softmax). Возвращает (model, acc_test).

    Важно: чтобы не ловить OOM на больших матрицах признаков, тренируем стримингово
    через DataLoader, без создания гигантских промежуточных копий.
    """
    dev = torch.device(device)
    in_dim = int(Xtr.shape[1])
    head = nn.Linear(in_dim, 10).to(dev)
    opt = torch.optim.SGD(head.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss()

    # DataLoader читает батчи из уже созданных тензоров Xtr/ytr на CPU.
    # pin_memory ускоряет H2D при dev=cuda.
    loader = DataLoader(
        TensorDataset(Xtr, ytr),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(dev.type == "cuda"),
        drop_last=False,
    )

    head.train()
    for _ in range(int(epochs)):
        for xb, yb in loader:
            xb = xb.to(dev, non_blocking=True)
            yb = yb.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = crit(head(xb), yb)
            loss.backward()
            opt.step()

    head.eval()
    with torch.no_grad():
        # Evaluate in chunks to keep memory bounded.
        correct = 0
        total = 0
        te_loader = DataLoader(
            TensorDataset(Xte, yte),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(dev.type == "cuda"),
            drop_last=False,
        )
        for xb, yb in te_loader:
            xb = xb.to(dev, non_blocking=True)
            yb = yb.to(dev, non_blocking=True)
            pred = head(xb).argmax(1)
            correct += int((pred == yb).sum().item())
            total += int(yb.numel())
        acc = (correct / max(1, total))

    return head, float(acc)

def _pca_whiten_counts(
    Xtr_counts: torch.Tensor, Xte_counts: torch.Tensor, var_keep: float = 0.95
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    PCA по ковариации и whitening для counts. Возвращает (Ztr_w, Zte_w, eigvals, W, d).
    """
    Xtr_c = torch.log1p(Xtr_counts)
    Xte_c = torch.log1p(Xte_counts)
    mu = Xtr_c.mean(0, keepdim=True)
    sd = Xtr_c.std(0, keepdim=True).clamp_min(1e-6)
    Xtr_z = (Xtr_c - mu) / sd
    Xte_z = (Xte_c - mu) / sd

    C = (Xtr_z.T @ Xtr_z) / (Xtr_z.shape[0] - 1)
    eigvals, eigvecs = torch.linalg.eigh(C)   # по возрастанию
    idx = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[idx]; eigvecs = eigvecs[:, idx]
    cum = torch.cumsum(eigvals, dim=0) / eigvals.sum()
    d = int((cum < var_keep).sum().item() + 1)
    W = eigvecs[:, :d]
    Ztr = Xtr_z @ W
    Zte = Xte_z @ W
    Ztr_w = Ztr / torch.sqrt(eigvals[:d] + 1e-8)
    Zte_w = Zte / torch.sqrt(eigvals[:d] + 1e-8)
    return Ztr_w, Zte_w, eigvals, W, d

def probe_readouts_counts(
    Xtr: torch.Tensor, ytr: torch.Tensor,
    Xte: torch.Tensor, yte: torch.Tensor,
    n_hidden: int,
    mlp_hidden: int = 256, mlp_epochs: int = 30,
    *,
    status_cb=None,
) -> Dict[str, float]:
    """
    Пробует 3 пайплайна над spike-counts:
      1) counts -> log1p -> zscore -> Linear
      2) counts -> PCA(95%)+whiten -> Linear
      3) counts -> TF-IDF -> zscore -> MLP
    Возвращает dict с точностями.
    """
    N = n_hidden

    # 1) counts zscore + Linear
    Xtr_c = torch.log1p(Xtr[:, :N]); Xte_c = torch.log1p(Xte[:, :N])
    mu = Xtr_c.mean(0, keepdim=True); sd = Xtr_c.std(0, keepdim=True).clamp_min(1e-6)
    Xtr_z = (Xtr_c - mu) / sd; Xte_z = (Xte_c - mu) / sd
    if status_cb is not None:
        try:
            status_cb(stage="readout_linear")
        except Exception:
            pass
    _, acc_counts_lin = _train_linear_readout(Xtr_z, ytr, Xte_z, yte, device=(Xtr_z.device if hasattr(Xtr_z,'device') else "cpu"))

    # 2) PCA-whiten + Linear (optional, very heavy for large N)
    acc_pca_lin = None
    import os
    if os.environ.get("SNN_READOUT_PCA", "0") == "1":
        try:
            if status_cb is not None:
                try:
                    status_cb(stage="readout_pca")
                except Exception:
                    pass
            Ztr_w, Zte_w, *_ = _pca_whiten_counts(Xtr[:, :N], Xte[:, :N], var_keep=0.95)
            _, acc_pca_lin = _train_linear_readout(Ztr_w, ytr, Zte_w, yte)
        except Exception as e:
            print(f"[eval] PCAwhiten skipped: {type(e).__name__}: {e}")

    # 3) TF-IDF + MLP (optional)
    acc_tfidf_mlp = None
    import os
    if os.environ.get("SNN_READOUT_TFIDF_MLP", "0") == "1":
        try:
            Xtr_n, Xte_n, _, _, _ = tfidf_from_counts(Xtr[:, :N], Xte[:, :N])

            def _mlp_status_cb(**kw):
                if status_cb is not None:
                    try:
                        status_cb(stage="readout_mlp", **kw)
                    except Exception:
                        pass

            _, acc_tfidf_mlp = train_mlp_readout(
                Xtr_n, ytr, Xte_n, yte, hidden=mlp_hidden, dropout=0.2,
                epochs=mlp_epochs, batch_size=256,
                status_cb=_mlp_status_cb,
            )
        except Exception as e:
            print(f"[eval] TFIDF+MLP skipped: {type(e).__name__}: {e}")

    out = {
        "counts_zscore+Linear": acc_counts_lin,
    }
    if acc_tfidf_mlp is not None:
        out["TFIDF+MLP"] = acc_tfidf_mlp
    if acc_pca_lin is not None:
        out["PCAwhiten+Linear"] = acc_pca_lin
    return out

def eval_readouts_from_net(
    net, lif_layer, encoder, cfg, label_map=None,
    n_train_counts: int = 60000, n_test_counts: int = 10000,
    status_cb=None,
) -> Dict[str, float]:
    """
    Собирает counts(+WTA-hist) через counts_readout.collect_counts_plus,
    затем гоняет probe_readouts_counts над counts и возвращает метрики.
    """
    ds_train, ds_test = make_mnist_datasets()
    spikes_transform = None
    is_conv = False
    try:
        # Conv LIFNodes typically have shape=(C,H,W)
        is_conv = hasattr(lif_layer, "shape") and lif_layer.shape is not None and len(tuple(lif_layer.shape)) == 3
    except Exception:
        is_conv = False

    if is_conv:
        # Force conv input [T,B,1,28,28] regardless of encoder output.
        from csnn_mnist_net import _spikes_flat_to_hw
        dev = cfg.torch_device() if hasattr(cfg, "torch_device") else None
        spikes_transform = (lambda sp: _spikes_flat_to_hw(sp, dev or (sp.device if hasattr(sp,'device') else "cpu")))

    if label_map is None:
        # No label_map: disable WTA-hist features.
        label_map = [-1] * int(lif_layer.n)

    dev = cfg.torch_device() if hasattr(cfg, "torch_device") else None

    def _mk_on_batch(stage_name: str):
        if status_cb is None:
            return None
        def _cb(**kw):
            status_cb(stage=stage_name, **kw)
        return _cb

    # Use progress=False to avoid tqdm stdout issues (BrokenPipe) in background runs.
    # Stream large X matrices to disk-backed memmaps when requested.
    import os
    use_mm = os.environ.get("SNN_COUNTS_MEMMAP", "0") in ("1", "true", "True", "yes", "YES")
    if use_mm:
        from pathlib import Path
        mm_dir = Path(os.environ.get("SNN_COUNTS_MEMMAP_DIR", "."))
        mm_dir.mkdir(parents=True, exist_ok=True)
        # Attach flags to net for counts_readout.collect_counts_plus_fast.
        net._use_memmap = True
        net._memmap_path = str(mm_dir / "Xtr.memmap")

    Xtr, ytr = collect_counts_plus_fast(
        net, lif_layer, encoder, ds_train, n_train_counts,
        T=cfg.time, label_map=label_map, move_net=True, batch_size=128,
        encoder_rate_boost=cfg.encoder_rate_boost,
        spikes_transform=spikes_transform,
        device=dev,
        progress=False,
        on_batch=_mk_on_batch("collect_train_counts"),
    )
    if use_mm:
        net._memmap_path = str(mm_dir / "Xte.memmap")

    Xte, yte = collect_counts_plus_fast(
        net, lif_layer, encoder, ds_test, n_test_counts,
        T=cfg.time, label_map=label_map, move_net=True, batch_size=128,
        encoder_rate_boost=cfg.encoder_rate_boost,
        spikes_transform=spikes_transform,
        device=dev,
        progress=False,
        on_batch=_mk_on_batch("collect_test_counts"),
    )
    N = lif_layer.n

    # NOTE: probe_readouts_counts can be heavy / unstable in some environments.
    # Allow disabling it via env var for robust end-to-end runs.
    # When disabled, we still train the stable Linear readout.
    import os
    if os.environ.get("SNN_DISABLE_READOUT_PROBE", "0") not in ("0", "false", "False", "no", "NO"):
        if status_cb is not None:
            try:
                status_cb(stage="readout_probe_skipped")
            except Exception:
                pass
        # Reuse the same stable pipeline as probe_readouts_counts: log1p(counts) + zscore + Linear.
        # Support disk-backed memmaps (NumPy) to avoid OOM.
        N = int(lif_layer.n)
        import numpy as np
        if not torch.is_tensor(Xtr):
            Xtr = torch.from_numpy(np.asarray(Xtr))
        if not torch.is_tensor(Xte):
            Xte = torch.from_numpy(np.asarray(Xte))
        if not torch.is_tensor(ytr):
            ytr = torch.from_numpy(np.asarray(ytr)).long()
        if not torch.is_tensor(yte):
            yte = torch.from_numpy(np.asarray(yte)).long()
        Xtr_c = torch.log1p(Xtr[:, :N]); Xte_c = torch.log1p(Xte[:, :N])
        mu = Xtr_c.mean(0, keepdim=True)
        sd = Xtr_c.std(0, keepdim=True).clamp_min(1e-6)
        Xtr_z = (Xtr_c - mu) / sd
        Xte_z = (Xte_c - mu) / sd

        if status_cb is not None:
            try:
                status_cb(stage="readout_linear")
            except Exception:
                pass
        _, acc_counts_lin = _train_linear_readout(Xtr_z, ytr, Xte_z, yte, device=(dev or "cpu"))
        if status_cb is not None:
            try:
                status_cb(stage="readout_done")
            except Exception:
                pass
        return {"counts_zscore+Linear": float(acc_counts_lin)}

    if status_cb is not None:
        try:
            status_cb(stage="readout_probe")
        except Exception:
            pass

    accs = probe_readouts_counts(Xtr, ytr, Xte, yte, n_hidden=N, status_cb=status_cb)

    if status_cb is not None:
        try:
            status_cb(stage="readout_done")
        except Exception:
            pass

    return accs

