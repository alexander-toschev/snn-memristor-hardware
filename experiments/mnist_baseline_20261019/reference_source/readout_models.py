# readout_models.py
# PCA-whitening and TF-IDF(+MLP) readout utilities for SNN spike-count features.

from __future__ import annotations
from typing import Tuple, Optional

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

__all__ = [
    "pca_whiten_counts",
    "tfidf_from_counts",
    "train_mlp_readout",
]


@torch.no_grad()
def pca_whiten_counts(
    Xtr_counts: Tensor,
    Xte_counts: Tensor,
    var_keep: float = 0.95,
    eps: float = 1e-8,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, int, Tensor, Tensor]:
    """
    PCA on per-neuron counts with whitening.

    Returns:
        Ztr_w, Zte_w, W, eigvals_sorted, d, mu, sigma
    """
    # 1) log1p + per-feature z-score
    Xtr_c = torch.log1p(Xtr_counts)
    Xte_c = torch.log1p(Xte_counts)

    mu = Xtr_c.mean(0, keepdim=True)
    sigma = Xtr_c.std(0, keepdim=True).clamp_min(1e-6)
    Xtr_z = (Xtr_c - mu) / sigma
    Xte_z = (Xte_c - mu) / sigma

    # 2) PCA on covariance
    n = Xtr_z.shape[0]
    C = (Xtr_z.T @ Xtr_z) / (n - 1)  # [N,N]
    eigvals, eigvecs = torch.linalg.eigh(C)  # ascending
    idx = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    cum = torch.cumsum(eigvals, dim=0) / eigvals.sum().clamp_min(eps)
    d = int((cum < var_keep).sum().item() + 1)

    W = eigvecs[:, :d]                     # [N,d]
    Ztr = Xtr_z @ W                        # [n_train, d]
    Zte = Xte_z @ W                        # [n_test, d]
    Ztr_w = Ztr / torch.sqrt(eigvals[:d] + eps)
    Zte_w = Zte / torch.sqrt(eigvals[:d] + eps)
    return Ztr_w, Zte_w, W, eigvals[:d].clone(), d, mu, sigma


@torch.no_grad()
def tfidf_from_counts(
    Xtr_counts: Tensor,
    Xte_counts: Tensor,
    eps: float = 1e-6,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Build TF-IDF features from spike-counts and z-score normalize.

    Returns:
        Xtr_n, Xte_n, idf, mu, sigma
    """
    # TF: log counts
    TF_tr = torch.log1p(Xtr_counts)
    TF_te = torch.log1p(Xte_counts)

    # IDF: number of samples that activated neuron at least once
    df = (Xtr_counts > 0).sum(dim=0).to(torch.float32)
    idf = torch.log((Xtr_counts.shape[0] + 1.0) / (df + 1.0)).clamp_min(0.0)

    Xtr_tfidf = TF_tr * idf
    Xte_tfidf = TF_te * idf

    mu = Xtr_tfidf.mean(0, keepdim=True)
    sigma = Xtr_tfidf.std(0, keepdim=True).clamp_min(eps)
    Xtr_n = (Xtr_tfidf - mu) / sigma
    Xte_n = (Xte_tfidf - mu) / sigma
    return Xtr_n, Xte_n, idf, mu, sigma


def train_mlp_readout(
    Xtr: Tensor,
    ytr: Tensor,
    Xte: Tensor,
    yte: Tensor,
    hidden: int = 256,
    dropout: float = 0.2,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: Optional[str] = None,
    status_cb=None,
) -> Tuple[nn.Module, float]:
    """
    Simple MLP readout training on prepared features.

    Returns:
        model, test_accuracy
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    N = Xtr.shape[1]
    model = nn.Sequential(
        nn.Linear(N, hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, 10),
    ).to(dev)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss()

    ds = TensorDataset(Xtr.to(dev), ytr.long().to(dev))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model.train()
    for ep in range(int(epochs)):
        for xb, yb in loader:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
        if status_cb is not None:
            try:
                status_cb(epoch=int(ep + 1), epochs=int(epochs), loss=float(loss.detach().cpu().item()))
            except Exception:
                pass

    model.eval()
    with torch.no_grad():
        pred = model(Xte.to(dev)).argmax(1).cpu()
        acc = float((pred == yte.cpu().long()).float().mean().item())

    return model, acc
