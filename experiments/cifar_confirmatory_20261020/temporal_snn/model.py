"""TemporalCSNN – end-to-end temporal spiking network with variable depth.

Architecture
------------
Input:  [T, B, 3, 32, 32]  (latency-encoded spikes, bright→early)

n_conv_layers=1:
  Conv1  3→c1, 5×5, no pad  → [T*B, c1, 28, 28]
  BN1 / LIF_conv[0]          → spikes s1  [T, B, c1, 28, 28]
  AvgPool 2×2                → [T, B, c1, 14, 14]
  Flatten → FC1 → BN_fc1 → Dropout → LIF_fc → FC2 → LIF_out

n_conv_layers=2 (default, matches all prior runs):
  Conv1  3→c1, 5×5, no pad  → [T*B, c1, 28, 28]
  BN1 / LIF_conv[0]          → pool → [T, B, c1, 14, 14]
  Conv2  c1→c2, 3×3, pad=1  → [T, B, c2, 14, 14]
  BN2 / LIF_conv[1]          → pool → [T, B, c2, 7, 7]
  Flatten → FC1 → BN_fc1 → Dropout → LIF_fc → FC2 → LIF_out

n_conv_layers=3:
  Conv1 … pool → [c1, 14, 14]
  Conv2 … pool → [c2, 7, 7]
  Conv3  c2→c3, 3×3, pad=1  → [T, B, c3, 7, 7]   (NO pool after conv3)
  BN3 / LIF_conv[2]          → [T, B, c3, 7, 7]
  Flatten → FC1 → BN_fc1 → Dropout → LIF_fc → FC2 → LIF_out

Pool policy:
  layer 0: always pool  (28→14)
  layer 1: pool if n_conv_layers >= 2  (14→7)
  layer 2: NO pool  (avoid odd-dimension backward issue: 7→3)

Parameter naming (used by e-prop grad dict):
  conv_layers.{i}.weight   for each conv layer i
  bn_layers.{i}.weight/bias
  fc1.weight, fc2.weight, fc2.bias  (unchanged)
  bn_fc1.weight/bias
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple

from .lif import LIFCell


def _parse_wta_layers(raw: str, n_conv_layers: int) -> set:
    """Parse wta_layers string into a set of 0-based conv layer indices."""
    raw = str(raw).strip().lower()
    if not raw or raw == "none":
        return set()
    if raw == "all":
        return set(range(n_conv_layers))
    result = set()
    for part in raw.replace(",", " ").split():
        part = part.strip()
        if part in ("conv1", "c1", "1"):
            result.add(0)
        elif part in ("conv2", "c2", "2"):
            result.add(1)
        elif part in ("conv3", "c3", "3"):
            result.add(2)
        elif part.isdigit():
            result.add(int(part) - 1)
    return result


class TWDDecoder(nn.Module):
    """Temporal Weighting Decoder (Che et al., ICML 2026).

    Replaces argmin(first_spike_time) with a learned temporal scoring:
        score[b, c] = Σ_t  softplus(α[t]) · spike[t, b, c]

    α[t] are T trainable scalars, initialized as decreasing so that
    earlier spikes get higher initial weight.  The network learns to
    refine this distribution end-to-end via CE(score, target).
    """

    def __init__(self, T: int):
        super().__init__()
        # Decreasing init: α[0]=T, α[1]=T-1, ..., α[T-1]=1  (before softplus)
        init_w = torch.arange(T, 0, -1, dtype=torch.float32)   # [T]
        self.alpha = nn.Parameter(init_w)

    def forward(self, spikes: Tensor) -> Tensor:
        """spikes: [T, B, K]  →  score: [B, K]"""
        # Keep weights positive; shape [T, 1, 1] for broadcasting
        w = F.softplus(self.alpha)[:, None, None]
        return (w * spikes).sum(0)   # [B, K]

    def predict(self, spikes: Tensor) -> Tensor:
        """[T, B, K] → [B] class predictions."""
        return self.forward(spikes).argmax(1)


class TemporalCSNN(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        n_classes: int = 10,
        input_h: int = 32,
        input_w: int = 32,
        c1_out: int = 32,
        c2_out: int = 64,
        fc1_out: int = 256,
        tau: float = 0.5,
        v_threshold: float = 1.0,
        surrogate_alpha: float = 2.0,
        dropout_fc: float = 0.3,
        n_conv_layers: int = 2,
        c3_out: int = 128,
        c4_out: int = 256,
        wta_enable: bool = False,
        wta_layers: str = "conv1",
        v_threshold_out: float = -1.0,  # -1 = same as v_threshold
        ettfs_init: bool = False,       # ETTFS-init: scale weights by sqrt(T)
        T_for_init: int = 16,           # T used for ETTFS scaling
        use_twd: bool = False,          # Temporal Weighting Decoder
        use_bntt: bool = False,         # BatchNorm Through Time
        use_gap: bool = False,          # Global Average Pooling instead of Flatten
    ):
        super().__init__()
        assert n_conv_layers in (1, 2, 3, 4), f"n_conv_layers must be 1, 2, 3, or 4, got {n_conv_layers}"
        self.n_conv_layers = n_conv_layers
        self.n_classes = n_classes
        self.tau = tau
        self.v_threshold = v_threshold
        self.surrogate_alpha = surrogate_alpha

        # ── Channel list and conv params ──────────────────────────────────
        all_channels = [in_channels, c1_out, c2_out, c3_out, c4_out]
        channels = all_channels[: n_conv_layers + 1]   # [in, c1], [in, c1, c2], [in, c1, c2, c3]

        kernels  = [5] + [3] * (n_conv_layers - 1)     # [5], [5,3], [5,3,3]
        paddings = [0] + [1] * (n_conv_layers - 1)     # [0], [0,1], [0,1,1]

        # Pool after each conv layer:
        #   n=1: [True]           (28→14)
        #   n=2: [True, True]     (28→14, 14→7)
        #   n=3: [True, True, False]  (28→14, 14→7, no pool – avoids odd-dim backward issue)
        # Pool after conv layers: pool the first 2, no pool for layers 3+
        # n=1: [T]       n=2: [T,T]   n=3: [T,T,F]   n=4: [T,T,F,F]
        self._pool_after: List[bool] = ([True] * min(n_conv_layers, 2) +
                                        [False] * max(0, n_conv_layers - 2))

        # ── Build conv / BN / LIF lists ───────────────────────────────────
        self.use_bntt = use_bntt
        self._T = T_for_init
        self.conv_layers = nn.ModuleList()
        self.bn_layers   = nn.ModuleList()
        self.lif_conv_cells = nn.ModuleList()
        lif_kwargs = dict(tau=tau, v_threshold=v_threshold, surrogate_alpha=surrogate_alpha)

        for i in range(n_conv_layers):
            self.conv_layers.append(
                nn.Conv2d(channels[i], channels[i + 1],
                          kernel_size=kernels[i], padding=paddings[i], bias=False)
            )
            if use_bntt:
                # One BN per timestep — each learns its own scale/shift
                self.bn_layers.append(
                    nn.ModuleList([nn.BatchNorm2d(channels[i + 1]) for _ in range(T_for_init)])
                )
            else:
                self.bn_layers.append(nn.BatchNorm2d(channels[i + 1]))
            self.lif_conv_cells.append(LIFCell(**lif_kwargs))

        self.pool = nn.AvgPool2d(2, 2)

        # ── Compute spatial dimensions after each layer ───────────────────
        # _conv_out_dims[i]: (h, w) after conv_i (before pool)
        # _pool_dims[i]:     (h, w) after pool_i (or same as conv_out if no pool)
        h, w = input_h, input_w
        self._conv_out_dims: List[Tuple[int, int]] = []
        self._pool_dims: List[Tuple[int, int]] = []
        for i in range(n_conv_layers):
            h = h - kernels[i] + 2 * paddings[i] + 1
            w = w - kernels[i] + 2 * paddings[i] + 1
            self._conv_out_dims.append((h, w))
            if self._pool_after[i]:
                h, w = h // 2, w // 2
            self._pool_dims.append((h, w))

        last_c        = channels[n_conv_layers]
        last_h, last_w = self._pool_dims[-1]
        self.use_gap  = use_gap
        if use_gap:
            self.gap      = nn.AdaptiveAvgPool2d(1)  # [B, C, H, W] -> [B, C, 1, 1]
            self._fc_in   = last_c
        else:
            self._fc_in   = last_c * last_h * last_w

        # ── FC block ─────────────────────────────────────────────────────
        self.fc1     = nn.Linear(self._fc_in, fc1_out, bias=False)
        if use_bntt:
            self.bn_fc1 = nn.ModuleList([nn.BatchNorm1d(fc1_out) for _ in range(T_for_init)])
        else:
            self.bn_fc1 = nn.BatchNorm1d(fc1_out)
        self.dropout = nn.Dropout(dropout_fc)
        self.fc2     = nn.Linear(fc1_out, n_classes, bias=True)

        # ── LIF cells for FC and output layers ────────────────────────────
        self.lif_fc_cell  = LIFCell(**lif_kwargs)   # replaces lif3_cell
        out_thresh = v_threshold if v_threshold_out < 0 else v_threshold_out
        self.lif_out_cell = LIFCell(tau=tau, v_threshold=out_thresh, surrogate_alpha=surrogate_alpha)

        # -- WTA --
        self.wta_enable = wta_enable
        self._wta_layers: set = _parse_wta_layers(wta_layers, n_conv_layers) if wta_enable else set()

        # -- ETTFS init config --
        self._ettfs_init = ettfs_init
        self._T_for_init = T_for_init

        # -- TWD decoder --
        self.twd: TWDDecoder | None = TWDDecoder(T_for_init) if use_twd else None

        self._init_weights()

    def _init_weights(self):
        import math
        scale = math.sqrt(self._T_for_init) if self._ettfs_init else 1.0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if self._ettfs_init:
                    m.weight.data.mul_(scale)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if self._ettfs_init:
                    m.weight.data.mul_(scale)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        x: [T, B, C, H, W]  latency-encoded spike input
        Returns:
            membranes : [T, B, n_classes]   output membrane potentials (for TET loss)
            spikes_out: [T, B, n_classes]   output spikes (for first-spike decoding)
        """
        T, B = x.shape[0], x.shape[1]

        v_conv = [None] * self.n_conv_layers
        v_fc = v_out = None

        mems_list:    list[Tensor] = []
        spk_out_list: list[Tensor] = []

        for t in range(T):
            h = x[t]   # [B, C, H, W]

            # ── Convolutional layers ──────────────────────────────────────
            for i in range(self.n_conv_layers):
                raw = self.conv_layers[i](h)
                i_conv = self.bn_layers[i][t](raw) if self.use_bntt else self.bn_layers[i](raw)
                if v_conv[i] is None:
                    v_conv[i] = torch.zeros_like(i_conv)
                s_conv, v_conv[i] = self.lif_conv_cells[i](i_conv, v_conv[i])
                # Hard WTA: per spatial position, keep only the channel with
                # max membrane potential; other spikes masked to 0.
                # Gradients still flow through v_conv (membrane is untouched).
                if self.wta_enable and i in self._wta_layers:
                    winner = v_conv[i].argmax(dim=1, keepdim=True)  # [B,1,H,W]
                    wta_mask = torch.zeros_like(s_conv).scatter_(1, winner, 1.0)
                    s_conv = s_conv * wta_mask
                h = self.pool(s_conv) if self._pool_after[i] else s_conv

            # ── FC1 ──────────────────────────────────────────────────────
            h_flat = self.gap(h).flatten(1) if self.use_gap else h.flatten(1)
            fc_raw = self.fc1(h_flat)
            bn_out = self.bn_fc1[t](fc_raw) if self.use_bntt else self.bn_fc1(fc_raw)
            i_fc = self.dropout(bn_out)
            if v_fc is None:
                v_fc = torch.zeros_like(i_fc)
            s_fc, v_fc = self.lif_fc_cell(i_fc, v_fc)

            # ── Output (LIF) ──────────────────────────────────────────────
            i_out = self.fc2(s_fc)
            if v_out is None:
                v_out = torch.zeros_like(i_out)
            s_out, v_out = self.lif_out_cell(i_out, v_out)

            mems_list.append(v_out)
            spk_out_list.append(s_out)

        spikes_out = torch.stack(spk_out_list)   # [T, B, K]
        mems_out   = torch.stack(mems_list)        # [T, B, K]

        if self.twd is not None:
            twd_score = self.twd(spikes_out)       # [B, K]
            return mems_out, spikes_out, twd_score
        return mems_out, spikes_out

    # ── Inference helpers ─────────────────────────────────────────────────

    @staticmethod
    def first_spike_pred(spikes_out: Tensor) -> Tensor:
        """Compute first-spike prediction from already-computed spikes.

        spikes_out: [T, B, K]
        Returns: [B] predicted class indices.
        """
        T, B, K = spikes_out.shape
        dev = spikes_out.device
        t_idx = torch.arange(T, device=dev, dtype=torch.float32)[:, None, None]
        fired_times = torch.where(
            spikes_out > 0.5,
            t_idx.expand(T, B, K),
            torch.full_like(spikes_out, float(T)),
        )
        first_fire, _ = fired_times.min(dim=0)
        return first_fire.argmin(dim=1)

    @torch.no_grad()
    def predict_first_spike(self, x: Tensor) -> Tensor:
        _, spikes_out = self.forward(x)
        return self.first_spike_pred(spikes_out)

    @torch.no_grad()
    def predict_membrane(self, x: Tensor) -> Tensor:
        membranes, _ = self.forward(x)
        return membranes.sum(0).argmax(1)
