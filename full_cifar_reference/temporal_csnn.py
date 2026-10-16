"""TemporalCSNN — Convolutional SNN with first-spike / TWD decoding.

Architecture (default, n_conv_layers=2):
    Input:  [T, B, 3, 32, 32]  (latency-encoded spikes)
    Conv1   3→c1,  5×5, no-pad  → [T*B, c1, 28, 28]
    BN1 / LIF_conv[0]           → pool → [T, B, c1, 14, 14]
    Conv2   c1→c2, 3×3, pad=1  → [T, B, c2, 14, 14]
    BN2 / LIF_conv[1]           → pool → [T, B, c2, 7, 7]
    Flatten → FC1 → BN_fc1 → Dropout → LIF_fc
    → FC2 → LIF_out
    → membranes [T, B, K],  spikes [T, B, K]

Decoding options:
    1. First-spike:  argmin over t where spikes[t,b,k] > 0
    2. Membrane sum: argmax of membranes.sum(0)
    3. TWD (Temporal Weighting Decoder): learned α[t] weights on spikes

References:
    Che et al. (2026) "Temporally-Aware Training for First-Spike Inference
    on Resistive Memory Arrays" — Paper 1 of this repository.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple, Optional

from .lif import LIFCell, atan_spike


# ─────────────────────────────────────────────────────────────────────────────
# Temporal Weighting Decoder (TWD)
# ─────────────────────────────────────────────────────────────────────────────

class TWDDecoder(nn.Module):
    """Temporal Weighting Decoder.

    Replaces argmin(first_spike_time) with a learned temporal scoring:
        score[b, c] = Σ_t  softplus(α[t]) · spike[t, b, c]

    α[t] are T trainable scalars, initialised decreasing so earlier spikes
    get higher initial weight.  The network refines this distribution via
    CE(score, target).

    Args:
        T: Number of timesteps.

    Reference:
        TWD concept from our experiments 2026-07-02.
    """

    def __init__(self, T: int):
        super().__init__()
        init_w = torch.arange(T, 0, -1, dtype=torch.float32)
        self.alpha = nn.Parameter(init_w)

    def forward(self, spikes: Tensor) -> Tensor:
        """
        Args:
            spikes: [T, B, K]

        Returns:
            score: [B, K]
        """
        w = F.softplus(self.alpha)[:, None, None]   # [T, 1, 1]
        return (w * spikes).sum(0)                   # [B, K]

    def predict(self, spikes: Tensor) -> Tensor:
        """[T, B, K] → [B] predicted class indices."""
        return self.forward(spikes).argmax(1)


# ─────────────────────────────────────────────────────────────────────────────
# TemporalCSNN
# ─────────────────────────────────────────────────────────────────────────────

class TemporalCSNN(nn.Module):
    """Convolutional SNN trained with BPTT + surrogate gradients.

    Args:
        in_channels:      Input image channels (3 for RGB).
        n_classes:        Number of output classes.
        input_h, input_w: Spatial dimensions of input (32×32 for CIFAR).
        c1_out, c2_out:   Output channels for Conv1 and Conv2.
        c3_out:           Output channels for Conv3 (only when n_conv_layers=3).
        fc1_out:          Hidden units in FC1.
        tau:              LIF membrane decay.
        v_threshold:      LIF firing threshold.
        surrogate_alpha:  ATan surrogate steepness.
        dropout_fc:       Dropout probability after FC1 BN.
        n_conv_layers:    Number of conv layers: 1, 2, or 3.
        use_twd:          If True, attach a TWDDecoder (forward returns 3-tuple).
        use_bntt:         If True, use BatchNorm Through Time (separate BN per t).
        use_gap:          If True, use Global Average Pooling before FC1.
        v_threshold_out:  Output-layer threshold (-1 = same as v_threshold).
        ettfs_init:       If True, scale weights by sqrt(T_for_init) after Kaiming.
        T_for_init:       T used for ETTFS scaling and BNTT allocation.
    """

    def __init__(
        self,
        in_channels: int = 3,
        n_classes: int = 10,
        input_h: int = 32,
        input_w: int = 32,
        c1_out: int = 32,
        c2_out: int = 64,
        c3_out: int = 128,
        fc1_out: int = 256,
        tau: float = 0.5,
        v_threshold: float = 1.0,
        surrogate_alpha: float = 2.0,
        dropout_fc: float = 0.3,
        n_conv_layers: int = 2,
        use_twd: bool = False,
        use_bntt: bool = False,
        use_gap: bool = False,
        v_threshold_out: float = -1.0,
        ettfs_init: bool = False,
        T_for_init: int = 16,
    ):
        super().__init__()
        assert n_conv_layers in (1, 2, 3), f"n_conv_layers must be 1, 2, or 3"
        self.n_conv_layers = n_conv_layers
        self.use_bntt = use_bntt

        all_ch = [in_channels, c1_out, c2_out, c3_out]
        channels = all_ch[:n_conv_layers + 1]
        kernels  = [5] + [3] * (n_conv_layers - 1)
        paddings = [0] + [1] * (n_conv_layers - 1)
        # Pool after first two conv layers only (avoids odd spatial dims)
        self._pool_after: List[bool] = (
            [True] * min(n_conv_layers, 2) + [False] * max(0, n_conv_layers - 2)
        )

        lif_kw = dict(tau=tau, v_threshold=v_threshold, surrogate_alpha=surrogate_alpha)

        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.lif_conv_cells = nn.ModuleList()

        for i in range(n_conv_layers):
            self.conv_layers.append(
                nn.Conv2d(channels[i], channels[i+1],
                          kernel_size=kernels[i], padding=paddings[i], bias=False)
            )
            if use_bntt:
                self.bn_layers.append(
                    nn.ModuleList([nn.BatchNorm2d(channels[i+1]) for _ in range(T_for_init)])
                )
            else:
                self.bn_layers.append(nn.BatchNorm2d(channels[i+1]))
            self.lif_conv_cells.append(LIFCell(**lif_kw))

        self.pool = nn.AvgPool2d(2, 2)

        # Compute spatial dimensions
        h, w = input_h, input_w
        self._pool_dims: List[Tuple[int, int]] = []
        for i in range(n_conv_layers):
            h = h - kernels[i] + 2 * paddings[i] + 1
            w = w - kernels[i] + 2 * paddings[i] + 1
            if self._pool_after[i]:
                h, w = h // 2, w // 2
            self._pool_dims.append((h, w))

        last_c = channels[n_conv_layers]
        last_h, last_w = self._pool_dims[-1]

        self.use_gap = use_gap
        if use_gap:
            self.gap = nn.AdaptiveAvgPool2d(1)
            fc_in = last_c
        else:
            fc_in = last_c * last_h * last_w

        self.fc1 = nn.Linear(fc_in, fc1_out, bias=False)
        if use_bntt:
            self.bn_fc1 = nn.ModuleList([nn.BatchNorm1d(fc1_out) for _ in range(T_for_init)])
        else:
            self.bn_fc1 = nn.BatchNorm1d(fc1_out)
        self.dropout = nn.Dropout(dropout_fc)
        self.fc2 = nn.Linear(fc1_out, n_classes, bias=True)

        self.lif_fc_cell = LIFCell(**lif_kw)
        out_thresh = v_threshold if v_threshold_out < 0 else v_threshold_out
        self.lif_out_cell = LIFCell(tau=tau, v_threshold=out_thresh, surrogate_alpha=surrogate_alpha)

        self.twd: Optional[TWDDecoder] = TWDDecoder(T_for_init) if use_twd else None

        self._ettfs_init = ettfs_init
        self._T_for_init = T_for_init
        self._init_weights()

    def _init_weights(self):
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

    def forward(
        self, x: Tensor
    ) -> "Tuple[Tensor, Tensor] | Tuple[Tensor, Tensor, Tensor]":
        """
        Args:
            x: [T, B, C, H, W]  latency-encoded spike input.

        Returns:
            Without TWD:  (membranes [T,B,K],  spikes_out [T,B,K])
            With TWD:     (membranes [T,B,K],  spikes_out [T,B,K],  twd_score [B,K])
        """
        T, B = x.shape[0], x.shape[1]
        v_conv = [None] * self.n_conv_layers
        v_fc = v_out = None
        mems_list, spk_list = [], []

        for t in range(T):
            h = x[t]

            for i in range(self.n_conv_layers):
                raw = self.conv_layers[i](h)
                i_conv = self.bn_layers[i][t](raw) if self.use_bntt else self.bn_layers[i](raw)
                if v_conv[i] is None:
                    v_conv[i] = torch.zeros_like(i_conv)
                s_conv, v_conv[i] = self.lif_conv_cells[i](i_conv, v_conv[i])
                h = self.pool(s_conv) if self._pool_after[i] else s_conv

            h_flat = self.gap(h).flatten(1) if self.use_gap else h.flatten(1)
            fc_raw = self.fc1(h_flat)
            bn_out = self.bn_fc1[t](fc_raw) if self.use_bntt else self.bn_fc1(fc_raw)
            i_fc = self.dropout(bn_out)
            if v_fc is None:
                v_fc = torch.zeros_like(i_fc)
            s_fc, v_fc = self.lif_fc_cell(i_fc, v_fc)

            i_out = self.fc2(s_fc)
            if v_out is None:
                v_out = torch.zeros_like(i_out)
            s_out, v_out = self.lif_out_cell(i_out, v_out)

            mems_list.append(v_out)
            spk_list.append(s_out)

        spikes_out = torch.stack(spk_list)   # [T, B, K]
        mems_out = torch.stack(mems_list)    # [T, B, K]

        if self.twd is not None:
            return mems_out, spikes_out, self.twd(spikes_out)
        return mems_out, spikes_out

    # ── Inference helpers ─────────────────────────────────────────────────

    @staticmethod
    def first_spike_pred(spikes_out: Tensor) -> Tensor:
        """First-spike decoding: argmin over firing time.

        Args:
            spikes_out: [T, B, K]

        Returns:
            Predicted class indices [B].
        """
        T, B, K = spikes_out.shape
        dev = spikes_out.device
        t_idx = torch.arange(T, device=dev, dtype=torch.float32)[:, None, None]
        fired = torch.where(
            spikes_out > 0.5,
            t_idx.expand(T, B, K),
            torch.full_like(spikes_out, float(T)),
        )
        first, _ = fired.min(0)
        return first.argmin(1)

    @torch.no_grad()
    def predict_first_spike(self, x: Tensor) -> Tensor:
        out = self.forward(x)
        _, spikes_out = out[0], out[1]
        return self.first_spike_pred(spikes_out)

    @torch.no_grad()
    def predict_membrane(self, x: Tensor) -> Tensor:
        out = self.forward(x)
        membranes = out[0]
        return membranes.sum(0).argmax(1)
