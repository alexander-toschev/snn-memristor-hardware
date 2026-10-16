"""LIF (Leaky Integrate-and-Fire) neuron primitives.

LIFCell  – single-timestep update; stateful (caller manages v).
LIFLayer – processes a whole [T, B, *] tensor via time-loop.

Dynamics (hard reset):
    v[t] = tau * v[t-1] + i[t]
    s[t] = H(v[t] - v_thresh)          ← ATan surrogate in backward
    v[t] = v[t] - s[t] * (v[t] - v_reset)   ← hard reset

References:
    Fang et al. (2021) "Incorporating Learnable Membrane Time Constant to
    Enhance Learning of Spiking Neural Networks"
    Neftci et al. (2019) "Surrogate Gradient Learning in Spiking Neural Networks"
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
from torch import Tensor


# ─────────────────────────────────────────────────────────────────────────────
# ATan surrogate gradient
# ─────────────────────────────────────────────────────────────────────────────

class _ATanSurrogate(torch.autograd.Function):
    """Heaviside forward + ATan pseudo-derivative backward."""

    @staticmethod
    def forward(ctx, x: Tensor, alpha: float) -> Tensor:
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return (x >= 0).float()

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha
        grad = alpha / 2.0 / (1.0 + (math.pi / 2.0 * alpha * x) ** 2) * grad_output
        return grad, None


def atan_spike(x: Tensor, alpha: float = 2.0) -> Tensor:
    """Heaviside with ATan surrogate gradient."""
    return _ATanSurrogate.apply(x, alpha)


# ─────────────────────────────────────────────────────────────────────────────
# LIF cells
# ─────────────────────────────────────────────────────────────────────────────

class LIFCell(nn.Module):
    """Single-timestep LIF update.

    Args:
        tau:              Membrane decay factor ∈ (0, 1).
        v_threshold:      Firing threshold.
        v_reset:          Post-spike reset potential.
        surrogate_alpha:  ATan surrogate steepness (higher = steeper).

    Usage::
        cell = LIFCell(tau=0.5, v_threshold=1.0)
        v = torch.zeros(B, C, H, W, device=device)
        for t in range(T):
            spike, v = cell(input_t, v)
    """

    def __init__(
        self,
        tau: float = 0.5,
        v_threshold: float = 1.0,
        v_reset: float = 0.0,
        surrogate_alpha: float = 2.0,
    ):
        super().__init__()
        self.tau = tau
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.surrogate_alpha = surrogate_alpha

    def forward(self, i: Tensor, v: Tensor):
        """
        Args:
            i: Input current  [B, *]
            v: Membrane state [B, *] at t-1

        Returns:
            (spike [B,*], new_v [B,*])
        """
        v_new = self.tau * v + i
        spike = atan_spike(v_new - self.v_threshold, self.surrogate_alpha)
        v_new = v_new - spike.detach() * (v_new - self.v_reset)
        return spike, v_new


class LIFLayer(nn.Module):
    """Process full temporal sequence [T, B, *] through LIF dynamics.

    Returns:
        (spikes [T, B, *], membranes [T, B, *])
    """

    def __init__(
        self,
        tau: float = 0.5,
        v_threshold: float = 1.0,
        v_reset: float = 0.0,
        surrogate_alpha: float = 2.0,
    ):
        super().__init__()
        self.cell = LIFCell(tau, v_threshold, v_reset, surrogate_alpha)

    def forward(self, x: Tensor):
        """x: [T, B, *]  →  (spikes [T,B,*], membranes [T,B,*])"""
        T = x.shape[0]
        v = torch.zeros_like(x[0])
        spikes, mems = [], []
        for t in range(T):
            s, v = self.cell(x[t], v)
            spikes.append(s)
            mems.append(v)
        return torch.stack(spikes), torch.stack(mems)
