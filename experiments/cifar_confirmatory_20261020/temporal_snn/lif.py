"""LIF (Leaky Integrate-and-Fire) neuron primitives.

LIFCell  – single-timestep update; stateful (caller manages v).
LIFLayer – processes a whole [T, B, *] tensor via time-loop.

Dynamics (hard reset):
    v[t] = tau * v[t-1] + i[t]
    s[t] = H(v[t] - v_thresh)          ← ATan surrogate in backward
    v[t] = v[t] - s[t] * (v[t] - v_reset)   ← hard reset
"""
from __future__ import annotations
import torch
import torch.nn as nn
from .surrogate import atan_spike


class LIFCell(nn.Module):
    """Single-timestep LIF update.

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

    def forward(self, i: torch.Tensor, v: torch.Tensor):
        """
        i : input current  [B, *]
        v : membrane state [B, *]  (same shape as i)
        Returns (spike [B,*], new_v [B,*])
        """
        v_new = self.tau * v + i
        spike = atan_spike(v_new - self.v_threshold, self.surrogate_alpha)
        # Hard reset: V → V_reset where spike=1, otherwise unchanged
        v_new = v_new - spike.detach() * (v_new - self.v_reset)
        return spike, v_new


from typing import Optional


class LIFLayer(nn.Module):
    """Process full temporal sequence [T, B, *] through LIF dynamics.

    Returns (spikes [T,B,*], membranes [T,B,*]).
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

    def forward(self, x: torch.Tensor):
        """x: [T, B, *]  →  (spikes [T,B,*], membranes [T,B,*])"""
        T = x.shape[0]
        v = torch.zeros_like(x[0])
        spikes, mems = [], []
        for t in range(T):
            s, v = self.cell(x[t], v)
            spikes.append(s)
            mems.append(v)
        return torch.stack(spikes), torch.stack(mems)


class RecurrentLIFCell(nn.Module):
    """LIF cell with lateral recurrent connection.

    v[t]  = tau * v[t-1] + i_ext[t] + W_rec @ s[t-1]
    s[t]  = H(v[t] - v_threshold)
    v[t]  = v[t] - s[t] * (v[t] - v_reset)   ← hard reset

    Maintains internal state: v (membrane) and s_prev (previous spike).
    Reset via reset_state() between sequences.

    Args:
        hidden:          number of units (W_rec is [hidden, hidden])
        tau:             membrane decay factor
        v_threshold:     firing threshold
        v_reset:         reset potential after spike
        surrogate_alpha: ATan surrogate steepness
        init_scale:      W_rec initialised with kaiming_normal * init_scale
                         (default 0.1 keeps initial activity from exploding)

    Usage::
        cell = RecurrentLIFCell(256, tau=0.5)
        cell.reset_state()
        for t in range(T):
            spike = cell(input_t)   # state managed internally
    """

    def __init__(
        self,
        hidden: int,
        tau: float = 0.5,
        v_threshold: float = 1.0,
        v_reset: float = 0.0,
        surrogate_alpha: float = 2.0,
        init_scale: float = 0.1,
    ):
        super().__init__()
        self.tau = tau
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.surrogate_alpha = surrogate_alpha

        self.W_rec = nn.Linear(hidden, hidden, bias=False)
        nn.init.kaiming_normal_(self.W_rec.weight, mode="fan_out", nonlinearity="relu")
        self.W_rec.weight.data.mul_(init_scale)

        # Internal state — None until first forward call or after reset_state()
        self._v:      Optional[torch.Tensor] = None
        self._s_prev: Optional[torch.Tensor] = None

    def reset_state(self):
        """Reset membrane and spike state (re-initialised lazily on next call)."""
        self._v = None
        self._s_prev = None

    def forward(self, i_ext: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        i_ext: external input current [B, hidden]
        Returns: spike tensor [B, hidden]

        State (v, s_prev) persists across calls within a sequence.
        Call reset_state() before starting a new sequence.
        """
        if self._v is None:
            self._v = torch.zeros_like(i_ext)
        if self._s_prev is None:
            self._s_prev = torch.zeros_like(i_ext)

        # Recurrent contribution: W_rec @ s[t-1]
        # s_prev is detached so gradients don't flow through time
        i_rec = self.W_rec(self._s_prev)

        v_new = self.tau * self._v + i_ext + i_rec
        spike = atan_spike(v_new - self.v_threshold, self.surrogate_alpha)
        # Hard reset
        v_new = v_new - spike.detach() * (v_new - self.v_reset)

        self._v = v_new
        self._s_prev = spike.detach()   # stop gradient through time
        return spike


class ALIFCell(nn.Module):
    """Adaptive LIF: firing threshold rises after each spike.

    Dynamics (hard reset)::

        a[t]   = rho  * a[t-1] + s[t-1]    # adaptation (prev-spike accumulator)
        v_th[t]= v_threshold + beta * a[t]  # adaptive threshold
        v[t]   = tau  * v[t-1] + i[t]      # membrane
        s[t]   = H(v[t] - v_th[t])          # ATan surrogate spike
        v[t]   = v[t] - s[t]*(v[t]-v_reset) # hard reset

    The caller manages ``v`` (membrane) and ``a`` (adaptation) state tensors.
    ``_s_prev`` (previous spike for the adaptation update) is tracked internally
    and is reset via ``reset_state()`` — call this once before each new sequence.

    Args:
        tau:              membrane decay factor
        v_threshold:      base firing threshold
        v_reset:          reset potential after spike
        surrogate_alpha:  ATan surrogate steepness
        rho:              adaptation decay  (Bellec 2020: 0.96)
        beta:             threshold coupling strength (Bellec 2020: 0.07)

    Usage::
        cell = ALIFCell(rho=0.96, beta=0.07)
        cell.reset_state()
        v = torch.zeros(B, N)
        a = torch.zeros(B, N)
        for t in range(T):
            spike, v, a, psi, v_th = cell(input_t, v, a)
    """

    def __init__(
        self,
        tau: float = 0.5,
        v_threshold: float = 1.0,
        v_reset: float = 0.0,
        surrogate_alpha: float = 2.0,
        rho: float = 0.96,
        beta: float = 0.07,
    ):
        super().__init__()
        self.tau = tau
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.surrogate_alpha = surrogate_alpha
        self.rho = rho
        self.beta = beta
        self._s_prev: Optional[torch.Tensor] = None  # s[t-1] for adaptation

    def reset_state(self):
        """Clear internal spike history.  Call once before each new sequence."""
        self._s_prev = None

    def forward(
        self, i: torch.Tensor, v: torch.Tensor, a: torch.Tensor
    ):  # type: ignore[override]
        """
        i : input current  [B, *]
        v : membrane state [B, *] at t-1
        a : adaptation state [B, *] at t-1

        Returns (spike, new_v, new_a, psi, v_th) all [B, *]::

            spike : binary spike at t
            new_v : membrane potential after update + hard reset at t
            new_a : adaptation variable at t
            psi   : ATan pseudo-derivative at t  (w.r.t. adaptive threshold)
            v_th  : adaptive threshold at t  (v_threshold + beta*a[t])
        """
        if self._s_prev is None:
            self._s_prev = torch.zeros_like(i)

        # 1. Update adaptation with PREVIOUS spike (gives threshold rise at t)
        a_new = self.rho * a + self._s_prev                        # [B, *]

        # 2. Adaptive threshold
        v_th = self.v_threshold + self.beta * a_new                # [B, *]

        # 3. Membrane update
        v_new = self.tau * v + i                                   # [B, *]

        # 4. Spike
        spike = atan_spike(v_new - v_th, self.surrogate_alpha)     # [B, *]

        # 5. Pseudo-derivative (evaluated at pre-reset membrane vs adaptive threshold)
        import math as _math
        psi = (
            (self.surrogate_alpha / 2.0)
            / (1.0 + (_math.pi / 2.0 * self.surrogate_alpha * (v_new - v_th)) ** 2)
        )                                                          # [B, *]

        # 6. Hard reset
        v_new = v_new - spike.detach() * (v_new - self.v_reset)   # [B, *]

        # 7. Store spike for next timestep's adaptation update
        self._s_prev = spike.detach()

        return spike, v_new, a_new, psi, v_th
