"""Latency (TTFS) encoder: pixel intensity → spike timing.

Bright pixel (≈1.0) → fires at t=0  (earliest)
Dark   pixel (≈0.0) → fires at t=T-1 (latest) or stays silent

This is the standard Time-To-First-Spike (TTFS) encoding scheme.
Silent pixels (below x_min) never fire, providing implicit background
suppression without additional thresholding.

Reference:
    Park et al. (2020) "T2FSNN: Deep Spiking Neural Networks with Time-to-first-
    spike Coding" — DAC 2020.
"""
from __future__ import annotations
import torch
from torch import Tensor


class LatencyEncoder:
    """Convert a normalised pixel tensor [0,1] to a latency-coded spike train.

    Args:
        T:     Number of timesteps.
        x_min: Pixels below this threshold produce no spike (background suppression).

    Input:  [C, H, W]  (single image) or [B, C, H, W] (batch)
    Output: [T, C, H, W] or [T, B, C, H, W]
    """

    def __init__(self, T: int = 16, x_min: float = 0.1):
        self.T = T
        self.x_min = x_min

    @torch.no_grad()
    def __call__(self, img: Tensor) -> Tensor:
        """
        Args:
            img: Float32 tensor in [0, 1].  Shape: [C,H,W] or [B,C,H,W].

        Returns:
            Spike tensor.  Shape: [T,C,H,W] or [T,B,C,H,W].
        """
        single = img.dim() == 3
        if single:
            img = img.unsqueeze(0)          # [1, C, H, W]

        B, C, H, W = img.shape
        T = self.T
        dev = img.device

        # Spike time: bright (1.0) → t=0, dark (0.0) → t=T-1
        t_fire = torch.floor((1.0 - img) * (T - 1)).long().clamp(0, T - 1)
        mask = img >= self.x_min

        spikes = torch.zeros(T, B, C, H, W, device=dev)
        b_idx = torch.arange(B, device=dev)[:, None, None, None].expand_as(mask)
        c_idx = torch.arange(C, device=dev)[None, :, None, None].expand_as(mask)
        h_idx = torch.arange(H, device=dev)[None, None, :, None].expand_as(mask)
        w_idx = torch.arange(W, device=dev)[None, None, None, :].expand_as(mask)
        spikes[t_fire[mask], b_idx[mask], c_idx[mask], h_idx[mask], w_idx[mask]] = 1.0

        if single:
            spikes = spikes.squeeze(1)      # [T, C, H, W]
        return spikes
