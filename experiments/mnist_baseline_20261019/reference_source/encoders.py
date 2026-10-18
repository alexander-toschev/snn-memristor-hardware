# encoders_universal.py
# Universal (single + batch) device-agnostic encoders for SNN-MNIST / BindsNET pipelines.
#
# Supported input shapes:
#   - Single: [28,28] or [1,28,28]
#   - Batch : [B,28,28] or [B,1,28,28]
#
# Output shapes (configurable):
#   - out_format="auto" (default):
#       * single input  -> [T, 1, 784]  (legacy-friendly)
#       * batch input   -> [T, B, 1, 784]
#   - out_format="TBN"  -> [T, B, 784]
#   - out_format="TBN1" -> [T, B, 1, 784]
#
# Notes:
# - Encoders always operate on image.device (CPU/CUDA) with no internal device attribute.
# - PoissonEncoder supports deterministic=True without per-sample Generator/manual_seed overhead.

from __future__ import annotations

import torch
from torch import Tensor


def _normalize_mnist_shape(image: Tensor) -> tuple[Tensor, bool]:
    """Return (x, had_batch) where x is [B,28,28] and had_batch indicates whether input
    was originally a batch (B>1 or explicit batch dim).

    Accepts:
      [28,28], [1,28,28], [B,28,28], [B,1,28,28]
    """
    x = image

    # [B,1,28,28] -> [B,28,28]
    if x.ndim == 4 and x.shape[1] == 1:
        x = x.squeeze(1)

    # [1,28,28] could be single or batch of 1; treat as single unless caller provided batch explicitly.
    if x.ndim == 3 and x.shape[0] == 1:
        # ambiguous; keep it as single by default (had_batch=False)
        # convert to [1,28,28] batch form but mark as non-batch
        return x, False

    # [B,28,28]
    if x.ndim == 3:
        return x, True

    # [28,28] -> [1,28,28]
    if x.ndim == 2:
        return x.unsqueeze(0), False

    raise ValueError(f"Unsupported image shape: {tuple(image.shape)}")


def _normalize_range(x: Tensor) -> Tensor:
    # Works for float or uint8. If max>1 assume [0..255] and scale.
    if x.max() > 1:
        x = x / 255.0
    return x.clamp(0, 1)


def _format_output(spikes_tbn: Tensor, had_batch: bool, out_format: str) -> Tensor:
    """spikes_tbn is [T,B,784]."""
    out_format = out_format.lower()
    if out_format == "tbn":
        return spikes_tbn
    if out_format == "tbn1":
        return spikes_tbn.view(spikes_tbn.shape[0], spikes_tbn.shape[1], 1, spikes_tbn.shape[2])
    if out_format == "tbnchw":
        return spikes_tbn.view(spikes_tbn.shape[0], spikes_tbn.shape[1], 1, 28, 28)
    if out_format == "auto":
        if had_batch:
            return spikes_tbn.view(spikes_tbn.shape[0], spikes_tbn.shape[1], 1, spikes_tbn.shape[2])
        # legacy-friendly single: [T,1,784]
        return spikes_tbn[:, :1, :]
    raise ValueError('out_format must be one of {"auto","TBN","TBN1","TBNCHW"}')


class LatencyEncoder:
    """Latency (time-to-first-spike) encoder.

    Single + batch universal. See module header for shapes.
    """

    def __init__(self, time: int = 100, out_format: str = "auto", x_min: float = 0.0):
        self.time = int(time)
        self.out_format = out_format
        # Порог тишины: пиксели ниже x_min не генерируют импульсов (чтобы фон не создавал залп на t=T-1)
        self.x_min = float(x_min)

    @torch.no_grad()
    def __call__(self, image: Tensor) -> Tensor:
        x, had_batch = _normalize_mnist_shape(image)
        x = _normalize_range(x)

        B = x.shape[0]
        T = self.time
        dev = x.device

        x_flat = x.reshape(B, -1)  # [B,784]
        mask = x_flat >= self.x_min
        t_fire = torch.floor((1.0 - x_flat) * (T - 1)).to(torch.long)  # [B,784]

        spikes = torch.zeros((T, B, 784), dtype=torch.float32, device=dev)

        b_idx = torch.arange(B, device=dev)[:, None].expand(B, 784)
        n_idx = torch.arange(784, device=dev)[None, :].expand(B, 784)

        # Ставим импульсы только там, где пиксель “значимый” (mask=True)
        spikes[t_fire[mask], b_idx[mask], n_idx[mask]] = 1.0
        return _format_output(spikes, had_batch=had_batch, out_format=self.out_format)


class PoissonEncoder:
    """Poisson rate encoder.

    Params:
      - T: simulation time steps
      - rate_scale: lambda = x * rate_scale
      - deterministic:
          False: fastest (torch.rand)
          True : deterministic pseudo-random stream derived from image content (GPU-friendly)
      - out_format: "auto" | "TBN" | "TBN1" (see module header)
    """

    def __init__(self, T: int, rate_scale: float, base_seed: int = 123, deterministic: bool = False, out_format: str = "auto"):
        self.T = int(T)
        self.rate_scale = float(rate_scale)
        self.base_seed = int(base_seed)
        self.deterministic = bool(deterministic)
        self.out_format = out_format

    @staticmethod
    def _hash_u32(x: torch.Tensor) -> torch.Tensor:
        """Vectorized 32-bit mix/hash (GPU-friendly)."""
        x = x & 0xFFFFFFFF
        x ^= (x >> 16)
        x = (x * 0x7FEB352D) & 0xFFFFFFFF
        x ^= (x >> 15)
        x = (x * 0x846CA68B) & 0xFFFFFFFF
        x ^= (x >> 16)
        return x & 0xFFFFFFFF

    @torch.no_grad()
    def __call__(self, image: Tensor) -> Tensor:
        x, had_batch = _normalize_mnist_shape(image)
        x = _normalize_range(x)

        dev = x.device
        B = x.shape[0]
        T = self.T

        x_flat = x.reshape(B, -1)  # [B,784]
        lam = x_flat * self.rate_scale
        p = 1.0 - torch.exp(-lam)  # [B,784]

        if not self.deterministic:
            rand = torch.rand((T, B, 784), device=dev)
        else:
            # Deterministic pseudo-random stream derived from image content (no per-sample Generator)
            q = (x_flat * 255.0).to(torch.int64)  # [B,784]
            idx = torch.arange(784, device=dev, dtype=torch.int64)[None, :]  # [1,784]
            h1 = q.sum(dim=1)                      # [B]
            h2 = (q * idx).sum(dim=1)              # [B]
            seed_b = (self.base_seed
                      ^ (h1 * 1315423911)
                      ^ (h2 * 2654435761)) & 0xFFFFFFFF
            seed_b = seed_b.to(torch.int64)

            t = torch.arange(T, device=dev, dtype=torch.int64)[:, None, None]   # [T,1,1]
            n = torch.arange(784, device=dev, dtype=torch.int64)[None, None, :] # [1,1,784]
            b = seed_b[None, :, None]                                           # [1,B,1]

            xmix = (b + t * 2246822519 + n * 3266489917) & 0xFFFFFFFF
            h = self._hash_u32(xmix)  # [T,B,784]
            rand = h.to(torch.float32) / 4294967296.0

        spikes = (rand < p[None, :, :]).to(torch.float32)  # [T,B,784]
        return _format_output(spikes, had_batch=had_batch, out_format=self.out_format)
