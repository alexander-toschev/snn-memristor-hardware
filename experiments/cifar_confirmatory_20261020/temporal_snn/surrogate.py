"""Surrogate gradient functions for differentiating through spike events.

The spike function H(v - threshold) is non-differentiable (step function).
During backprop we substitute a smooth approximation.

Supported surrogates:
  - ATan  : d/dx [arctan(alpha * pi/2 * x) / pi + 0.5]
  - Sigmoid: derivative of scaled sigmoid
"""
from __future__ import annotations
import math
import torch


class _ATanSurrogate(torch.autograd.Function):
    """Forward: Heaviside (spike).  Backward: ATan surrogate."""
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return (x >= 0.0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, = ctx.saved_tensors
        alpha = ctx.alpha
        # Derivative of arctan(alpha * pi/2 * x) / pi is:
        #   alpha/2 * 1/(1 + (alpha*pi/2 * x)^2)
        grad = (alpha / 2.0) / (1.0 + (math.pi / 2.0 * alpha * x) ** 2) * grad_output
        return grad, None


class _SigmoidSurrogate(torch.autograd.Function):
    """Forward: Heaviside.  Backward: scaled sigmoid derivative."""
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return (x >= 0.0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, = ctx.saved_tensors
        alpha = ctx.alpha
        sg = torch.sigmoid(alpha * x)
        grad = alpha * sg * (1.0 - sg) * grad_output
        return grad, None


def atan_spike(x: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
    """Spike with ATan surrogate gradient.  x should be (v - threshold)."""
    return _ATanSurrogate.apply(x, alpha)


def sigmoid_spike(x: torch.Tensor, alpha: float = 4.0) -> torch.Tensor:
    """Spike with sigmoid surrogate gradient."""
    return _SigmoidSurrogate.apply(x, alpha)
