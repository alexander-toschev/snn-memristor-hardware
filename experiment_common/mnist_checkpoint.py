"""Загрузка конкретного архивного MNIST64 checkpoint, не fresh STDP init."""
from __future__ import annotations
import hashlib
from pathlib import Path
from typing import TypedDict
import torch
from torch import Tensor
from .durable import sha

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT/'weights/mnist_baseline_20261019/model.pt'
CHECKPOINT_SHA = '43f7d6237b0ec8405956a4f9ef34c2ea0c8a3a592c4c76467e1acc00beee47de'
WEIGHT_SHA = '89cbe275beac267f0842fc83ff5abb805b7c85ae2153a4495c80f0933041dad4'


class Checkpoint(TypedDict):
    W: Tensor
    v_thresh: Tensor
    cfg: dict[str, object]


def load_checkpoint() -> Checkpoint:
    if sha(CHECKPOINT) != CHECKPOINT_SHA:
        raise ValueError('Checkpoint SHA mismatch')
    data: object = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    if not isinstance(data, dict) or not isinstance(data.get('cfg'), dict):
        raise ValueError('Malformed checkpoint')
    weights: object = data['W']
    threshold: object = data['v_thresh']
    if not isinstance(weights, Tensor) or not isinstance(threshold, Tensor):
        raise ValueError('Checkpoint tensors required')
    digest = hashlib.sha256(weights.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    if weights.dtype != torch.float32 or tuple(weights.shape) != (64,1,5,5) or not torch.isfinite(weights).all() or digest != WEIGHT_SHA:
        raise ValueError('Invalid frozen weights')
    if threshold.dtype != torch.float32 or threshold.ndim != 0 or not torch.isfinite(threshold):
        raise ValueError('Invalid threshold')
    expected: dict[str, object] = {
        'time':100, 'seed':42, 'encoder':'poisson', 'poisson_rate_scale':.0015,
        'poisson_base_seed':123, 'poisson_deterministic':True, 'encoder_rate_boost':1.,
        'c1_out':64, 'c1_kernel':5, 'c1_stride':1, 'c1_pad':0,
        'w_norm_enable':True, 'ei_enable':True, 'wta_enable':False,
        'local_inhib_enable':False, 'adapt_thresh_enable':False,
    }
    cfg: dict[str, object] = {}
    for key, value in data['cfg'].items():
        if not isinstance(key, str):
            raise ValueError('Checkpoint config keys must be strings')
        cfg[key] = value
    if any(cfg.get(key) != value for key, value in expected.items()):
        raise ValueError('Unexpected archived configuration')
    return dict(W=weights, v_thresh=threshold, cfg=cfg)
