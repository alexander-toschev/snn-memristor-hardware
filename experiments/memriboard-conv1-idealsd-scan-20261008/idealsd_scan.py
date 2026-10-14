#!/usr/bin/env python3
"""Offline ideal_sd scan for all 64 Conv1 filters (no MemriBoard, no MVM).

Reproduces the exact patch generation from the frozen v2 pilot and computes the
idealized (clean PyTorch) first-conv outputs, then per-filter signal-variance
statistics used to stratify a representative MVM sample and to design a
variance-aware validation gate.
"""
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/mnt/d/projects/snn-mnist')
CKPT = ROOT / 'runs_csnn/confirmatory_wide_bntt_20261006/twd1_sigma0p0_seed0/best_fs.pt'
CKPT_SHA = 'ee4fc4f58a3d010218116e9ad4c7ba808787959f9ed7687cd58cbb32d90f62fb'
SPLIT_DIGEST = 'f3a5acaf7d61d1221b28a6ba66e468f905a44463630a8bdc5cf38cbb956f0492'
OUT = ROOT / '.tmp/openclaw-spikes/memriboard-conv1-idealsd-scan-20261008'

T = 12
X_MIN = 0.1
FINE = [0, 1, 2, 3, 5, 8, 13, 14, 17, 19]
SPLIT_SEED = 20261006
POSITIONS = [0, 1, 2, 3, 4, 5, 6, 7]
TOPS = [[0, 0], [7, 7], [14, 14], [21, 21]]
SPIKE_TIMES = [0, 3, 6, 9]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def patches():
    with open(ROOT / 'data/cifar-100-python/train', 'rb') as f:
        raw = pickle.load(f, encoding='bytes')
    labels = np.asarray(raw[b'fine_labels'])
    selected = np.flatnonzero(np.isin(labels, FINE))
    filtered = labels[selected]
    rng = np.random.default_rng(SPLIT_SEED)
    val = []
    for cls in np.unique(filtered):
        pool = np.flatnonzero(filtered == cls)
        val.extend(rng.permutation(pool)[: round(len(pool) * 0.1)].tolist())
    val = sorted(val)
    mask = np.ones(len(selected), bool)
    mask[val] = False
    train = np.flatnonzero(mask).tolist()
    digest = hashlib.sha256(json.dumps({'train_indices': train, 'val_indices': val}, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert digest == SPLIT_DIGEST, f'split mismatch {digest}'
    rows = []
    for split, indices in [('train', train), ('val', val)]:
        for pos in POSITIONS:
            fi = indices[pos]
            oi = int(selected[fi])
            img = raw[b'data'][oi].reshape(3, 32, 32).astype(np.float32) / 255.0
            for top, left in TOPS:
                px = img[:, top:top + 5, left:left + 5].reshape(75)
                for t in SPIKE_TIMES:
                    fire = np.floor((1 - px) * (T - 1)).astype(int).clip(0, T - 1)
                    x = ((px >= X_MIN) & (fire == t)).astype(float)
                    rows.append((split, x))
    assert len(rows) == 256
    return rows


def main():
    assert sha(CKPT) == CKPT_SHA, 'checkpoint hash mismatch'
    w = torch.load(CKPT, map_location='cpu', weights_only=True)['conv_layers.0.weight']
    assert w.shape == (64, 3, 5, 5)
    W = w.detach().numpy().reshape(64, 75).astype(float)
    rows = patches()
    Xtr = np.stack([x for s, x in rows if s == 'train'])   # (128, 75)
    Xva = np.stack([x for s, x in rows if s == 'val'])     # (128, 75)
    per_filter = []
    for f in range(64):
        ytr = Xtr @ W[f]
        yva = Xva @ W[f]
        yall = np.concatenate([ytr, yva])
        per_filter.append({
            'filter': f,
            'ideal_sd_train': float(np.std(ytr)),
            'ideal_sd_val': float(np.std(yva)),
            'ideal_sd_all': float(np.std(yall)),
            'mean_abs_val': float(np.mean(np.abs(yva))),
            'max_abs_val': float(np.max(np.abs(yva))),
            'nonzero_frac_val': float(np.mean(np.abs(yva) > 1e-12)),
        })
    sds = np.array([d['ideal_sd_val'] for d in per_filter])
    order = np.argsort(sds)
    quart = {q: float(np.quantile(sds, q)) for q in (0.0, 0.25, 0.5, 0.75, 1.0)}
    result = {
        'checkpoint_sha256': sha(CKPT),
        'split_digest': SPLIT_DIGEST,
        'n_train_patches': int(len(Xtr)),
        'n_val_patches': int(len(Xva)),
        'ideal_sd_val_quartiles': quart,
        'ideal_sd_val_minmax': [float(sds.min()), float(sds.max())],
        'filters_sorted_by_ideal_sd_val': [
            {'filter': int(per_filter[i]['filter']), 'ideal_sd_val': per_filter[i]['ideal_sd_val'],
             'ideal_sd_all': per_filter[i]['ideal_sd_all'], 'mean_abs_val': per_filter[i]['mean_abs_val'],
             'max_abs_val': per_filter[i]['max_abs_val']}
            for i in order
        ],
        'per_filter': per_filter,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'idealsd_scan.json').write_text(json.dumps(result, indent=2))
    print('CHECKPOINT_SHA', result['checkpoint_sha256'])
    print('SPLIT_DIGEST', result['split_digest'])
    print('VAL_SD quartiles', quart)
    print('VAL_SD min/max', result['ideal_sd_val_minmax'])
    print('\nfilters sorted by val ideal_sd (ascending):')
    for d in result['filters_sorted_by_ideal_sd_val']:
        print(f"  f={d['filter']:2d}  sd_val={d['ideal_sd_val']:.4f}  sd_all={d['ideal_sd_all']:.4f}  mean_abs={d['mean_abs_val']:.4f}  max_abs={d['max_abs_val']:.4f}")
    print('\nOUTPUT', OUT / 'idealsd_scan.json')


if __name__ == '__main__':
    main()
