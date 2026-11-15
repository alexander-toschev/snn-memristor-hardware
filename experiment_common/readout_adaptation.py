"""CPU-only train/validation selection and refit over frozen measured features."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor
from torch.optim.sgd import SGD

from .durable import atomic_json, sha
from .readout import standardized
from .selective_metrology import object_dict

Counts = NDArray[np.uint8]
Ids = NDArray[np.int64]
Progress = Callable[[str], None]


@dataclass(frozen=True)
class Scaler:
    mean: Tensor
    sd: Tensor


def stratified_split(truth: Ids, validation_count: int, seed: int = 42) -> tuple[Ids, Ids]:
    """Exact validation size via largest remainders; no test data accepted here."""
    if truth.ndim != 1 or not 0 < validation_count < len(truth):
        raise ValueError('Invalid train/validation sizes')
    classes, sizes = np.unique(truth, return_counts=True)
    exact = sizes.astype(np.float64) * validation_count / len(truth)
    quotas = np.floor(exact).astype(np.int64)
    order = sorted(range(len(classes)), key=lambda i: (-(exact[i]-quotas[i]), int(classes[i])))
    for i in order[:validation_count-int(quotas.sum())]:
        quotas[i] += 1
    if np.any(quotas <= 0) or np.any(quotas >= sizes):
        raise ValueError('Each class must occur in both train partitions')
    rng = np.random.default_rng(seed)
    fit_parts: list[Ids] = []
    valid_parts: list[Ids] = []
    for label, quota in zip(classes, quotas):
        ids = rng.permutation(np.flatnonzero(truth == label)).astype(np.int64)
        valid_parts.append(ids[:int(quota)])
        fit_parts.append(ids[int(quota):])
    fit = rng.permutation(np.concatenate(fit_parts)).astype(np.int64)
    valid = rng.permutation(np.concatenate(valid_parts)).astype(np.int64)
    if len(valid) != validation_count or len(np.intersect1d(fit, valid)):
        raise ValueError('Train partitions overlap or have incorrect size')
    return fit, valid


def fit_scaler(counts: Counts, ids: Ids) -> Scaler:
    if len(ids) < 2 or len(np.unique(ids)) != len(ids):
        raise ValueError('At least two unique fitting examples required')
    total = torch.zeros(counts.shape[1], dtype=torch.float64)
    squares = torch.zeros_like(total)
    for start in range(0, len(ids), 512):
        x = torch.log1p(torch.from_numpy(np.array(counts[ids[start:start+512]], dtype=np.float32))).double()
        total += x.sum(0)
        squares += (x*x).sum(0)
    n = len(ids)
    return Scaler((total/n).float(), torch.sqrt(torch.clamp((squares-total*total/n)/(n-1),min=0.)).float().clamp_min(1e-6))


def train_epoch(model: torch.nn.Linear, optimizer: SGD, counts: Counts,
                truth: Ids, ids: Ids, scaler: Scaler) -> float:
    order = ids[torch.randperm(len(ids)).numpy()]
    loss_sum = 0.
    model.train()
    for start in range(0, len(order), 512):
        batch = order[start:start+512]
        x = standardized(counts[batch], scaler.mean, scaler.sd)
        target = torch.from_numpy(truth[batch])
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x), target)
        torch.autograd.backward(loss)
        optimizer.step()
        loss_sum += float(loss.detach()) * len(batch)
    return loss_sum / len(ids)


def validation_accuracy(model: torch.nn.Linear, counts: Counts, truth: Ids,
                        ids: Ids, scaler: Scaler) -> tuple[int, int]:
    """Strict correct count: silent or tied responses remain errors."""
    strict = 0
    conventional = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(ids), 512):
            batch = ids[start:start+512]
            subset = counts[batch]
            scores = model(standardized(subset,scaler.mean,scaler.sd)).numpy()
            correct = np.argmax(scores,axis=1) == truth[batch]
            tied = np.sum(scores == np.max(scores,axis=1,keepdims=True),axis=1) > 1
            silent = np.sum(subset,axis=1) == 0
            strict += int(np.count_nonzero(correct & ~tied & ~silent))
            conventional += int(np.count_nonzero(correct))
    return strict, conventional


def earliest_best(correct_counts: list[int]) -> int:
    if not correct_counts:
        raise ValueError('Validation history is empty')
    return correct_counts.index(max(correct_counts)) + 1


def adapt(counts: Counts, truth: Ids, out: Path, progress: Progress,
          *, validation_count: int = 2000, max_epochs: int = 25) -> tuple[torch.nn.Linear, int]:
    """Choose epochs on train-only validation, then discard model/scaler and refit."""
    if len(counts) != len(truth) or not 1 <= max_epochs <= 25:
        raise ValueError('Invalid count/label length or epoch budget')
    fit, valid = stratified_split(truth, validation_count)
    np.savez(out/'split.npz',fit=fit,validation=valid)
    scaler = fit_scaler(counts,fit)
    torch.save(dict(mean=scaler.mean,std=scaler.sd,fit_ids=fit.tolist(),train_only=True,correction=1),out/'selection_scaler.pt')
    torch.manual_seed(42)
    model = torch.nn.Linear(counts.shape[1],10,bias=True)
    optimizer = SGD(model.parameters(),lr=.001,weight_decay=.0001)
    history: list[dict[str, object]] = []
    correct_counts: list[int] = []
    for epoch in range(1,max_epochs+1):
        loss = train_epoch(model,optimizer,counts,truth,fit,scaler)
        strict, argmax = validation_accuracy(model,counts,truth,valid,scaler)
        correct_counts.append(strict)
        history.append(dict(epoch=epoch,fit_loss=loss,validation_strict_correct=strict,
            validation_strict_accuracy=strict/len(valid),validation_argmax_accuracy=argmax/len(valid)))
        atomic_json(out/'selection_history.json',dict(history=history,selected_epoch_so_far=earliest_best(correct_counts)))
        progress(f'Validation epoch {epoch}/{max_epochs}: strict {strict}/{len(valid)}')
    selected = earliest_best(correct_counts)
    atomic_json(out/'selection.json',dict(selected_epoch=selected,criterion='highest validation STRICT; earliest tie',
        seed=42,fit_images=len(fit),validation_images=len(valid),test_used=False))
    # Fresh scaler and head; validation labels now become legitimate refit data.
    all_ids = np.arange(len(counts),dtype=np.int64)
    final_scaler = fit_scaler(counts,all_ids)
    torch.save(dict(mean=final_scaler.mean,std=final_scaler.sd,train_only=True,correction=1,
                    train_images=len(counts)),out/'scaler.pt')
    torch.manual_seed(42)
    final_model = torch.nn.Linear(counts.shape[1],10,bias=True)
    final_optimizer = SGD(final_model.parameters(),lr=.001,weight_decay=.0001)
    losses: list[float] = []
    for epoch in range(1,selected+1):
        losses.append(train_epoch(final_model,final_optimizer,counts,truth,all_ids,final_scaler))
        progress(f'Refit epoch {epoch}/{selected}: loss {losses[-1]:.6f}')
    final_model.eval()
    torch.save(dict(state_dict=final_model.state_dict(),epochs=selected,seed=42,loss=losses,
                    train_images=len(counts),selection='train validation only'),out/'readout.pt')
    # This immutable checkpoint seal is established before the test reader runs.
    atomic_json(out/'checkpoint_before_test.json',dict(test_labels_accessed=False,selected_epoch=selected,
        sha256={name:sha(out/name) for name in ('readout.pt','scaler.pt','selection.json','split.npz')}))
    return final_model,selected


def verify_checkpoint(out: Path) -> None:
    seal = object_dict(json.loads((out/'checkpoint_before_test.json').read_text(encoding='utf-8')))
    hashes = object_dict(seal['sha256'])
    for name in ('readout.pt','scaler.pt','selection.json','split.npz'):
        if sha(out/name) != hashes[name]:
            raise ValueError('Checkpoint changed before test label access: '+name)


def verified_file(folder: Path, name: str, manifest: dict[str, object]) -> Path:
    path = (folder/name).resolve()
    if not path.is_relative_to(folder.resolve()) or path.is_symlink():
        raise ValueError('Evidence path escapes session')
    if sha(path) != object_dict(manifest['result_sha256'])[name]:
        raise ValueError('Evidence SHA mismatch: '+name)
    return path
