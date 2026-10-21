"""Архивный CPU readout: log1p/counts-zscore -> Linear(36864,10), SGD25.

Здесь нет MemriBoard. Признаки уже получены из nativeMVM→архивного LIF.
Train-only scaler/head сохраняются до открытия test labels. Epoch25 фиксирован,
лучший epoch по test не выбирается. Это новый head: старый head69.65% не сохранён.
"""
from __future__ import annotations
from collections.abc import Callable
from pathlib import Path
import math
import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor
from torch.optim.sgd import SGD
from .schema import Accuracy
from .mnist_inputs import labels


def standardized(counts: NDArray[np.uint8],mean: Tensor,sd: Tensor) -> Tensor:
    # log1p позволяет сохранить нули и сжимает диапазон spike counts.
    return (torch.log1p(torch.from_numpy(np.array(counts,dtype=np.float32)))-mean)/sd


def fit(counts: NDArray[np.uint8],out: Path,progress: Callable[[str],None],
        *, label_reader: Callable[[str,int],NDArray[np.int64]]|None=None) -> torch.nn.Linear:
    n,features=counts.shape
    if n<2:raise ValueError('At least2 train images required')
    # Streaming статистики только train. log1p float32 как в архиве; суммы
    # float64 уменьшают накопление ошибки. Это явно новая численная реализация,
    # не обещание bit-identical отсутствующего исторического scaler.
    total=torch.zeros(features,dtype=torch.float64);squares=torch.zeros_like(total)
    for start in range(0,n,512):
        x=torch.log1p(torch.from_numpy(np.array(counts[start:start+512],dtype=np.float32))).double()
        total+=x.sum(0);squares+=(x*x).sum(0)
    mean=(total/n).float()
    sd=torch.sqrt(torch.clamp((squares-total*total/n)/(n-1),min=0.)).float().clamp_min(1e-6)
    torch.save(dict(mean=mean,std=sd,correction=1,train_only=True),out/'scaler.pt')
    get_labels=labels if label_reader is None else label_reader
    y=torch.from_numpy(get_labels('train',n));torch.manual_seed(42)
    model=torch.nn.Linear(features,10,bias=True)
    optimizer=SGD(model.parameters(),lr=.001,weight_decay=.0001)
    # RNG фиксирован заранее; batch shuffle даёт новый воспроизводимый head.
    history=[]
    for epoch in range(25):
        permutation=torch.randperm(n);loss_sum=0.
        for offset in range(0,n,512):
            ids=permutation[offset:offset+512]
            x=standardized(counts[ids.numpy()],mean,sd)
            optimizer.zero_grad();logits=model(x)
            loss=torch.nn.functional.cross_entropy(logits,y[ids]);torch.autograd.backward(loss);optimizer.step()
            loss_sum+=float(loss.detach())*len(ids)
        history.append(loss_sum/n);progress('CPU readout epoch %d/25 loss=%.6f'%(epoch+1,history[-1]))
    # Файл head действительно записан ДО единственного доступа к test labels.
    torch.save(dict(state_dict=model.state_dict(),epochs=25,seed=42,loss=history),out/'readout.pt')
    return model


def wilson(correct: int,n: int) -> list[float]:
    z=1.959963984540054;p=correct/n;d=1+z*z/n
    center=(p+z*z/(2*n))/d;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [center-half,center+half]


def evaluate(model: torch.nn.Linear,counts: NDArray[np.uint8],out: Path,
             *, label_reader: Callable[[str,int],NDArray[np.int64]]|None=None) -> Accuracy:
    if not (out/'readout.pt').is_file():raise ValueError('Frozen readout required before test labels')
    scaler=torch.load(out/'scaler.pt',weights_only=True,map_location='cpu')
    scores=[]
    with torch.no_grad():
        for start in range(0,len(counts),512):scores.append(model(standardized(counts[start:start+512],scaler['mean'],scaler['std'])).numpy())
    get_labels=labels if label_reader is None else label_reader
    logits=np.concatenate(scores);truth=get_labels('test',len(counts))
    prediction=np.argmax(logits,axis=1);silent=np.sum(counts,axis=1)==0
    tied=np.sum(logits==np.max(logits,axis=1,keepdims=True),axis=1)>1
    valid=~(silent|tied);correct=prediction==truth;strict=valid&correct
    confusion=np.zeros((10,10),dtype=np.int64)
    for actual,predicted in zip(truth,prediction):confusion[actual,predicted]+=1
    np.savez(out/'predictions.npz',logits=logits,labels=truth,argmax=prediction,silent=silent,tied=tied,accepted=valid,
             strict_correct=strict,counts_total=np.sum(counts,axis=1),confusion_argmax=confusion)
    # Conventional argmax соответствует привычной архивной метрике. Strict
    # сохраняет все test в знаменателе и считает silent/tied отказом, не успехом.
    return dict(test_images=len(counts),conventional_argmax_accuracy=float(np.mean(correct)),
        strict_accuracy=float(np.mean(strict)),strict_correct=int(strict.sum()),silent=int(silent.sum()),tied=int(tied.sum()),
        coverage=float(np.mean(valid)),conditional_accuracy=float(np.mean(correct[valid])) if valid.any() else None,
        wilson95_strict=wilson(int(strict.sum()),len(counts)),wilson95_argmax=wilson(int(correct.sum()),len(counts)),
        test_spikes=int(counts.sum()),test_feature_nonzero=int(np.count_nonzero(counts)),
        original_6965_reproduction_claim=False,new_readout=True,physical_hardware=False)
