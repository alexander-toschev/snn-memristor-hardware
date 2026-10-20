"""Offline readout verification only; never opens Connector or trains a model.

Only a terminal completed archive64 session with 10000 train/10000 test is
accepted. Test labels are read after this check and saved-head verification.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import cast

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor

ROOT=Path(__file__).resolve().parents[1]
LABEL_SHA='f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6'


def require(condition: bool,message: str) -> None:
    if not condition:raise ValueError(message)


def digest(path: Path) -> str:
    result=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):result.update(block)
    return result.hexdigest()


def mapping(value: object) -> dict[str,object]:
    require(isinstance(value,dict),'Expected dictionary')
    if not isinstance(value,dict):raise ValueError('Expected dictionary')
    require(all(isinstance(k,str) for k in value),'Non-string dictionary key')
    return {str(k):v for k,v in value.items()}


def read_json(path: Path) -> dict[str,object]:
    value: object=json.loads(path.read_text(encoding='utf-8'))
    return mapping(value)


def tensor(value: object,shape: tuple[int,...]) -> Tensor:
    require(isinstance(value,Tensor),'Expected tensor')
    if not isinstance(value,Tensor):raise ValueError('Expected tensor')
    require(tuple(value.shape)==shape and value.dtype==torch.float32,'Wrong tensor shape/dtype')
    require(bool(torch.isfinite(value).all()),'Nonfinite tensor')
    return value


def counts(path: Path) -> NDArray[np.uint8]:
    value: object=np.load(path,mmap_mode='r',allow_pickle=False)
    require(isinstance(value,np.ndarray),'Expected counts array')
    if not isinstance(value,np.ndarray):raise ValueError('Expected counts array')
    require(value.dtype==np.uint8 and value.shape==(10000,36864),'Wrong count shape/dtype')
    return cast(NDArray[np.uint8],value)


def wilson(correct: int,n: int) -> list[float]:
    z=1.959963984540054;p=correct/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [center-half,center+half]


def verify(session: Path) -> dict[str,object]:
    # Нельзя читать test labels или результаты незавершённой сессии.
    manifest=read_json(session/'manifest.json')
    require(manifest.get('status')=='completed','Session is not completed')
    require(manifest.get('source')=='simulator' and manifest.get('physical_hardware') is False,'Not native SIM scope')
    require(manifest.get('train_images')==10000 and manifest.get('test_images')==10000,'Expected10000/10000')
    files=('train_counts.npy','test_counts.npy','scaler.pt','readout.pt','predictions.npz','accuracy.json')
    hashes=mapping(manifest['result_sha256'])
    actual={name:digest(session/name) for name in files}
    require(all(hashes.get(name)==sha for name,sha in actual.items()),'Result SHA mismatch')
    train=counts(session/'train_counts.npy');test=counts(session/'test_counts.npy')
    scaler=mapping(torch.load(session/'scaler.pt',weights_only=True,map_location='cpu'))
    head=mapping(torch.load(session/'readout.pt',weights_only=True,map_location='cpu'))
    state=mapping(head['state_dict'])
    weight=tensor(state['weight'],(10,36864));bias=tensor(state['bias'],(10,))
    mean=tensor(scaler['mean'],(36864,));sd=tensor(scaler['std'],(36864,))
    require(bool((sd>=1e-6).all()) and scaler.get('train_only') is True and scaler.get('correction')==1,'Wrong scaler contract')
    require(head.get('epochs')==25 and head.get('seed')==42,'Unexpected fixed readout protocol')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    # Статистика строго из train; здесь нет подбора SD floor/маски/head.
    total=torch.zeros(36864,dtype=torch.float64);squares=torch.zeros_like(total)
    active=np.zeros(36864,dtype=np.bool_)
    for start in range(0,10000,512):
        batch=np.array(train[start:start+512],dtype=np.float32)
        require(bool((batch<=100).all()),'Invalid train spike count')
        active|=np.any(batch>0,axis=0)
        x=torch.log1p(torch.from_numpy(batch)).double()
        total+=x.sum(0);squares+=(x*x).sum(0)
    variance=torch.clamp((squares-total*total/10000)/9999,min=0.)
    require(torch.equal(mean,(total/10000).float()),'Train mean mismatch')
    require(torch.equal(sd,variance.sqrt().float().clamp_min(1e-6)),'Train std mismatch')
    model=torch.nn.Linear(36864,10,bias=True)
    model.load_state_dict({'weight':weight,'bias':bias});model.eval()
    fresh=[];test_active=np.zeros(36864,dtype=np.bool_)
    unseen_image_count=0;unseen_spikes=0;test_nonzero=0;max_abs_z=0.
    for start in range(0,10000,512):
        batch=np.array(test[start:start+512],dtype=np.float32)
        require(bool((batch<=100).all()),'Invalid test spike count')
        test_active|=np.any(batch>0,axis=0)
        unseen_image_count+=int(np.any(batch[:,~active]>0,axis=1).sum())
        unseen_spikes+=int(batch[:,~active].sum());test_nonzero+=int(np.count_nonzero(batch))
        x=(torch.log1p(torch.from_numpy(batch))-mean)/sd
        max_abs_z=max(max_abs_z,float(x.abs().max()))
        with torch.no_grad():fresh.append(model(x).numpy())
    logits=np.concatenate(fresh)
    require(bool(np.isfinite(logits).all()),'Nonfinite logits')
    # Только теперь открывается официальный test IDX; порядок0..9999 фиксирован.
    label_file=ROOT/'experiments/mnist_baseline_20261019/data/MNIST/raw/t10k-labels-idx1-ubyte.gz'
    require(digest(label_file)==LABEL_SHA,'Official labels SHA mismatch')
    with gzip.open(label_file,'rb') as stream:header=stream.read(8);raw=stream.read()
    require(struct.unpack('>II',header)==(2049,10000) and len(raw)==10000,'Official test labels shape')
    truth=np.frombuffer(raw,dtype=np.uint8).astype(np.int64)
    require(bool((truth<=9).all()),'Invalid class')
    pred=logits.argmax(axis=1);totals=test.sum(axis=1);silent=totals==0
    tied=(logits==logits.max(axis=1,keepdims=True)).sum(axis=1)>1
    accepted=~(silent|tied);correct=pred==truth;strict=accepted&correct
    confusion=np.zeros((10,10),dtype=np.int64)
    np.add.at(confusion,(truth,pred),1)
    with np.load(session/'predictions.npz',allow_pickle=False) as saved:
        for name,value in dict(logits=logits,labels=truth,argmax=pred,silent=silent,tied=tied,accepted=accepted,
                               strict_correct=strict,counts_total=totals,confusion_argmax=confusion).items():
            require(np.array_equal(saved[name],value),'Saved prediction mismatch: '+name)
    metrics: dict[str,object]=dict(test_images=10000,conventional_argmax_accuracy=float(correct.mean()),
        strict_accuracy=float(strict.mean()),strict_correct=int(strict.sum()),silent=int(silent.sum()),tied=int(tied.sum()),
        coverage=float(accepted.mean()),conditional_accuracy=float(correct[accepted].mean()) if accepted.any() else None,
        wilson95_strict=wilson(int(strict.sum()),10000),wilson95_argmax=wilson(int(correct.sum()),10000),
        test_spikes=int(totals.sum()),test_feature_nonzero=test_nonzero)
    accuracy=read_json(session/'accuracy.json');manifest_accuracy=mapping(manifest['accuracy'])
    require(all(accuracy.get(k)==v and manifest_accuracy.get(k)==v for k,v in metrics.items()),'Accuracy mismatch')
    return dict(status='PASS',scope='offline saved-head/scaler/counts verification; not native raw replay or training replay',
        session_id=manifest['session_id'],manifest_sha256=digest(session/'manifest.json'),result_sha256=actual,
        verifier_sha256=digest(Path(__file__)),metrics=metrics,
        diagnostics=dict(train_zero_variance_features=int((variance==0).sum()),train_never_active_features=int((~active).sum()),
            train_sd_floor_features=int((sd==1e-6).sum()),test_active_features=int(test_active.sum()),
            unseen_train_features_active_test=int((~active&test_active).sum()),test_images_with_unseen_features=unseen_image_count,
            test_spikes_on_unseen_features=unseen_spikes,max_abs_standardized_test=max_abs_z,max_abs_logit=float(np.abs(logits).max())),
        parameters_tuned=False,physical_hardware=False)


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();session=(ROOT/args.session).resolve();output=(ROOT/args.output).resolve()
    require(session.is_relative_to(ROOT) and output.is_relative_to(ROOT),'Use repository-local paths')
    require(not output.is_relative_to(session),'Do not modify acquired session')
    report=verify(session)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8',newline='\n') as stream:stream.write(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'],metrics=report['metrics']),ensure_ascii=False))
    return 0


if __name__=='__main__':raise SystemExit(main())
