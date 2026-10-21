"""Official inputs and archived deterministic encoder; no labels in mask discovery."""
from __future__ import annotations
import gzip
import hashlib
import struct
from pathlib import Path
from collections.abc import Iterator, Callable
import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor
from experiment_common import mnist_data as official
IMAGE_SOURCES=official.SOURCES
from experiment_common.poisson import PoissonEncoder

UInt8Array=NDArray[np.uint8]
MaskArray=NDArray[np.uint32]


def images(split: str, count: int) -> UInt8Array:
    name=('train' if split=='train' else 't10k')+'-images-idx3-ubyte.gz'
    official.checked(name)
    with gzip.open(official.DATA/name,'rb') as stream:
        magic,n,h,w=struct.unpack('>IIII',stream.read(16))
        if magic!=2051 or (h,w)!=(28,28) or not 0<count<=n:raise ValueError('Invalid image prefix')
        raw=stream.read(count*784)
    if len(raw)!=count*784:raise ValueError('Truncated images')
    return np.frombuffer(raw,dtype=np.uint8).copy().reshape(count,1,28,28)


def labels(split: str,count: int) -> NDArray[np.int64]:
    name=('train' if split=='train' else 't10k')+'-labels-idx1-ubyte.gz'
    official.checked(name)
    with gzip.open(official.DATA/name,'rb') as stream:
        magic,n=struct.unpack('>II',stream.read(8));raw=stream.read(count)
    if magic!=2049 or count>n or len(raw)!=count or any(y>9 for y in raw):raise ValueError('Invalid labels')
    return np.frombuffer(raw,dtype=np.uint8).astype(np.int64)


def encode(batch: UInt8Array) -> Tensor:
    # CPU: архивный Poisson, вероятность pixel/255 * .0015 за один tick.
    encoder=PoissonEncoder(T=100,rate_scale=.0015,base_seed=123,deterministic=True,out_format='TBNCHW')
    return encoder(torch.from_numpy(batch.astype(np.float32)/255.))


def patch_masks(spikes: Tensor) -> MaskArray:
    a=spikes.detach().cpu().numpy().astype(np.uint8)
    if a.ndim!=5 or a.shape[0]!=100 or a.shape[2:]!=(1,28,28):raise ValueError('Archived spike shape')
    # CPU: бит bl соответствует позиции (bl//5,bl%5) внутри 5x5 patch.
    # Это извлечение бинарных входов, не умножение на веса.
    masks=np.zeros(a.shape[:2]+(24,24),dtype=np.uint32)
    for dy in range(5):
        for dx in range(5):
            masks |= a[:,:,0,dy:dy+24,dx:dx+24].astype(np.uint32)<<np.uint32(5*dy+dx)
    return masks


def batches(data: UInt8Array,batch_size: int) -> Iterator[tuple[int,MaskArray]]:
    for start in range(0,len(data),batch_size):
        yield start,patch_masks(encode(data[start:start+batch_size]))


def inventory(datasets: dict[str,UInt8Array],batch_size: int,progress: Callable[[str],None]) -> tuple[list[int],dict[str,str]]:
    # CPU: только изображения и content-seeded Poisson, никаких labels/ответов модели.
    seen={0};hashes={}
    for split,data in datasets.items():
        h=hashlib.sha256()
        for start,masks in batches(data,batch_size):
            seen.update(map(int,np.unique(masks)));h.update(masks.tobytes())
            if start%256==0 or start+batch_size>=len(data):progress('Mask inventory %s %d/%d'%(split,min(start+batch_size,len(data)),len(data)))
        hashes[split]=h.hexdigest()
    return sorted(seen),hashes
