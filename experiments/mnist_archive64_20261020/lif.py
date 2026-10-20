"""CPU LIF: точная динамика архивного BindsNET без T×B×64×24×24 Monitor.

Мемристорный nativeSIM завершает MVM до этого модуля. Этот модуль НЕ вызывает
плату/симулятор и НЕ вычисляет свёртку по идеальным весам: вход — измеренный LUT.
"""
from __future__ import annotations
from collections.abc import Callable
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor
from bindsnet.network.nodes import LIFNodes
from .inputs import UInt8Array, MaskArray, batches


def make_lif(channels: int,batch: int,threshold: float) -> LIFNodes:
    # Архив: tau=150ms, dt=1ms, reset/rest=0, threshold=.38, refrac=2ms.
    # В BindsNET ток приходит в условных единицах мембранного потенциала.
    layer=LIFNodes(shape=(channels,24,24),traces=True,thresh=threshold,
                   rest=0.,reset=0.,refrac=2,tc_decay=150.)
    layer.compute_decays(dt=1.);layer.set_batch_size(batch);layer.reset_state_variables()
    return layer


def run_currents(currents: Tensor,threshold: float) -> tuple[Tensor,Tensor,Tensor]:
    """Unit-test boundary: currents[t] already includes synchronous input delay."""
    if currents.ndim!=5 or currents.shape[0]!=100 or tuple(currents.shape[-2:])!=(24,24):
        raise ValueError('Expected100×batch×channels×24×24 currents')
    lif=make_lif(currents.shape[2],currents.shape[1],threshold)
    count=torch.zeros_like(lif.v,dtype=torch.int16)
    with torch.no_grad():
        for t in range(100):
            # LIF.forward modifies x in-place during refractory masking.
            # Копия обязательна: исходный LUT/current tensor остаётся неизменным.
            lif.forward(currents[t].clone());count+=lif.s.to(torch.int16)
    return count,lif.v.clone(),lif.refrac_count.clone()


def counts_for_masks(masks: MaskArray,keys: NDArray[np.uint32],lut: NDArray[np.float32],threshold: float) -> NDArray[np.uint8]:
    batch=masks.shape[1];lif=make_lif(lut.shape[1],batch,threshold)
    count=torch.zeros_like(lif.v,dtype=torch.int16)
    with torch.no_grad():
        for t in range(100):
            # Архив Network.run синхронный: input.s предыдущего шага используется
            # на текущем. t=0 получает ноль; вход t=99 не успевает попасть в Conv1.
            if t==0:current=torch.zeros_like(lif.v)
            else:
                used=masks[t-1];index=np.searchsorted(keys,used)
                if np.any(index>=len(keys)) or not np.array_equal(keys[index],used):
                    raise ValueError('Unknown native mask; zero/ideal fallback forbidden')
                values=lut[index].transpose(0,3,1,2).copy()
                current=torch.from_numpy(values)
            # Внутри цифрового BindsNET LIF на CPU: leak exp(-1/150), маска refractory, сумма тока,
            # threshold, reset и refractory=2. Никакого нового STDP/E-I/WTA.
            lif.forward(current.clone());count+=lif.s.to(torch.int16)
    return count.numpy().astype(np.uint8)


def extract(data: UInt8Array,keys: NDArray[np.uint32],lut: NDArray[np.float32],threshold: float,
            target: NDArray[np.uint8],batch_size: int,progress: Callable[[str],None],split: str) -> dict[str,int]:
    logical=0
    for start,masks in batches(data,batch_size):
        counts=counts_for_masks(masks,keys,lut,threshold)
        target[start:start+len(counts)]=counts.reshape(len(counts),-1)
        # Эти логические MVM обслужены кешем, а не новыми Connector-вызовами.
        logical+=int(masks[:99].size*lut.shape[1])
        if start%256==0 or start+batch_size>=len(data):
            progress('CPU LIF64 %s %d/%d'%(split,min(start+batch_size,len(data)),len(data)))
    return dict(cached_logical_filter_mvm=logical,digital_split_mask_uses=0)
