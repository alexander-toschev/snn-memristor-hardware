"""Совместимый путь общего typed-модуля; алгоритмы в experiment_common."""
from experiment_common.mnist_inputs import (
    UInt8Array as UInt8Array,
    MaskArray as MaskArray,
    images as images,
    labels as labels,
    encode as encode,
    patch_masks as patch_masks,
    batches as batches,
    inventory as inventory,
    IMAGE_SOURCES as IMAGE_SOURCES,
)
from experiment_common.poisson import PoissonEncoder as PoissonEncoder
