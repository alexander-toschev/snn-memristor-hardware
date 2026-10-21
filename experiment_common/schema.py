"""Точные структуры результатов и узкая граница старого NumPy memmap API."""
from __future__ import annotations
from pathlib import Path
from typing import Protocol,TypedDict
import numpy as np

CountMap=np.memmap[tuple[int,...],np.dtype[np.uint8]]
class MemmapFactory(Protocol):
    # numpy2.0 open_memmap существует, но не имеет аннотаций. Это описание
    # единственной реально используемой сигнатуры, не новое поведение.
    def __call__(self,filename: Path,*,mode: str,dtype: type[np.uint8],shape: tuple[int,int]) -> CountMap: ...

class Accuracy(TypedDict):
    test_images: int
    conventional_argmax_accuracy: float
    strict_accuracy: float
    strict_correct: int
    silent: int
    tied: int
    coverage: float
    conditional_accuracy: float|None
    wilson95_strict: list[float]
    wilson95_argmax: list[float]
    test_spikes: int
    test_feature_nonzero: int
    original_6965_reproduction_claim: bool
    new_readout: bool
    physical_hardware: bool

class RunResult(TypedDict):
    session: str
    status: str
    error: str|None
    accuracy: Accuracy|None

class Manifest(TypedDict,total=False):
    session_id: str
    source: str
    scope: str
    status: str
    started_utc: str
    finished_utc: str
    protocol: dict[str,object]
    source_sha256: dict[str,str]
    invocation: dict[str,object]
    checkpoint_sha256: str
    characterization: dict[str,object]
    seed: int
    batch_size: int
    train_images: int
    test_images: int
    physical_hardware: bool
    new_stdp_training: bool
    packages: dict[str,str]
    encoding_sha256: dict[str,str]
    unique_masks: int
    native_masks: int
    max_popcount: int
    native_lut_call_cap: int
    maximum_impacts: int
    split_masks: int
    digital_microvector_addition: bool
    cached_logical_filter_mvm: int
    initial_array_sha256: str
    readback_weight_rmse: float
    train_spikes: int
    test_spikes: int
    accuracy: Accuracy
    error: str
    cleanup_error: str
    raw_finish_error: str
    impacts: int
    pulses: int
    programming_decisions: int
    result_sha256: dict[str,str]
