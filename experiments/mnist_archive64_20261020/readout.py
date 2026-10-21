"""Совместимый readout API; арифметика находится в experiment_common.

Старый patch readout.labels сохранён явной передачей label_reader,
без подмены sys.modules или временного изменения общих глобальных функций.
"""
from collections.abc import Callable
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
import torch
from experiment_common import readout as shared
from experiment_common.mnist_inputs import labels as labels
from experiment_common.schema import Accuracy
from experiment_common.readout import standardized as standardized, wilson as wilson


def fit(counts: NDArray[np.uint8], out: Path, progress: Callable[[str], None]) -> torch.nn.Linear:
    return shared.fit(counts, out, progress, label_reader=labels)


def evaluate(model: torch.nn.Linear, counts: NDArray[np.uint8], out: Path) -> Accuracy:
    return shared.evaluate(model, counts, out, label_reader=labels)
