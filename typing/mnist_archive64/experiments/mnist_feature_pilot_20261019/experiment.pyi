from pathlib import Path
from typing import TypedDict
from torch import Tensor
class Checkpoint(TypedDict):
    W: Tensor
    v_thresh: Tensor
    cfg: dict[str,object]
CHECKPOINT: Path
CHECKPOINT_SHA: str
def load_checkpoint() -> Checkpoint: ...
