from torch import Tensor
from .nodes import Nodes
class Conv2dConnection:
    w: Tensor
    def __init__(self,source: Nodes,target: Nodes,*,kernel_size: int,stride: int,padding: int,w: Tensor,wmin: float,wmax: float) -> None: ...
