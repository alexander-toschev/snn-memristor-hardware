"""Цифровые LIF/traces и штатное правило STDP; свёртка здесь НЕ вычисляется."""
from __future__ import annotations
import torch
from torch import Tensor
from bindsnet.network.nodes import Input,LIFNodes
from bindsnet.network.topology import Conv2dConnection
from bindsnet.learning import WeightDependentPostPre


def initialization() -> tuple[Tensor,Tensor]:
    # Новое случайное состояние, без checkpoint. Архивная нормировка78.4
    # выводит часть W выше1; отдельно сохраняем её и результат первого clip.
    # Новый seed определяет новую реконструкцию, не неизвестный RNG архива.
    generator=torch.Generator(device='cpu').manual_seed(42)
    raw=torch.empty((64,1,5,5)).uniform_(0.,.3,generator=generator)
    raw*=78.4/raw.abs().sum((1,2,3),keepdim=True).clamp_min(1e-8)
    return raw,raw.clamp(0.,1.)


class DigitalStep:
    def __init__(self) -> None:
        self.source=Input(shape=(1,28,28),traces=True,tc_trace=20.)
        self.target=LIFNodes(shape=(4,24,24),traces=True,tc_trace=20.,
            thresh=.38,rest=0.,reset=0.,refrac=2,tc_decay=150.)
        for layer in (self.source,self.target):
            layer.compute_decays(dt=1.);layer.set_batch_size(1);layer.reset_state_variables()
        self.connection=Conv2dConnection(self.source,self.target,kernel_size=5,stride=1,padding=0,
            w=torch.zeros((4,1,5,5)),wmin=0.,wmax=1.)
        self.rule=WeightDependentPostPre(connection=self.connection,nu=(.001,.0001),weight_decay=1.)

    def reset(self) -> None:
        self.source.reset_state_variables();self.target.reset_state_variables()

    def step(self,current: Tensor,new_input: Tensor,measured_nominal: Tensor) -> tuple[Tensor,Tensor]:
        if current.shape!=(1,4,24,24) or new_input.shape!=(1,1,28,28) or measured_nominal.shape!=(4,25):
            raise ValueError('Wrong online tick shapes')
        if not all(bool(torch.isfinite(v).all()) for v in (current,new_input,measured_nominal)):
            raise ValueError('Nonfinite digital input')
        with torch.no_grad():
            # Это только измеренный nominal W для weight-dependent множителя.
            # Ни connection.compute, ни W@input не вызываются. Forward уже
            # пришёл из mode10 и НЕ ограничивается этим clip.
            self.connection.w.copy_(measured_nominal.reshape(4,1,5,5).clamp(0.,1.))
            # MVMcurrent использует pre[t-1]. Но правило STDP, как Network.run,
            # использует новые pre[t]/trace[t], после обоих forward.
            self.source.forward(new_input.clone());self.target.forward(current.clone())
            self.rule.update()
            # Native WeightDependentPostPre суммирует576 spatial correlations
            # в каждом из25 shared coefficients; pre-rate положителен, LTD
            # задан знаком минус внутри правила. Оба term используют один W.
            return self.target.s.clone(),self.connection.w.reshape(4,25).clone()
