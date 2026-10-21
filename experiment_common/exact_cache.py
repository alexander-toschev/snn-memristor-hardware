"""Точный кеш native MVM: нет интерполяции, нулевой/идеальной подстановки."""
from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Iterable


@dataclass(frozen=True)
class Response:
    adc: int
    voltage: float
    guards_r: tuple[float, ...]
    task_id: int


class ExactCache:
    def __init__(self, generation: int, masks: Iterable[int]) -> None:
        self.generation=generation
        self.expected={(wl,m) for wl in range(8) for m in masks}
        self.values: dict[tuple[int,int],Response]={}
        self.repeats: dict[tuple[int,int],int]={}
        self.sealed=False

    def observe(self, generation: int, wl: int, mask: int, response: Response) -> None:
        key=(wl,mask)
        if self.sealed or generation!=self.generation or key not in self.expected:
            raise ValueError('Unexpected cache generation/key/state')
        # Предварительный nativeR зависит от служебного last_resistance:
        # он не является выходом mode10. Все guards проверены отдельно при
        # каждом запросе, но exact-cache требует равенства только ADC/V MVM.
        old=self.values.get(key)
        if old is not None and (old.adc,old.voltage)!=(response.adc,response.voltage):
            raise ValueError('Native MVM repeat changed; cache is invalid, no averaging/fallback')
        self.values[key]=response if old is None else old
        self.repeats[key]=self.repeats.get(key,0)+1
        if self.repeats[key]>3:raise ValueError('Unexpected extra query')

    def seal(self) -> None:
        # CPU разрешает reuse только после трёх реальных запросов каждого ключа.
        if set(self.values)!=self.expected or any(n!=3 for n in self.repeats.values()):
            raise ValueError('Missing native query/repeat/end-check')
        self.sealed=True

    def get(self,generation: int,wl: int,mask: int) -> Response:
        # Только чтение измеренного значения из RAM; нового обращения к SIM нет.
        if generation!=self.generation or not self.sealed:raise ValueError('Unsealed/stale native cache')
        try:return self.values[(wl,mask)]
        except KeyError as exc:raise ValueError('Native LUT miss; ideal/zero substitution forbidden') from exc

    def invalidate(self) -> None:
        # Новая запись могла изменить состояние: старая generation неприменима.
        self.sealed=False
        self.values.clear();self.repeats.clear()
