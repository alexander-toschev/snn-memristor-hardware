"""Только сериализация ответа и захват stdout публичного MemriBoard."""
from __future__ import annotations
import io
import math
from typing import Protocol


def safe(value: object) -> object:
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float:
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    return repr(value)


class EventOwner(Protocol):
    calls: int

    def emit(self, event: str, **fields: object) -> None: ...


class DriverOutput(io.TextIOBase):
    def __init__(self, owner: EventOwner, channel: str) -> None:
        self.owner = owner
        self.channel = channel

    def write(self, text: str) -> int:
        if text:
            self.owner.emit('driver_output', channel=self.channel, text=text, task_id=self.owner.calls)
        return len(text)

    def flush(self) -> None:
        pass
