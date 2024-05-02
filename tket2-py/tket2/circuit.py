# Re-export native bindings
from dataclasses import dataclass
from typing import Protocol, TypeVar

from .tket2 import _circuit
from .tket2._circuit import *  # noqa: F403

__all__ = _circuit.__all__


QB_T = HugrType.qubit()
LB_T = HugrType.linear_bit()
BOOL_T = HugrType.bool()


class ToCustom(Protocol):
    def to_custom(self) -> CustomOp: ...


@dataclass(frozen=True)
class GateDef(ToCustom):
    n_qubits: int
    name: str
    n_bits: int = 0

    def to_custom(self) -> CustomOp:
        types = [QB_T] * self.n_qubits + [LB_T] * self.n_bits
        return CustomOp("quantum.tket2", self.name, types, types)


T = TypeVar("T", bound=ToCustom)


class Command(Protocol[T]):
    gate: ToCustom

    def qubits(self) -> list[int]: ...
    def bits(self) -> list[int]:
        return []
