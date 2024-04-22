# Re-export native bindings
from dataclasses import dataclass
from typing import Protocol, TypeVar

from .tket2 import _circuit
from .tket2._circuit import *  # noqa: F403

__all__ = _circuit.__all__


class Gate(Protocol):
    n_qubits: int
    name: str

    def to_custom(self) -> CustomOp:
        return CustomOp.new_custom_quantum(
            "quantum.tket2", self.name, (self.n_qubits, self.n_qubits)
        )


@dataclass(frozen=True)
class GateDef(Gate):
    n_qubits: int
    name: str


T = TypeVar("T", bound=Gate)


class Command(Protocol[T]):
    gate: T

    def qubits(self) -> list[int]: ...
