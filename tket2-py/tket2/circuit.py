# Re-export native bindings
from typing import Protocol
from .tket2._circuit import *  # noqa: F403
from .tket2 import _circuit

__all__ = _circuit.__all__


class Gate(Protocol):
    n_qubits: int
    name: str

    def to_custom(self) -> CustomOp:
        return CustomOp.new_custom_quantum(
            "quantum.tket2", self.name, (self.n_qubits, self.n_qubits)
        )
