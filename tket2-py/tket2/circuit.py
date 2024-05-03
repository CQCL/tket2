# Re-export native bindings
from typing import Protocol

from .tket2 import _circuit
from .tket2._circuit import *  # noqa: F403

__all__ = _circuit.__all__

from .tket2._circuit import HugrType, CustomOp

QB_T = HugrType.qubit()
LB_T = HugrType.linear_bit()
BOOL_T = HugrType.bool()


class Command(Protocol):
    gate_name: str
    n_qb: int
    n_lb: int = 0
    extension_name: str = "quantum.tket2"

    def qubits(self) -> list[int]: ...
    def bits(self) -> list[int]:
        return []

    @classmethod
    def op(cls) -> CustomOp:
        types = [QB_T] * cls.n_qb + [LB_T] * cls.n_lb
        return CustomOp(cls.extension_name, cls.gate_name, types, types)
