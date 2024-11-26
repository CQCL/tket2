from __future__ import annotations

from enum import Enum, auto
from typing import Protocol

import tket2

from tket2._tket2.ops import CustomOp
from tket2.types import QB_T

__all__ = ["CustomOp", "ToCustomOp", "Tk2Op", "Pauli"]


class ToCustomOp(Protocol):
    """Operation that can be converted to a HUGR CustomOp."""

    def to_custom(self) -> CustomOp:
        """Convert to a custom operation."""
        ...

    @property
    def name(self) -> str:
        """Name of the operation."""
        ...


class Tk2Op(Enum):
    """A Tket2 built-in operation.

    Implements the `ToCustomOp` protocol.
    """

    H = auto()
    CX = auto()
    CY = auto()
    CZ = auto()
    CRz = auto()
    T = auto()
    Tdg = auto()
    S = auto()
    Sdg = auto()
    X = auto()
    Y = auto()
    Z = auto()
    Rz = auto()
    Rx = auto()
    Ry = auto()
    Toffoli = auto()
    Measure = auto()
    MeasureFree = auto()
    QAlloc = auto()
    TryQAlloc = auto()
    QFree = auto()
    Reset = auto()

    def to_custom(self) -> CustomOp:
        """Convert to a custom operation."""
        return self._to_rs().to_custom()

    def _to_rs(self) -> tket2._tket2.ops.Tk2Op:
        """Convert to the Rust-backed Tk2Op representation."""
        return tket2._tket2.ops.Tk2Op(self.name)

    @staticmethod
    def _from_rs(op: tket2._tket2.ops.Tk2Op) -> "Tk2Op":
        """Convert from the Rust-backed Tk2Op representation."""
        return Tk2Op[op.name]

    def __eq__(self, other: object) -> bool:
        """Check if two Tk2Ops are equal."""
        if isinstance(other, Tk2Op):
            return self.name == other.name
        elif isinstance(other, tket2._tket2.ops.Tk2Op):
            return self == Tk2Op._from_rs(other)
        elif isinstance(other, str):
            return self.name == other
        return False


class Pauli(Enum):
    """Simple enum representation of Pauli matrices.

    Implements the `ToCustomOp` protocol.
    """

    I = auto()  # noqa: E741
    X = auto()
    Y = auto()
    Z = auto()

    def to_custom(self) -> CustomOp:
        extension_name = "tket2.quantum"
        gate_name = self.name
        return CustomOp(extension_name, gate_name, [QB_T], [QB_T])

    def _to_rs(self) -> tket2._tket2.ops.Pauli:
        """Convert to the Rust-backed Pauli representation."""
        return tket2._tket2.ops.Pauli(self.name)

    @staticmethod
    def _from_rs(pauli: tket2._tket2.ops.Pauli) -> "Pauli":
        """Convert from the Rust-backed Pauli representation."""
        return Pauli[pauli.name]

    def __eq__(self, other: object) -> bool:
        """Check if two Paulis are equal."""
        if isinstance(other, Pauli):
            return self.name == other.name
        elif isinstance(other, tket2._tket2.ops.Pauli):
            return self == Pauli._from_rs(other)
        elif isinstance(other, str):
            return self.name == other
        return False
