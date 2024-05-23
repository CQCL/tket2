from enum import Enum, auto

import tket2


class Tk2Op(Enum):
    """A Tket2 built-in operation."""

    H = auto()
    CX = auto()
    T = auto()
    S = auto()
    X = auto()
    Y = auto()
    Z = auto()
    Tdg = auto()
    Sdg = auto()
    ZZMax = auto()
    Measure = auto()
    RzF64 = auto()
    RxF64 = auto()
    PhasedX = auto()
    ZZPhase = auto()
    AngleAdd = auto()
    CZ = auto()
    TK1 = auto()
    QAlloc = auto()
    QFree = auto()
    Reset = auto()

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
    """Simple enum representation of Pauli matrices."""

    I = auto()  # noqa: E741
    X = auto()
    Y = auto()
    Z = auto()

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
