from __future__ import annotations
from hugr.hugr import Hugr
from hugr import tys, ops
from hugr.ops import ComWire, Command
from hugr.std.float import FLOAT_T
from hugr.build.tracked_dfg import TrackedDfg
from tket2.circuit import Tk2Circuit

from dataclasses import dataclass


class CircBuild(TrackedDfg):
    """Helper class to build a circuit from commands by tracking qubits,
    allowing commands to be specified by qubit index."""

    @classmethod
    def with_nqb(cls, n_qb: int) -> CircBuild:
        return cls(*[tys.Qubit] * n_qb, track_inputs=True)

    def finish(self) -> Tk2Circuit:
        """Finish building the circuit by setting all the qubits as the output
        and validate."""
        return load_hugr(self.hugr)


def from_coms(*args: Command) -> Tk2Circuit:
    """Build a circuit from a sequence of commands, assuming
    only qubits are referred to by index."""
    commands: list[Command] = []
    n_qb = 0
    # traverses commands twice which isn't great
    for arg in args:
        max_qb = max(i for i in arg.incoming if isinstance(i, int)) + 1
        n_qb = max(n_qb, max_qb)
        commands.append(arg)

    build = CircBuild.with_nqb(n_qb)
    build.extend(*commands)
    build.set_tracked_outputs()
    return build.finish()


def load_hugr(h: Hugr) -> Tk2Circuit:
    return Tk2Circuit.from_hugr_json(h.to_json())


def load_custom(serialized: bytes) -> ops.Custom:
    import hugr._serialization.ops as sops
    import json

    # TODO: We should return an "ExtOp" instead
    ext = json.loads(serialized)
    return sops.ExtensionOp(**ext).deserialize()


def id_circ(n_qb: int) -> Tk2Circuit:
    b = CircBuild.with_nqb(n_qb)
    b.set_tracked_outputs()
    return b.finish()


@dataclass(frozen=True)
class QuantumOps(ops.Custom):
    extension: tys.ExtensionId = "quantum.tket2"


_OneQbSig = tys.FunctionType.endo([tys.Qubit])


@dataclass(frozen=True)
class OneQbGate(QuantumOps):
    op_name: str  # type: ignore[misc] # no-default fields follows one with a default
    num_out: int = 1
    signature: tys.FunctionType = _OneQbSig

    def __call__(self, q: ComWire) -> Command:
        return super().__call__(q)


H = OneQbGate("H")
PauliX = OneQbGate("X")
PauliY = OneQbGate("Y")
PauliZ = OneQbGate("Z")

_TwoQbSig = tys.FunctionType.endo([tys.Qubit] * 2)


@dataclass(frozen=True)
class TwoQbGate(QuantumOps):
    op_name: str  # type: ignore[misc] # no-default fields follows one with a default
    num_out: int = 2
    signature: tys.FunctionType = _TwoQbSig

    def __call__(self, q0: ComWire, q1: ComWire) -> Command:
        return super().__call__(q0, q1)


CX = TwoQbGate("CX")

_MeasSig = tys.FunctionType([tys.Qubit], [tys.Qubit, tys.Bool])


@dataclass(frozen=True)
class MeasureDef(QuantumOps):
    op_name: str = "Measure"
    num_out: int = 2
    signature: tys.FunctionType = _MeasSig

    def __call__(self, q: ComWire) -> Command:
        return super().__call__(q)


Measure = MeasureDef()

# TODO use angle type once extension is serialised.
_RzSig = tys.FunctionType([tys.Qubit, FLOAT_T], [tys.Qubit])


@dataclass(frozen=True)
class RzDef(QuantumOps):
    op_name: str = "Rz"
    num_out: int = 1
    signature: tys.FunctionType = _RzSig

    def __call__(self, q: ComWire, fl_wire: ComWire) -> Command:
        return super().__call__(q, fl_wire)


Rz = RzDef()


_QallocSig = tys.FunctionType([], [tys.Qubit])


@dataclass(frozen=True)
class QAllocDef(QuantumOps):
    op_name: str = "QAlloc"
    num_out: int = 1
    signature: tys.FunctionType = _QallocSig

    def __call__(self) -> Command:
        return super().__call__()


QAlloc = QAllocDef()


_QfreeSig = tys.FunctionType([tys.Qubit], [])


@dataclass(frozen=True)
class QFreeDef(QuantumOps):
    op_name: str = "QFree"
    num_out: int = 0
    signature: tys.FunctionType = _QfreeSig

    def __call__(self, qb: ComWire) -> Command:
        return super().__call__(qb)


QFree = QFreeDef()
