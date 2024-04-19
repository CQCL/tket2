from dataclasses import dataclass
from pytket._tket.circuit import Circuit
import itertools

from tket2.circuit import Tk2Circuit, Tk2Op, to_hugr_dot, Dfg, Node, Gate, Wire
from tket2.pattern import Rule, RuleMatcher


@dataclass
class CustomCost:
    gate_count: int
    h_count: int

    def __add__(self, other):
        return CustomCost(
            self.gate_count + other.gate_count, self.h_count + other.h_count
        )


def test_cost():
    circ = Tk2Circuit(Circuit(4).CX(0, 1).H(1).CX(1, 2).CX(0, 3).H(0))

    print(circ.circuit_cost(lambda op: int(op == Tk2Op.CX)))

    assert circ.circuit_cost(lambda op: int(op == Tk2Op.CX)) == 3
    assert circ.circuit_cost(lambda op: CustomCost(1, op == Tk2Op.H)) == CustomCost(
        5, 2
    )


def test_hash():
    circA = Tk2Circuit(Circuit(4).CX(0, 1).CX(1, 2).CX(0, 3))
    circB = Tk2Circuit(Circuit(4).CX(1, 2).CX(0, 1).CX(0, 3))
    circC = Tk2Circuit(Circuit(4).CX(0, 1).CX(0, 3).CX(1, 2))

    assert hash(circA) != hash(circB)
    assert hash(circA) == hash(circC)


def test_conversion():
    tk1 = Circuit(4).CX(0, 2).CX(1, 2).CX(1, 3)
    tk1_dot = to_hugr_dot(tk1)

    tk2 = Tk2Circuit(tk1)
    tk2_dot = to_hugr_dot(tk2)

    assert type(tk2) == Tk2Circuit
    assert tk1_dot == tk2_dot

    tk1_back = tk2.to_tket1()

    assert tk1_back == tk1
    assert type(tk1_back) == Circuit


class CircBuild:
    dfg: Dfg
    qbs: list[Wire]

    def __init__(self, n_qb: int) -> None:
        self.dfg = Dfg(n_qb)
        self.qbs = self.dfg.inputs()

    def add(self, op: Gate, indices: list[int]) -> Node:
        qbs = [self.qbs[i] for i in indices]
        n = self.dfg.add_op(op, qbs)
        outs = n.outs(len(indices))
        for i, o in zip(indices, outs):
            self.qbs[i] = o

        return n

    def add_c(self, op: Gate, indices: list[int]) -> "CircBuild":
        self.add(op, indices)
        return self

    def finish(self) -> Tk2Circuit:
        return self.dfg.finish(self.qbs)


class CXDef(Gate):
    name = "CX"
    n_qubits = 2


CX = CXDef()


class HDef(Gate):
    name = "H"
    n_qubits = 1


H = HDef()


class PauliXDef(Gate):
    name = "X"
    n_qubits = 1


PauliX = PauliXDef()


class PauliZDef(Gate):
    name = "Z"
    n_qubits = 1


PauliZ = PauliZDef()


class PauliYDef(Gate):
    name = "Y"
    n_qubits = 1


PauliY = PauliYDef()


def merge_rules() -> list[Rule]:
    paulis = [PauliX, PauliY, PauliZ]
    identities = [
        Rule(CircBuild(1).add_c(p, [0]).add_c(p, [0]).finish(), CircBuild(1).finish())
        for p in paulis
    ]

    off_diag = [
        Rule(
            CircBuild(1).add_c(p0, [0]).add_c(p1, [0]).finish(),
            CircBuild(1).add_c(p2, [0]).finish(),
        )
        for p0, p1, p2 in itertools.permutations(paulis)
    ]
    return [*identities, *off_diag]


def propagate_rules() -> list[Rule]:
    hadamard_rules = [
        Rule(
            CircBuild(1).add_c(PauliX, [0]).add_c(H, [0]).finish(),
            CircBuild(1).add_c(H, [0]).add_c(PauliZ, [0]).finish(),
        ),
        Rule(
            CircBuild(1).add_c(PauliZ, [0]).add_c(H, [0]).finish(),
            CircBuild(1).add_c(H, [0]).add_c(PauliX, [0]).finish(),
        ),
    ]

    cx_rules = [
        Rule(
            CircBuild(2).add_c(PauliZ, [0]).add_c(CX, [0, 1]).finish(),
            CircBuild(2).add_c(CX, [0, 1]).add_c(PauliZ, [0]).finish(),
        ),
        Rule(
            CircBuild(2).add_c(PauliX, [1]).add_c(CX, [0, 1]).finish(),
            CircBuild(2).add_c(CX, [0, 1]).add_c(PauliX, [1]).finish(),
        ),
        Rule(
            CircBuild(2).add_c(PauliZ, [1]).add_c(CX, [0, 1]).finish(),
            CircBuild(2)
            .add_c(CX, [0, 1])
            .add_c(PauliZ, [0])
            .add_c(PauliZ, [1])
            .finish(),
        ),
        Rule(
            CircBuild(2).add_c(PauliX, [0]).add_c(CX, [0, 1]).finish(),
            CircBuild(2)
            .add_c(CX, [0, 1])
            .add_c(PauliX, [0])
            .add_c(PauliX, [1])
            .finish(),
        ),
    ]

    return [*hadamard_rules, *cx_rules]


def propagate(circ: Tk2Circuit) -> int:
    propagate_match = RuleMatcher(propagate_rules() + merge_rules())
    match_count = 0
    while match := propagate_match.find_match(circ):
        match_count += 1
        circ.apply_rewrite(match)

    return match_count


def test_simple_z_prop():
    c = Dfg(2)
    q0, q1 = c.inputs()
    # add error
    q0 = c.add_op(PauliX, [q0]).outs(1)[0]
    h_node = c.add_op(H, [q0])
    q0, q1 = c.add_op(CX, [h_node[0], q1]).outs(2)
    t2c = c.finish([q0, q1])

    assert t2c.to_tket1() == Circuit(2).X(0).H(0).CX(0, 1)

    assert propagate(t2c) == 2

    assert t2c.to_tket1() == Circuit(2).H(0).CX(0, 1).Z(0)


def test_cat():
    c = (
        CircBuild(4)
        .add_c(H, [2])
        .add_c(PauliX, [2])
        .add_c(CX, [2, 1])
        .add_c(CX, [2, 3])
        .add_c(CX, [1, 0])
    )
    t2c = c.finish()
    assert t2c.to_tket1() == Circuit(4).H(2).X(2).CX(2, 1).CX(2, 3).CX(1, 0)

    assert propagate(t2c) == 3

    assert t2c.to_tket1() == Circuit(4).H(2).CX(2, 1).CX(2, 3).CX(1, 0).X(0).X(1).X(
        2
    ).X(3)
