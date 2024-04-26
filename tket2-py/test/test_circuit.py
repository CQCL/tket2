from collections.abc import Iterable
from dataclasses import dataclass
from pytket._tket.circuit import Circuit
import itertools

from tket2.circuit import (
    Tk2Circuit,
    Tk2Op,
    to_hugr_dot,
    Dfg,
    Node,
    Gate,
    Wire,
    GateDef,
    Command,
    CustomOp,
)
from tket2.pattern import Rule, RuleMatcher
from tket2.rewrite import Subcircuit, CircuitRewrite


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

    def add_command(self, command: Command) -> Node:
        return self.add(command.gate, command.qubits())

    def extend(self, coms: Iterable[Command]) -> "CircBuild":
        for op in coms:
            self.add_command(op)
        return self

    def finish(self) -> Tk2Circuit:
        return self.dfg.finish(self.qbs)


def from_coms(*args: Command) -> Tk2Circuit:
    commands = []
    n_qb = 0
    # traverses commands twice which isn't great
    for arg in args:
        max_qb = max(arg.qubits()) + 1
        n_qb = max(n_qb, max_qb)
        commands.append(arg)

    build = CircBuild(n_qb)
    build.extend(commands)
    return build.finish()


CXGate = GateDef(2, "CX")
HGate = GateDef(1, "H")
PauliXGate = GateDef(1, "X")
PauliZGate = GateDef(1, "Z")
PauliYGate = GateDef(1, "Y")
QAlloc = CustomOp.new_custom_quantum("quantum.tket2", "QAlloc", (0, 1))
QFree = CustomOp.new_custom_quantum("quantum.tket2", "QFree", (1, 0))


@dataclass(frozen=True)
class H(Command[GateDef]):
    qubit: int

    @property
    def gate(self) -> GateDef:
        return HGate

    def qubits(self) -> list[int]:
        return [self.qubit]


@dataclass(frozen=True)
class CX(Command[GateDef]):
    control: int
    target: int

    @property
    def gate(self) -> GateDef:
        return CXGate

    def qubits(self) -> list[int]:
        return [self.control, self.target]


@dataclass(frozen=True)
class PauliX(Command[GateDef]):
    qubit: int

    @property
    def gate(self) -> GateDef:
        return PauliXGate

    def qubits(self) -> list[int]:
        return [self.qubit]


@dataclass(frozen=True)
class PauliZ(Command[GateDef]):
    qubit: int

    @property
    def gate(self) -> GateDef:
        return PauliZGate

    def qubits(self) -> list[int]:
        return [self.qubit]


@dataclass(frozen=True)
class PauliY(Command[GateDef]):
    qubit: int

    @property
    def gate(self) -> GateDef:
        return PauliYGate

    def qubits(self) -> list[int]:
        return [self.qubit]


def merge_rules() -> list[Rule]:
    paulis = [PauliX(0), PauliY(0), PauliZ(0)]
    identities = [
        Rule(CircBuild(1).extend((p, p)).finish(), CircBuild(1).finish())
        for p in paulis
    ]

    off_diag = [
        Rule(
            CircBuild(1).extend((p0, p1)).finish(),
            CircBuild(1).extend((p2,)).finish(),
        )
        for p0, p1, p2 in itertools.permutations(paulis)
    ]
    return [*identities, *off_diag]


def propagate_rules() -> list[Rule]:
    hadamard_rules = [
        Rule(from_coms(PauliX(0), H(0)), from_coms(H(0), PauliZ(0))),
        Rule(from_coms(PauliZ(0), H(0)), from_coms(H(0), PauliX(0))),
    ]

    cx_rules = [
        Rule(from_coms(PauliZ(0), CX(0, 1)), from_coms(CX(0, 1), PauliZ(0))),
        Rule(from_coms(PauliX(1), CX(0, 1)), from_coms(CX(0, 1), PauliX(1))),
        Rule(from_coms(PauliZ(1), CX(0, 1)), from_coms(CX(0, 1), PauliZ(0), PauliZ(1))),
        Rule(from_coms(PauliX(0), CX(0, 1)), from_coms(CX(0, 1), PauliX(0), PauliX(1))),
    ]

    return [*hadamard_rules, *cx_rules]


def propagate(circ: Tk2Circuit) -> int:
    propagate_match = RuleMatcher(propagate_rules() + merge_rules())
    match_count = 0
    while match := propagate_match.find_match(circ):
        match_count += 1
        circ.apply_rewrite(match)

    return match_count


def add_error_after(circ: Tk2Circuit, n_qb: int, node: Node, error: Gate):
    # TODO infer n_qb by querying port interface of `node`
    subc = Subcircuit([node], circ)
    replace_build = CircBuild(n_qb)
    current = circ.node_op(node)
    replace_build.add(current, [0])
    replace_build.add(error, [0])
    replacement = replace_build.finish()

    rw = CircuitRewrite(subc, circ, replacement)

    circ.apply_rewrite(rw)


def test_simple_z_prop():
    c = Dfg(2)
    q0, q1 = c.inputs()
    h_node_e = c.add_op(HGate, [q0])
    h_node = c.add_op(HGate, h_node_e.outs(1))
    q0, q1 = c.add_op(CXGate, [h_node[0], q1]).outs(2)
    t2c = c.finish([q0, q1])

    add_error_after(t2c, 1, h_node_e, PauliXGate)

    assert t2c.to_tket1() == Circuit(2).H(0).X(0).H(0).CX(0, 1)

    assert propagate(t2c) == 2

    assert t2c.to_tket1() == Circuit(2).H(0).H(0).CX(0, 1).Z(0)


def test_cat():
    c = CircBuild(4)
    h_node = c.add_command(H(2))
    c.extend(
        [CX(2, 1), CX(2, 3), CX(1, 0)],
    )
    t2c = c.finish()

    add_error_after(t2c, 1, h_node, PauliXGate)
    assert t2c.to_tket1() == Circuit(4).H(2).X(2).CX(2, 1).CX(2, 3).CX(1, 0)

    assert propagate(t2c) == 3

    assert t2c.to_tket1() == Circuit(4).H(2).CX(2, 1).CX(2, 3).CX(1, 0).X(0).X(1).X(
        2
    ).X(3)


def test_alloc_free():
    c = CircBuild(0)
    alloc = c.dfg.add_op(QAlloc, [])
    c.dfg.add_op(QFree, alloc.outs(1))
    c.finish()  # validates
