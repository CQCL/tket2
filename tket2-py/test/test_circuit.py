from collections.abc import Iterable
from dataclasses import dataclass
import pytest
from pytket._tket.circuit import Circuit
import itertools

from tket2.circuit import (
    BOOL_T,
    Tk2Circuit,
    Tk2Op,
    to_hugr_dot,
    Dfg,
    Node,
    Wire,
    Command,
    CustomOp,
    QB_T,
)
from tket2.pattern import Rule, RuleMatcher  # type: ignore
from tket2.rewrite import Subcircuit, CircuitRewrite  # type: ignore


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
        self.dfg = Dfg([QB_T] * n_qb, [QB_T] * n_qb)
        self.qbs = self.dfg.inputs()

    def add(self, op: CustomOp, indices: list[int]) -> Node:
        qbs = [self.qbs[i] for i in indices]
        n = self.dfg.add_op(op, qbs)
        outs = n.outs(len(indices))
        for i, o in zip(indices, outs):
            self.qbs[i] = o

        return n

    def measure_all(self) -> list[Wire]:
        return [self.add(Measure, [i]).outs(2)[1] for i in range(len(self.qbs))]

    def add_command(self, command: Command) -> Node:
        return self.add(command.op(), command.qubits())

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


QAlloc = CustomOp("quantum.tket2", "QAlloc", [], [QB_T])
QFree = CustomOp("quantum.tket2", "QFree", [QB_T], [])
Measure = CustomOp("quantum.tket2", "Measure", [QB_T], [QB_T, BOOL_T])
Not = CustomOp("logic", "Not", [BOOL_T], [BOOL_T])


@dataclass(frozen=True)
class H(Command):
    qubit: int
    gate_name = "H"
    n_qb = 1

    def qubits(self) -> list[int]:
        return [self.qubit]


@dataclass(frozen=True)
class CX(Command):
    control: int
    target: int
    gate_name = "CX"
    n_qb = 2

    def qubits(self) -> list[int]:
        return [self.control, self.target]


@dataclass(frozen=True)
class PauliX(Command):
    qubit: int
    gate_name = "X"
    n_qb = 1

    def qubits(self) -> list[int]:
        return [self.qubit]


@dataclass(frozen=True)
class PauliZ(Command):
    qubit: int
    gate_name = "Z"
    n_qb = 1

    def qubits(self) -> list[int]:
        return [self.qubit]


@dataclass(frozen=True)
class PauliY(Command):
    qubit: int
    gate_name = "Y"
    n_qb = 1

    def qubits(self) -> list[int]:
        return [self.qubit]


@pytest.fixture
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


@pytest.fixture
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


@pytest.fixture
def measure_rules() -> list[Rule]:
    # X flips the measured bit
    r_build = Dfg([QB_T], [QB_T, BOOL_T])
    qs = r_build.inputs()
    qs = r_build.add_op(PauliX.op(), qs).outs(1)
    q, b = r_build.add_op(Measure, qs).outs(2)
    ltk = r_build.finish([q, b])

    r_build = Dfg([QB_T], [QB_T, BOOL_T])
    qs = r_build.inputs()
    q, b = r_build.add_op(Measure, qs).outs(2)
    b = r_build.add_op(Not, [b])[0]
    rtk = r_build.finish([q, b])

    rules = [Rule(ltk, rtk)]

    # Z does not affect measurement result
    r_build = Dfg([QB_T], [QB_T, BOOL_T])
    qs = r_build.inputs()
    qs = r_build.add_op(PauliZ.op(), qs).outs(1)
    qs, b = r_build.add_op(Measure, qs).outs(2)
    ltk = r_build.finish([qs, b])

    r_build = Dfg([QB_T], [QB_T, BOOL_T])
    qs = r_build.inputs()
    q, b = r_build.add_op(Measure, qs).outs(2)
    rtk = r_build.finish([q, b])

    rules.append(Rule(ltk, rtk))

    return rules


@pytest.fixture
def propagate_matcher(
    merge_rules: list[Rule], propagate_rules: list[Rule], measure_rules: list[Rule]
) -> RuleMatcher:
    return RuleMatcher([*merge_rules, *propagate_rules, *measure_rules])


def apply_exhaustive(circ: Tk2Circuit, matcher: RuleMatcher) -> int:
    """Apply the first matching rule until no more matches are found. Return the
    number of rewrites applied."""
    match_count = 0
    while match := matcher.find_match(circ):
        match_count += 1
        circ.apply_rewrite(match)

    return match_count


def add_error_after(circ: Tk2Circuit, node: Node, error: CustomOp):
    n_qb = len(circ.node_outputs(node)) - 1  # ignore Order port
    subc = Subcircuit([node], circ)
    replace_build = CircBuild(n_qb)
    current = circ.node_op(node)
    replace_build.add(current, [0])
    replace_build.add(error, [0])
    replacement = replace_build.finish()

    rw = CircuitRewrite(subc, circ, replacement)

    circ.apply_rewrite(rw)


def final_pauli_string(circ: Tk2Circuit) -> str:
    """Assuming the circuit only has qubit outputs - check the final operations
    on each qubit, and if they are paulis concatenate them into a string."""

    def map_op(op: CustomOp) -> str:
        # strip the extension name
        n = op.name()[len("quantum.tket2.") :]
        return n if n in ("X", "Y", "Z") else "I"

    # TODO ignore non-qubit outputs
    return "".join(
        map_op(circ.node_op(w.node())) for w in circ.node_inputs(circ.output_node())
    )


def test_simple_z_prop(propagate_matcher: RuleMatcher):
    c = Dfg([QB_T] * 2, [QB_T] * 2)
    q0, q1 = c.inputs()
    h_node_e = c.add_op(H.op(), [q0])
    h_node = c.add_op(H.op(), h_node_e.outs(1))
    q0, q1 = c.add_op(CX.op(), [h_node[0], q1]).outs(2)
    t2c = c.finish([q0, q1])

    add_error_after(t2c, h_node_e, PauliX.op())

    assert t2c.to_tket1() == Circuit(2).H(0).X(0).H(0).CX(0, 1)

    assert apply_exhaustive(t2c, propagate_matcher) == 2

    assert t2c.to_tket1() == Circuit(2).H(0).H(0).CX(0, 1).Z(0)

    assert final_pauli_string(t2c) == "ZI"


def test_cat(propagate_matcher: RuleMatcher):
    c = CircBuild(4)
    h_node = c.add_command(H(2))
    t2c = c.extend(
        [CX(2, 1), CX(2, 3), CX(1, 0)],
    ).finish()

    add_error_after(t2c, h_node, PauliX.op())
    assert t2c.to_tket1() == Circuit(4).H(2).X(2).CX(2, 1).CX(2, 3).CX(1, 0)

    assert apply_exhaustive(t2c, propagate_matcher) == 3

    assert t2c.to_tket1() == Circuit(4).H(2).CX(2, 1).CX(2, 3).CX(1, 0).X(0).X(1).X(
        2
    ).X(3)

    assert final_pauli_string(t2c) == "XXXX"


def test_alloc_free():
    c = CircBuild(0)
    alloc = c.dfg.add_op(QAlloc, [])
    c.dfg.add_op(QFree, alloc.outs(1))
    c.finish()  # validates


def test_measure(propagate_matcher: RuleMatcher):
    c = Dfg([QB_T, QB_T], [QB_T, BOOL_T, QB_T, BOOL_T])
    q0, q1 = c.inputs()
    q0 = c.add_op(PauliX.op(), [q0])[0]
    outs = [w for q in (q0, q1) for w in c.add_op(Measure, [q]).outs(2)]
    before = c.finish(outs)

    assert apply_exhaustive(before, propagate_matcher) == 1

    c = Dfg([QB_T, QB_T], [QB_T, BOOL_T, QB_T, BOOL_T])
    q0, q1 = c.inputs()
    q0, b0, q1, b1 = [w for q in (q0, q1) for w in c.add_op(Measure, [q]).outs(2)]
    b0 = c.add_op(Not, [b0])[0]
    after = c.finish([q0, b0, q1, b1])

    # can't compare using tket1 circuits because measure can't be converted.
    assert hash(before) == hash(after)
