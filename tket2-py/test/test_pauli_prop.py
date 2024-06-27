import itertools

import pytest
from pytket._tket.circuit import Circuit

from tket2.ops import CustomOp, Tk2Op, Not
from tket2.circuit import (
    Dfg,
    Wire,
    Tk2Circuit,
)
from tket2.circuit.build import (
    BOOL_T,
    QB_T,
    CircBuild,
    H,
    from_coms,
    CX,
    PauliX,
    PauliY,
    PauliZ,
)
from tket2.pattern import Rule, RuleMatcher  # type: ignore
from tket2.rewrite import CircuitRewrite, Subcircuit  # type: ignore


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
    q, b = r_build.add_op(Tk2Op.Measure, qs).outs(2)
    ltk = r_build.finish([q, b])

    r_build = Dfg([QB_T], [QB_T, BOOL_T])
    qs = r_build.inputs()
    q, b = r_build.add_op(Tk2Op.Measure, qs).outs(2)
    b = r_build.add_op(Not, [b])[0]
    rtk = r_build.finish([q, b])

    rules = [Rule(ltk, rtk)]

    # Z does not affect measurement result
    r_build = Dfg([QB_T], [QB_T, BOOL_T])
    qs = r_build.inputs()
    qs = r_build.add_op(PauliZ.op(), qs).outs(1)
    q, b = r_build.add_op(Tk2Op.Measure, qs).outs(2)
    ltk = r_build.finish([q, b])

    r_build = Dfg([QB_T], [QB_T, BOOL_T])
    qs = r_build.inputs()
    q, b = r_build.add_op(Tk2Op.Measure, qs).outs(2)
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


def add_error_after(circ: Tk2Circuit, wire: Wire, error: CustomOp):
    """Use a rewrite to insert an operation on a qubit wire assuming the error is
    a one qubit operation, and the source gate of the wire only acts on qubits."""
    node = wire.node()
    port = wire.port()
    n_qb = len(circ.node_outputs(node)) - 1  # ignore Order port
    subc = Subcircuit([node], circ)
    replace_build = CircBuild(n_qb)
    current = circ.node_op(node)
    replace_build.add(current, list(range(n_qb)))
    replace_build.add(error, [port])
    replacement = replace_build.finish()

    rw = CircuitRewrite(subc, circ, replacement)

    circ.apply_rewrite(rw)


def final_pauli_string(circ: Tk2Circuit) -> str:
    """Assuming the circuit only has qubit outputs - check the final operations
    on each qubit, and if they are paulis concatenate them into a string."""

    def map_op(op: CustomOp) -> str:
        # strip the extension name
        n = op.name[len("quantum.tket2.") :]
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

    add_error_after(t2c, h_node_e[0], PauliX.op())

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

    add_error_after(t2c, h_node[0], PauliX.op())
    assert t2c.to_tket1() == Circuit(4).H(2).X(2).CX(2, 1).CX(2, 3).CX(1, 0)

    assert apply_exhaustive(t2c, propagate_matcher) == 3

    assert t2c.to_tket1() == Circuit(4).H(2).CX(2, 1).CX(2, 3).CX(1, 0).X(0).X(1).X(
        2
    ).X(3)

    assert final_pauli_string(t2c) == "XXXX"


def test_alloc_free():
    c = CircBuild(0)
    alloc = c.dfg.add_op(Tk2Op.QAlloc, [])
    c.dfg.add_op(Tk2Op.QFree, alloc.outs(1))
    c.finish()  # validates


def test_measure(propagate_matcher: RuleMatcher):
    c = Dfg([QB_T, QB_T], [QB_T, BOOL_T, QB_T, BOOL_T])
    q0, q1 = c.inputs()
    q0 = c.add_op(PauliX.op(), [q0])[0]
    outs = [w for q in (q0, q1) for w in c.add_op(Tk2Op.Measure, [q]).outs(2)]
    before = c.finish(outs)
    """
    ──►X───►Measure─►
            │
            └────►
    ───────►Measure─►
            │
            └────►
    """
    assert apply_exhaustive(before, propagate_matcher) == 1

    c = Dfg([QB_T, QB_T], [QB_T, BOOL_T, QB_T, BOOL_T])
    q0, q1 = c.inputs()
    q0, b0, q1, b1 = [w for q in (q0, q1) for w in c.add_op(Tk2Op.Measure, [q]).outs(2)]
    b0 = c.add_op(Not, [b0])[0]
    after = c.finish([q0, b0, q1, b1])
    """
    ──►Measure──────►
            └─►Not─►
    ──────►Measure──►
                └────►
    """

    # can't compare using tket1 circuits because measure can't be converted.
    assert hash(before) == hash(after)
