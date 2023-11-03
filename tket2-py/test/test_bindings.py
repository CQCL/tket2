from dataclasses import dataclass
from tket2 import passes, circuit, pattern

from pytket.circuit import Circuit

# TODO clean up after fixing module structure #169
Rule, RuleMatcher = pattern.Rule, pattern.RuleMatcher
T2Circuit = circuit.T2Circuit


@dataclass
class DepthOptimisePass:
    def apply(self, circ: Circuit) -> Circuit:
        (circ, n_moves) = passes.greedy_depth_reduce(circ)
        return circ


def test_depth_optimise():
    c = Circuit(4).CX(0, 2).CX(1, 2).CX(1, 3)

    assert c.depth() == 3

    c = DepthOptimisePass().apply(c)

    assert c.depth() == 2


def test_chunks():
    c = Circuit(4).CX(0, 2).CX(1, 3).CX(1, 2).CX(0, 3).CX(1, 3)

    assert c.depth() == 3

    chunks = circuit.chunks(c, 2)
    circuits = chunks.circuits()
    chunks.update_circuit(0, circuits[0])
    c2 = chunks.reassemble()

    assert c2.depth() == 3


def test_cx_rule():
    c = T2Circuit(Circuit(4).CX(0, 2).CX(1, 2).CX(1, 2))

    rule = Rule(Circuit(2).CX(0, 1).CX(0, 1), Circuit(2))
    matcher = RuleMatcher([rule])

    mtch = matcher.find_match(c)

    c.apply_match(mtch)

    out = c.finish()

    assert out == Circuit(4).CX(0, 2)


def test_multiple_rules():
    circuit = T2Circuit(Circuit(3).CX(0, 1).H(0).H(1).H(2).Z(0).H(0).H(1).H(2))

    rule1 = Rule(Circuit(1).H(0).Z(0).H(0), Circuit(1).X(0))
    rule2 = Rule(Circuit(1).H(0).H(0), Circuit(1))
    matcher = RuleMatcher([rule1, rule2])

    match_count = 0
    while match := matcher.find_match(circuit):
        match_count += 1
        circuit.apply_match(match)

    assert match_count == 3

    out = circuit.finish()
    assert out == Circuit(3).CX(0, 1).X(0)


# from dataclasses import dataclass
# from typing import Callable, Iterable
# import time
# from functools import wraps

# import pytest
# from tket2 import (
#     RsCircuit,
#     WireType,
#     RsOpType,
#     Subgraph,
#     CircuitRewrite,
#     greedy_pattern_rewrite,
#     remove_redundancies,
#     Direction,
#     greedy_iter_rewrite,
#     Rational,
#     Quaternion,
#     Angle,
#     check_soundness,
#     CustomOp,
#     Signature,
#     decompose_custom_pass,
#     count_pycustom,
# )
# from tket2.custom_base import CustomOpBase

# from pytket import Circuit, OpType, Qubit


# def simple_rs(op):
#     c = RsCircuit()
#     v = c.add_vertex_with_edges(
#         op,
#         [c.new_input(WireType.Qubit)],
#         [c.new_output(WireType.Qubit)],
#     )
#     check_soundness(c)
#     return c


# def test_conversion():
#     c = Circuit(2).H(0).CX(0, 1)
#     rc = RsCircuit.from_tket1(c)
#     assert len(rc.to_tket1().get_commands()) == 2

#     assert rc.dot_string()


# def test_apply_rewrite():
#     c = simple_rs(RsOpType.H)
#     assert c.edge_endpoints(0) == (0, 2)
#     assert c.edge_at_port(2, 0, Direction.Outgoing) == 1
#     c2 = simple_rs(RsOpType.Reset)

#     c.apply_rewrite(CircuitRewrite(Subgraph({2}, [0], [1]), c2, 0.0))
#     c.defrag()  # needed for exact equality check
#     print(c.dot_string())
#     print(c2.dot_string())
#     assert c == c2
#     assert c.remove_node(2) == RsOpType.Reset
#     assert c.remove_node(2) == None


# @pytest.fixture()
# def cx_circ() -> RsCircuit:
#     return RsCircuit.from_tket1(Circuit(2).CX(0, 1).CX(0, 1))


# def _noop_circ() -> RsCircuit:
#     c = Circuit(2)
#     c.add_gate(OpType.noop, [0])
#     c.add_gate(OpType.noop, [1])
#     return RsCircuit.from_tket1(c)


# @pytest.fixture()
# def noop_circ() -> RsCircuit:
#     return _noop_circ()


# def timed(f: Callable):
#     @wraps(f)
#     def wrapper(*args, **kwargs):
#         start = time.time()
#         out = f(*args, **kwargs)
#         print(time.time() - start)
#         return out

#     return wrapper


# def cx_pair_searcher(circ: RsCircuit) -> Iterable[CircuitRewrite]:
#     for nid in circ.node_indices():
#         if circ.node_op(nid) != RsOpType.CX:
#             continue
#         sucs = circ.node_edges(nid, Direction.Outgoing)

#         if len(sucs) != 2:
#             continue

#         _, target0 = circ.edge_endpoints(sucs[0])
#         _, target1 = circ.edge_endpoints(sucs[1])
#         if target0 != target1:
#             # same node
#             continue
#         next_nid = target0
#         if circ.node_op(next_nid) != RsOpType.CX:
#             continue

#         s0p = circ.port_of_edge(nid, sucs[0], Direction.Outgoing)
#         t0p = circ.port_of_edge(next_nid, sucs[0], Direction.Incoming)

#         s1p = circ.port_of_edge(nid, sucs[1], Direction.Outgoing)
#         t1p = circ.port_of_edge(next_nid, sucs[1], Direction.Incoming)
#         # check ports match
#         if s0p == t0p and s1p == t1p:
#             in_edges = circ.node_edges(nid, Direction.Incoming)
#             out_edges = circ.node_edges(next_nid, Direction.Outgoing)
#             yield CircuitRewrite(
#                 Subgraph({nid, next_nid}, in_edges, out_edges), _noop_circ(), 0.0
#             )


# def test_cx_rewriters(cx_circ, noop_circ):
#     c = Circuit(2).H(0).CX(1, 0).CX(1, 0)
#     rc = RsCircuit.from_tket1(c)
#     assert rc.node_edges(3, Direction.Incoming) == [3, 4]
#     assert rc.neighbours(4, Direction.Outgoing) == [1, 1]
#     check_soundness(rc)
#     # each one of these ways of applying this rewrite should take longer than
#     # the one before

#     c1 = timed(greedy_pattern_rewrite)(rc, cx_circ, lambda x: noop_circ)

#     c2 = timed(greedy_pattern_rewrite)(
#         rc, cx_circ, lambda x: noop_circ, lambda ni, op: op == cx_circ.node_op(ni)
#     )

#     c3 = timed(greedy_iter_rewrite)(rc, cx_pair_searcher)

#     correct = Circuit(2).H(0)
#     correct.add_gate(OpType.noop, [1])
#     correct.add_gate(OpType.noop, [0])
#     for c in (c1, c2, c3):
#         check_soundness(c)
#         assert c.to_tket1().get_commands() == correct.get_commands()


# def test_equality():
#     bell_circ = lambda: RsCircuit.from_tket1(Circuit(2).H(0).CX(0, 1))
#     assert bell_circ() == bell_circ()
#     assert bell_circ() != RsCircuit.from_tket1(Circuit(2).H(0))


# def test_auto_convert():
#     c = Circuit(2).CX(0, 1).CX(0, 1).Rx(2, 1)
#     c2 = remove_redundancies(c)
#     correct = Circuit(2).Rx(2, 1)

#     assert c2 == correct


# def test_const():
#     rat = Rational(1, 2)
#     quat = Quaternion([0.1, 0.2, 0.3, 0.4])
#     ang1 = Angle.rational(rat)
#     ang2 = Angle.float(2.3)

#     c = RsCircuit()
#     for const in (True, 2, 4.5, quat, ang1, ang2):
#         v = c.add_const(const)
#         assert c.get_const(v) == const

#     assert c.get_const(0) == c.get_const(1) == None
#     pass


# @dataclass
# class CustomBridge(CustomOpBase):
#     flip: bool

#     def signature(self) -> Signature:
#         return Signature([WireType.Qubit] * 3, ([], []))

#     def to_circuit(self) -> RsCircuit:
#         c = RsCircuit()

#         for i in range(3):
#             c.add_linear_unitid(Qubit("q", [i]))

#         if self.flip:
#             c.append(RsOpType.CX, [1, 2])
#             c.append(RsOpType.CX, [0, 1])
#             c.append(RsOpType.CX, [1, 2])
#             c.append(RsOpType.CX, [0, 1])
#         else:
#             c.append(RsOpType.CX, [0, 1])
#             c.append(RsOpType.CX, [1, 2])
#             c.append(RsOpType.CX, [0, 1])
#             c.append(RsOpType.CX, [1, 2])
#         return c


# @pytest.mark.parametrize("flip", (True, False))
# def test_custom(flip):
#     c = RsCircuit()
#     for i in range(3):
#         c.add_linear_unitid(Qubit("q", [i]))
#     op = CustomOp(CustomBridge(flip))
#     c.append(op, [0, 1, 2])
#     assert count_pycustom(c) == 1

#     c, success = decompose_custom_pass(c)
#     check_soundness(c)
#     assert success
#     assert c.node_count() == 6
#     assert count_pycustom(c) == 0
