from typing import Callable, Iterable
import time
from functools import wraps

import pytest
from pyrs import (
    RsCircuit,
    WireType,
    RsOpType,
    Subgraph,
    CircuitRewrite,
    greedy_pattern_rewrite,
    remove_redundancies,
    Direction,
    greedy_iter_rewrite,
    Rational,
    Quaternion,
    Angle,
    check_soundness,
    NodePort,
)

from pytket import Circuit, OpType, Qubit


def simple_rs(op):
    c = RsCircuit()
    c.add_unitid(Qubit("q", [0]))
    i, o = c.boundary()
    v = c.add_vertex(op)
    c.add_edge(NodePort(i, 0), NodePort(v, 0), WireType.Qubit)
    c.add_edge(NodePort(v, 0), NodePort(o, 0), WireType.Qubit)
    check_soundness(c)
    return c


def test_conversion():
    c = Circuit(2).H(0).CX(0, 1)
    rc = RsCircuit.from_tket1(c)
    assert len(rc.to_tket1().get_commands()) == 2

    assert rc.dot_string()


def test_apply_rewrite():

    c = simple_rs(RsOpType.H)
    assert c.edge_endpoints(0) == (
        NodePort(0, 0),
        NodePort(2, 0),
    )
    assert c.edge_at_port(NodePort(2, 0), Direction.Outgoing) == 1
    c2 = simple_rs(RsOpType.Reset)

    c.apply_rewrite(CircuitRewrite(Subgraph({2}, [0], [1]), c2, 0.0))
    c.defrag()  # needed for exact equality check
    assert c == c2
    assert c.remove_node(2) == RsOpType.Reset
    assert c.remove_node(2) == None


@pytest.fixture()
def cx_circ() -> RsCircuit:
    return RsCircuit.from_tket1(Circuit(2).CX(0, 1).CX(0, 1))


def _noop_circ() -> RsCircuit:
    c = Circuit(2)
    c.add_gate(OpType.noop, [0])
    c.add_gate(OpType.noop, [1])
    return RsCircuit.from_tket1(c)


@pytest.fixture()
def noop_circ() -> RsCircuit:
    return _noop_circ()


def timed(f: Callable):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time.time()
        out = f(*args, **kwargs)
        print(time.time() - start)
        return out

    return wrapper


def cx_pair_searcher(circ: RsCircuit) -> Iterable[CircuitRewrite]:
    for nid in circ.node_indices():
        if circ.node_op(nid) != RsOpType.CX:
            continue
        sucs = circ.node_edges(nid, Direction.Outgoing)

        if len(sucs) != 2:
            continue

        source0, target0 = circ.edge_endpoints(sucs[0])
        source1, target1 = circ.edge_endpoints(sucs[1])
        if target0.node != target1.node:
            # same node
            continue
        next_nid = target0.node
        if circ.node_op(next_nid) != RsOpType.CX:
            continue

        # check ports match
        if source0.port == target0.port and source1.port == target1.port:
            in_edges = circ.node_edges(nid, Direction.Incoming)
            out_edges = circ.node_edges(next_nid, Direction.Outgoing)
            yield CircuitRewrite(
                Subgraph({nid, next_nid}, in_edges, out_edges), _noop_circ(), 0.0
            )


def test_cx_rewriters(cx_circ, noop_circ):
    c = Circuit(2).H(0).CX(1, 0).CX(1, 0)
    rc = RsCircuit.from_tket1(c)
    assert rc.node_edges(3, Direction.Incoming) == [1, 2]
    assert rc.neighbours(4, Direction.Outgoing) == [
        NodePort(1, 1),
        NodePort(1, 0),
    ]
    check_soundness(rc)
    # each one of these ways of applying this rewrite should take longer than
    # the one before

    c1 = timed(greedy_pattern_rewrite)(rc, cx_circ, lambda x: noop_circ)

    c2 = timed(greedy_pattern_rewrite)(
        rc, cx_circ, lambda x: noop_circ, lambda ni, op: op == cx_circ.node_op(ni)
    )

    c3 = timed(greedy_iter_rewrite)(rc, cx_pair_searcher)

    correct = Circuit(2).H(0)
    correct.add_gate(OpType.noop, [1])
    correct.add_gate(OpType.noop, [0])
    for c in (c1, c2, c3):
        check_soundness(c)
        assert c.to_tket1().get_commands() == correct.get_commands()


def test_equality():
    bell_circ = lambda: RsCircuit.from_tket1(Circuit(2).H(0).CX(0, 1))
    assert bell_circ() == bell_circ()
    assert bell_circ() != RsCircuit.from_tket1(Circuit(2).H(0))


def test_auto_convert():
    c = Circuit(2).CX(0, 1).CX(0, 1).Rx(2, 1)
    c2 = remove_redundancies(c)
    correct = Circuit(2).Rx(2, 1)

    assert c2 == correct


def test_const():
    rat = Rational(1, 2)
    quat = Quaternion([0.1, 0.2, 0.3, 0.4])
    ang1 = Angle.rational(rat)
    ang2 = Angle.float(2.3)

    c = RsCircuit()
    for const in (True, 2, 4.5, quat, ang1, ang2):
        v = c.add_const(const)
        assert c.get_const(v) == const

    assert c.get_const(0) == c.get_const(1) == None
    pass
