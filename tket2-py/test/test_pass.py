from pytket import Circuit, OpType
from dataclasses import dataclass

from tket2.passes import badger_pass, greedy_depth_reduce, chunks
from tket2.circuit import Tk2Circuit
from tket2.pattern import Rule, RuleMatcher


def test_simple_badger_pass_no_opt():
    c = Circuit(3).CCX(0, 1, 2)
    badger = badger_pass(max_threads=1, timeout=0)
    badger.apply(c)
    assert c.n_gates_of_type(OpType.CX) == 6


@dataclass
class DepthOptimisePass:
    def apply(self, circ: Circuit) -> Circuit:
        (circ, n_moves) = greedy_depth_reduce(circ)
        return circ


def test_depth_optimise():
    c = Circuit(4).CX(0, 2).CX(1, 2).CX(1, 3)

    assert c.depth() == 3

    c = DepthOptimisePass().apply(c)

    assert c.depth() == 2


def test_chunks():
    c = Circuit(4).CX(0, 2).CX(1, 3).CX(1, 2).CX(0, 3).CX(1, 3)

    assert c.depth() == 3

    circ_chunks = chunks(c, 2)
    circuits = circ_chunks.circuits()
    circ_chunks.update_circuit(0, circuits[0])
    c2 = circ_chunks.reassemble()

    assert c2.depth() == 3
    assert type(c2) == Circuit

    # Split and reassemble, with a tket2 circuit
    tk2_chunks = chunks(Tk2Circuit(c2), 2)
    tk2 = tk2_chunks.reassemble()

    assert type(tk2) == Tk2Circuit


def test_cx_rule():
    c = Tk2Circuit(Circuit(4).CX(0, 2).CX(1, 2).CX(1, 2))

    rule = Rule(Circuit(2).CX(0, 1).CX(0, 1), Circuit(2))
    matcher = RuleMatcher([rule])

    mtch = matcher.find_match(c)

    c.apply_match(mtch)

    out = c.to_tket1()

    assert out == Circuit(4).CX(0, 2)


def test_multiple_rules():
    circ = Tk2Circuit(Circuit(3).CX(0, 1).H(0).H(1).H(2).Z(0).H(0).H(1).H(2))

    rule1 = Rule(Circuit(1).H(0).Z(0).H(0), Circuit(1).X(0))
    rule2 = Rule(Circuit(1).H(0).H(0), Circuit(1))
    matcher = RuleMatcher([rule1, rule2])

    match_count = 0
    while match := matcher.find_match(circ):
        match_count += 1
        circ.apply_match(match)

    assert match_count == 3

    out = circ.to_tket1()
    assert out == Circuit(3).CX(0, 1).X(0)
