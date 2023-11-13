from dataclasses import dataclass
from pytket.circuit import Circuit

from tket2 import passes
from tket2.passes import greedy_depth_reduce
from tket2.circuit import T2Circuit
from tket2.pattern import Rule, RuleMatcher


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

    chunks = passes.chunks(c, 2)
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
    circ = T2Circuit(Circuit(3).CX(0, 1).H(0).H(1).H(2).Z(0).H(0).H(1).H(2))

    rule1 = Rule(Circuit(1).H(0).Z(0).H(0), Circuit(1).X(0))
    rule2 = Rule(Circuit(1).H(0).H(0), Circuit(1))
    matcher = RuleMatcher([rule1, rule2])

    match_count = 0
    while match := matcher.find_match(circ):
        match_count += 1
        circ.apply_match(match)

    assert match_count == 3

    out = circ.finish()
    assert out == Circuit(3).CX(0, 1).X(0)
