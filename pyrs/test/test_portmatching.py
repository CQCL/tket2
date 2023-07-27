from pytket import Circuit
from pytket.qasm import circuit_from_qasm
from pyrs.pyrs import patterns


def test_simple_matching():
    """ a simple circuit matching test """
    c = Circuit(2).CX(0, 1).H(1).CX(0, 1)

    p1 = patterns.CircuitPattern(Circuit(2).CX(0, 1).H(1))
    p2 = patterns.CircuitPattern(Circuit(2).H(0).CX(1, 0))

    matcher = patterns.CircuitMatcher(iter([p1, p2]))

    assert len(matcher.find_matches(c)) == 2


def test_non_convex_pattern():
    """ two-qubit circuits can't match three-qb ones """
    p1 = patterns.CircuitPattern(Circuit(3).CX(0, 1).CX(1, 2))
    matcher = patterns.CircuitMatcher(iter([p1]))

    c = Circuit(2).CX(0, 1).CX(1, 0)
    assert len(matcher.find_matches(c)) == 0

    c = Circuit(2).CX(0, 1).CX(1, 0).CX(1, 2)
    assert len(matcher.find_matches(c)) == 0

    c = Circuit(2).H(0).CX(0, 1).CX(1, 0).CX(0, 2)
    assert len(matcher.find_matches(c)) == 1


def test_larger_matching():
    """ a larger crafted circuit with matches WIP """
    c = circuit_from_qasm("test/test_files/circ.qasm")

    p1 = patterns.CircuitPattern(Circuit(2).CX(0, 1).H(1))
    p2 = patterns.CircuitPattern(Circuit(2).H(0).CX(1, 0))

    matcher = patterns.CircuitMatcher(iter([p1, p2]))

    print(matcher.find_matches(c))
    assert len(matcher.find_matches(c)) == 3