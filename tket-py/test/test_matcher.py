from typing import TYPE_CHECKING, Any, List
from tket.circuit import Tk2Circuit
from tket.matcher import RotationMatcher, MatchReplaceRewriter
from tket.protocol import CircuitMatcher, CircuitReplacer
from pytket.circuit import Circuit as Tk1Circuit


class EmptyReplacement:
    def __init__(self):
        pass

    def replace_match(self, circuit: Tk2Circuit, match_info: Any) -> List[Tk2Circuit]:
        n_qubits = circuit.to_tket1().n_qubits

        return [Tk2Circuit(Tk1Circuit(n_qubits))]


if TYPE_CHECKING:
    _: type[CircuitReplacer] = EmptyReplacement
    __: type[CircuitMatcher] = RotationMatcher


def test_is_instance():
    matcher = RotationMatcher()
    assert isinstance(EmptyReplacement(), CircuitReplacer)
    assert isinstance(matcher, CircuitMatcher)


def test_run_matcher():
    matcher = RotationMatcher()
    rewriter = MatchReplaceRewriter(matcher, EmptyReplacement())

    circ = Tk1Circuit(2).Rz(0.2, 0).Rx(0.1, 1)

    assert len(rewriter.get_rewrites(Tk2Circuit(circ))) == 2
