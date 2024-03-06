from guppylang import guppy, qubit, quantum, GuppyModule
from guppylang.prelude.quantum import h, cx

# from pytket._tket.circuit import Circuit

from tket2.circuit import (
    Tk2Circuit,
)


def test_simple_guppy():
    module = GuppyModule("simple")
    module.load(quantum)

    @guppy(module)
    def qubit_id(q: qubit) -> qubit:
        return q

    _circ = Tk2Circuit(qubit_id)

    # TODO: Currently the parsed `circ` is a module with a single function
    # We need to extract the function first to use it as a circuit
    """
    assert circ.circuit_cost(lambda _op: 1) == 0
    """


def test_quantum_guppy():
    module = GuppyModule("mod")
    module.load(quantum)

    @guppy(module)
    def qubit_fn(q1: qubit, q2: qubit) -> tuple[qubit, qubit]:
        q2 = h(q2)
        q1, q2 = cx(q1, q2)
        q2 = h(q2)
        return (q1, q2)

    _circ = Tk2Circuit(module)

    # TODO: Currently the parsed `circ` is a module with a single function
    # We need to extract the function first to use it as a circuit
    """
    circ_as_tk1 = circ.to_tket1()

    tk1 = Circuit(2).H(1).CX(0, 1).H(1)

    assert tk1.n_qubits == circ_as_tk1.n_qubits
    assert tk1.n_gates == circ_as_tk1.n_gates
    assert tk1.depth() == circ_as_tk1.depth()
    """
