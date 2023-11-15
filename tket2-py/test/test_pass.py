from pytket import Circuit, OpType
from pytket.qasm.qasm import circuit_from_qasm
from tket2.circuit import T2Circuit
from tket2.passes import badger_pass


def test_simple_badger_pass_no_opt():
    c = Circuit(3).CCX(0, 1, 2)
    badger = badger_pass(max_threads=1, timeout=0)
    badger.apply(c)
    assert c.n_gates_of_type(OpType.CX) == 6


def test_failing():
    """a failing circuit optimisation test"""
    c = circuit_from_qasm("test_files/failing_simple.qasm")
    c = T2Circuit(c)

    # print(c.to_hugr_json())
    # print(c.to_tket1_json())

    c = c.to_tket1()

    badger = badger_pass(max_threads=1, timeout=0)
    badger.apply(c)

    assert False
