from pytket import Circuit, OpType
from tket2.passes import badger_pass


def test_simple_badger_pass_no_opt():
    c = Circuit(3).CCX(0, 1, 2)
    badger = badger_pass(max_threads=1, timeout=0)
    badger.apply(c)
    assert c.n_gates_of_type(OpType.CX) == 6
