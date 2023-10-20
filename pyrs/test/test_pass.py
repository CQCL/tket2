from pytket import Circuit, OpType
from pyrs.pyrs import passes


def test_simple_taso_pass_no_opt():
    c = Circuit(3).CCX(0, 1, 2)
    c = passes.taso_optimise(c, max_threads = 1, timeout = 0)
    print(c)
    assert c.n_gates_of_type(OpType.CX) == 6