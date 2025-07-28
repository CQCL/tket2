from pytket import Circuit
from tket2.optimiser import BadgerOptimiser


def test_simple_optimiser():
    """a simple circuit matching test"""
    c = Circuit(3).CX(0, 1).CX(0, 1).CX(1, 2)
    opt = BadgerOptimiser.compile_eccs("test_files/cx_cx_eccs.json")

    # optimise for 1s
    cc = opt.optimise(c, 3)
    exp_c = Circuit(3).CX(1, 2)

    assert cc == exp_c
