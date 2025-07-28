import tket2
from tket2.ops import Tk2Op, Pauli


def test_ops_roundtrip():
    for op in Tk2Op:
        assert Tk2Op._from_rs(op._to_rs()) == op

    for op in tket2._tket2.ops.Tk2Op.values():
        assert Tk2Op._from_rs(op)._to_rs() == op


def test_pauli_roundtrip():
    for pauli in Pauli:
        assert Pauli._from_rs(pauli._to_rs()) == pauli

    for pauli in tket2._tket2.ops.Pauli.values():
        assert Pauli._from_rs(pauli)._to_rs() == pauli
