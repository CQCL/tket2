from pyrs import RsCircuit, WireType, Op
from pytket import Circuit


def simple_rs():
    c = RsCircuit()
    c.py_add_qid("q")
    i, o = c.py_boundary()
    v = c.py_add_vertex(Op.H)
    c.py_add_edge((i, 0), (v, 0), WireType.Qubit)
    c.py_add_edge((v, 0), (o, 0), WireType.Qubit)
    return c


c = Circuit(2).H(0).CX(0, 1)
rc = RsCircuit.from_tket1_circ(c)

print(rc.dot_string())

print(rc.to_tket1_circ().get_commands())


print(simple_rs().to_tket1_circ().get_commands())
