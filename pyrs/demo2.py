from dataclasses import dataclass
from pyrs import RsCircuit, WireType, Op, PyOpenCircuit, PySubgraph, CircuitRewrite
from pytket import Circuit


@dataclass
class Qubit:
    name: str
    index: list[int]


def simple_rs(op):
    c = RsCircuit()
    c.add_unitid(Qubit("q", [0]))
    i, o = c.py_boundary()
    v = c.add_vertex(op)
    c.add_edge((i, 0), (v, 0), WireType.Qubit)
    c.add_edge((v, 0), (o, 0), WireType.Qubit)

    return c


c = Circuit(2).H(0).CX(0, 1)
rc = RsCircuit.from_tket1_circ(c)

print(rc.dot_string())
print(rc.to_tket1_circ().get_commands())


print(simple_rs(Op.H).to_tket1_circ().get_commands())


c = simple_rs(Op.H)
c2 = simple_rs(Op.Reset)

c.apply_rewrite(CircuitRewrite(PySubgraph({2}, [0], [1]), PyOpenCircuit(c2), 0.0))

print(c.to_tket1_circ().get_commands())
