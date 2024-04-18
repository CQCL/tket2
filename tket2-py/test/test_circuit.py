from dataclasses import dataclass
from pytket._tket.circuit import Circuit

from tket2.circuit import Tk2Circuit, Tk2Op, to_hugr_dot, Dfg, Node, Gate


@dataclass
class CustomCost:
    gate_count: int
    h_count: int

    def __add__(self, other):
        return CustomCost(
            self.gate_count + other.gate_count, self.h_count + other.h_count
        )


def test_cost():
    circ = Tk2Circuit(Circuit(4).CX(0, 1).H(1).CX(1, 2).CX(0, 3).H(0))

    print(circ.circuit_cost(lambda op: int(op == Tk2Op.CX)))

    assert circ.circuit_cost(lambda op: int(op == Tk2Op.CX)) == 3
    assert circ.circuit_cost(lambda op: CustomCost(1, op == Tk2Op.H)) == CustomCost(
        5, 2
    )


def test_hash():
    circA = Tk2Circuit(Circuit(4).CX(0, 1).CX(1, 2).CX(0, 3))
    circB = Tk2Circuit(Circuit(4).CX(1, 2).CX(0, 1).CX(0, 3))
    circC = Tk2Circuit(Circuit(4).CX(0, 1).CX(0, 3).CX(1, 2))

    assert hash(circA) != hash(circB)
    assert hash(circA) == hash(circC)


def test_conversion():
    tk1 = Circuit(4).CX(0, 2).CX(1, 2).CX(1, 3)
    tk1_dot = to_hugr_dot(tk1)

    tk2 = Tk2Circuit(tk1)
    tk2_dot = to_hugr_dot(tk2)

    assert type(tk2) == Tk2Circuit
    assert tk1_dot == tk2_dot

    tk1_back = tk2.to_tket1()

    assert tk1_back == tk1
    assert type(tk1_back) == Circuit


class CXDef(Gate):
    name = "CX"
    n_qubits = 2


CX = CXDef()


class HDef(Gate):
    name = "H"
    n_qubits = 1


H = HDef()


class PauliXDef(Gate):
    name = "X"
    n_qubits = 1


PauliX = PauliXDef()


class PauliZDef(Gate):
    name = "Z"
    n_qubits = 1


PauliZ = PauliZDef()


def insert_after(c: Tk2Circuit, node: Node, op: Gate):
    ...

def test_append():
    c = Dfg(2)
    q0, q1 = c.inputs()
    h_node = c.add_op(H, [q0])
    q0, q1 = c.add_op(CX, [h_node[0], q1]).outs(2)
    q0, q1 = c.add_op(CX, [q1, q0]).outs(2)

    q0 = c.add_op(PauliX, [q0]).outs(1)[0]
    t2c = c.finish([q0, q1])

    print(t2c.node_op(h_node))
    print(t2c.to_tket1().get_commands())
