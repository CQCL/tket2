from typing import no_type_check
import pytket.circuit
from tket2.circuit import Tk2Circuit

import math

from guppylang import guppy
from guppylang.module import GuppyModule
from guppylang.prelude import quantum
from guppylang.prelude.builtins import py
from guppylang.prelude.quantum import measure, phased_x, qubit, rz, zz_max

import pytket


def test_load_pure_circuit():
    module = GuppyModule("test")
    module.load(quantum)

    @guppy(module)
    @no_type_check
    def my_func(
        q0: qubit,
        q1: qubit,
    ) -> tuple[qubit, qubit]:  # pragma: no cover
        q0 = phased_x(q0, py(math.pi / 2), py(-math.pi / 2))
        q0 = rz(q0, py(math.pi))
        q1 = phased_x(q1, py(math.pi / 2), py(-math.pi / 2))
        q1 = rz(q1, py(math.pi))
        q0, q1 = zz_max(q0, q1)
        q0 = rz(q0, py(math.pi))
        q1 = rz(q1, py(math.pi))
        return (q0, q1)

    # Compile the module
    hugr = module.compile()
    json = hugr.to_raw().to_json()

    # Load it from the JSON string
    circ = Tk2Circuit.from_guppy_json(json, "my_func")
    assert circ.num_operations() == 7

    # Convert into a pytket Circuit
    tk1 = circ.to_tket1()
    assert tk1.n_gates == 7
    assert tk1.n_qubits == 2

    gates = list(tk1)
    assert gates[4].op.type == pytket.circuit.OpType.ZZMax


def test_load_hybrid_circuit():
    module = GuppyModule("test")
    module.load(quantum)

    @guppy(module)
    @no_type_check
    def my_func(
        q0: qubit,
        q1: qubit,
    ) -> tuple[bool,]:  # pragma: no cover
        q0 = phased_x(q0, py(math.pi / 2), py(-math.pi / 2))
        q0 = rz(q0, py(math.pi))
        q1 = phased_x(q1, py(math.pi / 2), py(-math.pi / 2))
        q1 = rz(q1, py(math.pi))
        q0, q1 = zz_max(q0, q1)
        _ = measure(q0)
        return (measure(q1),)

    # Compile the module, and convert it to a JSON string
    hugr = module.compile()
    json = hugr.to_raw().to_json()

    # Load the module from the JSON string
    circ = Tk2Circuit.from_guppy_json(json, "my_func")

    # The 7 operations in the function, plus two implicit QFree
    assert circ.num_operations() == 9
