from hugr.envelope import EnvelopeConfig
from tket.passes import lower_to_pytket
from tket.circuit import Tk2Circuit
from typing import Any
from guppylang import guppy
from guppylang.std.quantum import qubit, cx, rz, h, tdg, measure, discard, t, z
from guppylang.std.angles import angle
from guppylang.std.builtins import owned, result
import pytest

# mypy: disable-error-code=valid-type
# mypy: disable-error-code=call-arg


@guppy
def empty_func() -> int:
    return 1


@guppy
def const_op() -> angle:
    x = angle(3.141)
    return x


@guppy
def alias_op(x: int) -> int:
    y = x
    return y


@guppy
def one_rz(q0: qubit @ owned, angle: angle) -> qubit:
    rz(q0, angle)
    return q0


@guppy
def loop_conditional(q0: qubit @ owned, angle: angle, n: int, cond: bool) -> qubit:
    while n > 0:
        if cond:
            h(q0)
        else:
            rz(q0, angle)
        n = n - 1
    return q0


@guppy
def conditional_loop(q0: qubit @ owned, angle: angle, n: int, cond: bool) -> qubit:
    if cond:
        while n > 0:
            h(q0)
            n = n - 1
    else:
        rz(q0, angle)
    return q0


@guppy
def inner(q0: qubit @ owned, angle: angle) -> qubit:
    rz(q0, angle)
    return q0


@guppy
def mid(q0: qubit @ owned) -> qubit:
    a = angle(3.14)
    return inner(q0, a)


@guppy
def outer(q0: qubit @ owned) -> qubit:
    h(q0)
    return mid(q0)


@guppy
def outer_decl(q0: qubit @ owned) -> qubit:
    h(q0)

    @guppy
    def inner_decl(q0: qubit @ owned) -> qubit:
        h(q0)
        return q0

    return inner_decl(q0)


@guppy
def repeat_until_success(q: qubit) -> None:
    attempts = 0
    while True:
        attempts += 1

        a, b = qubit(), qubit()
        h(a)
        h(b)

        tdg(a)
        cx(b, a)
        t(a)
        h(a)
        if measure(a):
            discard(b)
            continue

        t(q)
        z(q)
        cx(q, b)
        t(b)
        h(b)
        if measure(b):
            z(q)
            continue

        result("attempts", attempts)
        break


def guppy_to_circuit(func_def: Any) -> Tk2Circuit:
    """Convert a Guppy function definition to a `Tk2Circuit`."""

    pkg = func_def.compile_function()

    f_name = pkg.modules[0].entrypoint_op().f_name

    json = pkg.to_str(EnvelopeConfig.TEXT)
    circ = Tk2Circuit.from_str(json, f_name)

    return lower_to_pytket(circ)


testdata = [
    (empty_func, 0),
    (const_op, 0),
    (alias_op, 0),
    (one_rz, 2),
    (loop_conditional, 8),
    (conditional_loop, 8),
    (outer, 3),
    (outer_decl, 2),
    (repeat_until_success, 21),
]


@pytest.mark.parametrize("test_input,expected", testdata)
def test_count_ops(test_input, expected):
    circ = guppy_to_circuit(test_input)
    assert circ.num_operations() == expected
