import importlib
import importlib.util

import pytest
from guppylang import guppy
from guppylang.std.builtins import array, exit, panic, result
from guppylang.std.qsystem.random import RNG
from guppylang.std.qsystem.utils import get_current_shot
from guppylang.std.quantum import (
    cx,
    discard,
    discard_array,
    h,
    measure,
    measure_array,
    qubit,
    t,
    tdg,
    x,
    z,
)
from hugr.ops import CFG
from pytest_snapshot.plugin import Snapshot
from selene_hugr_qis_compiler import HugrReadError, check_hugr, compile_to_llvm_ir

triples = [
    "x86_64-unknown-linux-gnu",
    "x86_64-apple-darwin",
    "aarch64-apple-darwin",
    "x86_64-windows-msvc",
]


def test_check() -> None:
    """Test the check_hugr function to ensure it can load a HUGR envelope."""

    @guppy
    def foo() -> None:
        q = qubit()
        discard(q)

    package = foo.compile()
    hugr_envelope = package.to_bytes()

    check_hugr(hugr_envelope)  # guppy produces a valid HUGR envelope!

    bad_number = hugr_envelope[1:]
    with pytest.raises(HugrReadError, match="Bad magic number"):
        check_hugr(bad_number)

    bad_end = hugr_envelope[:-1]
    with pytest.raises(HugrReadError, match="Premature end of file"):
        check_hugr(bad_end)

    hugr = package.modules[0]
    hugr.add_node(CFG([], []))
    with pytest.raises(HugrReadError, match="CFG must have children"):
        check_hugr(package.to_str().encode("utf-8"))


def test_unsupported_pytket_ops() -> None:
    """Test the check_hugr function to ensure it flags unsupported pytket ops."""
    if importlib.util.find_spec("tket") is None:
        pytest.skip("tket not installed; skipping test of unsupported pytket ops")
    from pytket._tket.circuit import Circuit

    # A pytket circuit with an unsupported op.
    circ = Circuit(2).CSXdg(0, 1)

    guppy_circ = guppy.load_pytket("guppy_circ", circ, use_arrays=False)

    @guppy
    def foo() -> None:
        a, b = qubit(), qubit()
        guppy_circ(a, b)
        discard(a)
        discard(b)

    package = foo.compile()
    hugr_envelope = package.to_bytes()

    with pytest.raises(
        HugrReadError,
        match="Pytket op 'CSXdg' is not currently "
        "supported by the Selene HUGR-QIS compiler",
    ):
        check_hugr(hugr_envelope)


@pytest.mark.parametrize("target_triple", triples)
def test_llvm_no_results(snapshot: Snapshot, target_triple: str) -> None:
    @guppy
    def bar() -> None:
        q0: qubit = qubit()
        h(q0)
        measure(q0)

    hugr_envelope = bar.compile().to_bytes()
    ir = compile_to_llvm_ir(hugr_envelope, target_triple=target_triple)  # type: ignore[call-arg]
    snapshot.assert_match(ir, f"no_results_{target_triple}")


@pytest.mark.parametrize("target_triple", triples)
def test_llvm_flip_some(snapshot: Snapshot, target_triple: str) -> None:
    @guppy
    def main() -> None:
        q0: qubit = qubit()
        q1: qubit = qubit()
        q2: qubit = qubit()
        q3: qubit = qubit()
        x(q0)
        x(q2)
        x(q3)
        result("c0", measure(q0))
        result("c1", measure(q1))
        result("c2", measure(q2))
        result("c3", measure(q3))

    hugr_envelope = main.compile().to_bytes()
    ir = compile_to_llvm_ir(hugr_envelope, target_triple=target_triple)  # type: ignore[call-arg]
    snapshot.assert_match(ir, f"flip_some_{target_triple}")


@pytest.mark.parametrize("target_triple", triples)
def test_llvm_discard_array(snapshot: Snapshot, target_triple: str) -> None:
    @guppy
    def main() -> None:
        qs = array(qubit() for _ in range(10))
        discard_array(qs)

    hugr_envelope = main.compile().to_bytes()
    ir = compile_to_llvm_ir(hugr_envelope, target_triple=target_triple)  # type: ignore[call-arg]
    snapshot.assert_match(ir, f"discard_array_{target_triple}")


@pytest.mark.parametrize("target_triple", triples)
def test_llvm_measure_array(snapshot: Snapshot, target_triple: str) -> None:
    @guppy
    def main() -> None:
        qs = array(qubit() for _ in range(10))
        x(qs[0])
        x(qs[2])
        x(qs[3])
        x(qs[9])
        measure_array(qs)

    hugr_envelope = main.compile().to_bytes()
    ir = compile_to_llvm_ir(hugr_envelope, target_triple=target_triple)  # type: ignore[call-arg]
    snapshot.assert_match(ir, f"measure_array_{target_triple}")


@pytest.mark.parametrize("target_triple", triples)
def test_llvm_print_array(snapshot: Snapshot, target_triple: str) -> None:
    @guppy
    def main() -> None:
        qs = array(qubit() for _ in range(10))
        x(qs[0])
        x(qs[2])
        x(qs[3])
        x(qs[9])
        cs = measure_array(qs)
        result("cs", cs)
        result("is", array(i for i in range(100)))
        result("fs", array(i * 0.0625 for i in range(100)))

    hugr_envelope = main.compile().to_bytes()
    ir = compile_to_llvm_ir(hugr_envelope, target_triple=target_triple)  # type: ignore[call-arg]
    snapshot.assert_match(ir, f"print_array_{target_triple}")


@pytest.mark.parametrize("target_triple", triples)
def test_llvm_exit(snapshot: Snapshot, target_triple: str) -> None:
    """
    This test verifies the behaviour of exit(), which should stop the shot
    and add the error message to the result stream, but should then resume
    further shots.
    """

    @guppy
    def main() -> None:
        q = qubit()
        h(q)
        outcome = measure(q)
        if outcome:
            exit("Postselection failed", 42)
        result("c", outcome)

    hugr_envelope = main.compile().to_bytes()
    ir = compile_to_llvm_ir(hugr_envelope, target_triple=target_triple)  # type: ignore[call-arg]
    snapshot.assert_match(ir, f"exit_{target_triple}")


@pytest.mark.parametrize("target_triple", triples)
def test_llvm_panic(snapshot: Snapshot, target_triple: str) -> None:
    """
    This test verifies the behaviour of panic(), which should stop the shot
    and should not allow any further shots to be performed. On the python
    client side, this should result in an Exception rather than being added
    into the results.
    """

    @guppy
    def main() -> None:
        q = qubit()
        h(q)
        outcome = measure(q)
        if outcome:
            panic("Postselection failed")
        result("c", outcome)

    hugr_envelope = main.compile().to_bytes()
    ir = compile_to_llvm_ir(hugr_envelope, target_triple=target_triple)  # type: ignore[call-arg]
    snapshot.assert_match(ir, f"panic_{target_triple}")


@pytest.mark.parametrize("target_triple", triples)
def test_llvm_rus(snapshot: Snapshot, target_triple: str) -> None:
    @guppy
    def rus(q: qubit) -> None:
        while True:
            # Prepare ancillary qubits
            a, b = qubit(), qubit()
            h(a)
            h(b)

            tdg(a)
            cx(b, a)
            t(a)
            if not measure(a):
                # First part failed; try again
                discard(b)
                continue

            t(q)
            z(q)
            cx(q, b)
            t(b)
            if measure(b):
                # Success, we are done
                break

            # Otherwise, apply correction
            x(q)

    @guppy
    def main() -> None:
        q = qubit()
        rus(q)
        result("result", measure(q))

    hugr_envelope = main.compile().to_bytes()
    ir = compile_to_llvm_ir(hugr_envelope, target_triple=target_triple)  # type: ignore[call-arg]
    snapshot.assert_match(ir, f"rus_{target_triple}")


@pytest.mark.parametrize("target_triple", triples)
def test_llvm_get_current_shot(snapshot: Snapshot, target_triple: str) -> None:
    @guppy
    def main() -> None:
        result("shot", get_current_shot())

    hugr_envelope = main.compile().to_bytes()
    ir = compile_to_llvm_ir(hugr_envelope, target_triple=target_triple)  # type: ignore[call-arg]
    snapshot.assert_match(ir, f"current_shot_{target_triple}")


@pytest.mark.parametrize("target_triple", triples)
def test_llvm_rng(snapshot: Snapshot, target_triple: str) -> None:
    @guppy
    def main() -> None:
        rng = RNG(42)
        rint = rng.random_int()
        rint1 = rng.random_int()
        rfloat = rng.random_float()
        rint_bnd = rng.random_int_bounded(100)
        rng.discard()
        result("rint", rint)
        result("rint1", rint1)
        result("rfloat", rfloat)
        result("rint_bnd", rint_bnd)
        rng = RNG(84)
        rint = rng.random_int()
        rfloat = rng.random_float()
        rint_bnd = rng.random_int_bounded(200)
        rng.discard()
        result("rint2", rint)
        result("rfloat2", rfloat)
        result("rint_bnd2", rint_bnd)

    hugr_envelope = main.compile().to_bytes()
    ir = compile_to_llvm_ir(hugr_envelope, target_triple=target_triple)  # type: ignore[call-arg]
    snapshot.assert_match(ir, f"rng_{target_triple}")


def test_entry_args() -> None:
    @guppy
    def foo(a: int) -> None:
        result("a", a)

    with pytest.raises(
        RuntimeError,
        match="Entry point function must have no input parameters",
    ):
        _ = compile_to_llvm_ir(foo.compile_function().to_bytes())
