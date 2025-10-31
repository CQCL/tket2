import importlib
import importlib.util
from pathlib import Path

import pytest
from hugr.ops import CFG
from hugr.package import Package
from pytest_snapshot.plugin import Snapshot
from selene_hugr_qis_compiler import HugrReadError, check_hugr, compile_to_llvm_ir

resources_dir = Path(__file__).parent / "resources"

triples = [
    "x86_64-unknown-linux-gnu",
    "x86_64-apple-darwin",
    "aarch64-apple-darwin",
    "x86_64-windows-msvc",
]


def load(name: str) -> bytes:
    hugr_file = resources_dir / f"{name}.hugr"
    return hugr_file.read_bytes()


def test_check() -> None:
    """Test the check_hugr function to ensure it can load a HUGR envelope."""
    hugr_envelope = load("check")
    check_hugr(hugr_envelope)  # guppy produces a valid HUGR envelope!

    bad_number = hugr_envelope[1:]
    with pytest.raises(HugrReadError, match="Bad magic number"):
        check_hugr(bad_number)

    bad_end = hugr_envelope[:-1]
    with pytest.raises(HugrReadError, match="Premature end of file"):
        check_hugr(bad_end)

    package = Package.from_bytes(hugr_envelope)
    hugr = package.modules[0]
    hugr.add_node(CFG([], []))
    with pytest.raises(HugrReadError, match="CFG must have children"):
        check_hugr(package.to_str().encode("utf-8"))


def test_unsupported_pytket_ops() -> None:
    """Test the check_hugr function to ensure it flags unsupported pytket ops."""
    if importlib.util.find_spec("tket") is None:
        pytest.skip("tket not installed; skipping test of unsupported pytket ops")

    hugr_envelope = load("unsupported_pytket_ops")
    with pytest.raises(
        HugrReadError,
        match="Pytket op 'CSXdg' is not currently "
        "supported by the Selene HUGR-QIS compiler",
    ):
        check_hugr(hugr_envelope)


@pytest.mark.parametrize("target_triple", triples)
def test_llvm_no_results(snapshot: Snapshot, target_triple: str) -> None:
    hugr_envelope = load("no_results")
    ir = compile_to_llvm_ir(hugr_envelope, target_triple=target_triple)  # type: ignore[call-arg]
    snapshot.assert_match(ir, f"no_results_{target_triple}")


@pytest.mark.parametrize("target_triple", triples)
def test_llvm_flip_some(snapshot: Snapshot, target_triple: str) -> None:
    hugr_envelope = load("flip_some")
    ir = compile_to_llvm_ir(hugr_envelope, target_triple=target_triple)  # type: ignore[call-arg]
    snapshot.assert_match(ir, f"flip_some_{target_triple}")


@pytest.mark.parametrize("target_triple", triples)
def test_llvm_discard_array(snapshot: Snapshot, target_triple: str) -> None:
    hugr_envelope = load("discard_array")
    ir = compile_to_llvm_ir(hugr_envelope, target_triple=target_triple)  # type: ignore[call-arg]
    snapshot.assert_match(ir, f"discard_array_{target_triple}")


@pytest.mark.parametrize("target_triple", triples)
def test_llvm_measure_array(snapshot: Snapshot, target_triple: str) -> None:
    hugr_envelope = load("measure_array")
    ir = compile_to_llvm_ir(hugr_envelope, target_triple=target_triple)  # type: ignore[call-arg]
    snapshot.assert_match(ir, f"measure_array_{target_triple}")


@pytest.mark.parametrize("target_triple", triples)
def test_llvm_print_array(snapshot: Snapshot, target_triple: str) -> None:
    hugr_envelope = load("print_array")
    ir = compile_to_llvm_ir(hugr_envelope, target_triple=target_triple)  # type: ignore[call-arg]
    snapshot.assert_match(ir, f"print_array_{target_triple}")


@pytest.mark.parametrize("target_triple", triples)
def test_llvm_exit(snapshot: Snapshot, target_triple: str) -> None:
    hugr_envelope = load("postselect_exit")
    ir = compile_to_llvm_ir(hugr_envelope, target_triple=target_triple)  # type: ignore[call-arg]
    snapshot.assert_match(ir, f"exit_{target_triple}")


@pytest.mark.parametrize("target_triple", triples)
def test_llvm_panic(snapshot: Snapshot, target_triple: str) -> None:
    hugr_envelope = load("postselect_panic")
    ir = compile_to_llvm_ir(hugr_envelope, target_triple=target_triple)  # type: ignore[call-arg]
    snapshot.assert_match(ir, f"panic_{target_triple}")


@pytest.mark.parametrize("target_triple", triples)
def test_llvm_rus(snapshot: Snapshot, target_triple: str) -> None:
    hugr_envelope = load("rus")
    ir = compile_to_llvm_ir(hugr_envelope, target_triple=target_triple)  # type: ignore[call-arg]
    snapshot.assert_match(ir, f"rus_{target_triple}")


@pytest.mark.parametrize("target_triple", triples)
def test_llvm_get_current_shot(snapshot: Snapshot, target_triple: str) -> None:
    hugr_envelope = load("get_current_shot")
    ir = compile_to_llvm_ir(hugr_envelope, target_triple=target_triple)  # type: ignore[call-arg]
    snapshot.assert_match(ir, f"current_shot_{target_triple}")


@pytest.mark.parametrize("target_triple", triples)
def test_llvm_rng(snapshot: Snapshot, target_triple: str) -> None:
    hugr_envelope = load("rng")
    ir = compile_to_llvm_ir(hugr_envelope, target_triple=target_triple)  # type: ignore[call-arg]
    snapshot.assert_match(ir, f"rng_{target_triple}")


def test_entry_args() -> None:
    with pytest.raises(
        RuntimeError,
        match="Entry point function must have no input parameters",
    ):
        _ = compile_to_llvm_ir(load("entry_args"))


@pytest.mark.parametrize("target_triple", triples)
def test_gpu(snapshot: Snapshot, target_triple: str) -> None:
    # when we get GPU support in guppy, we might write something like:
    #
    # @gpu_module("example_module.so", None)
    # class Decoder:
    #     @gpu
    #     @no_type_check
    #     def fn_returning_int(
    #         self: "Decoder", a: int, b: float
    #     ) -> int: ...
    #
    #     @gpu
    #     def fn_returning_float(self: "Decoder", x: int) -> float: ...
    #
    # @guppy
    # def main() -> None:
    #     decoder = Decoder()
    #     a = decoder.fn_returning_int(42, 2.71828)
    #     b = decoder.fn_returning_float(a)
    #     result("a", a)
    #     result("b", b)
    #     decoder.discard()
    #
    # hugr_envelope = main.compile().to_bytes()

    # resources/example_gpu.hugr contains the equivalent HUGR to the
    # above, using the tket_qsystem::extension::gpu entities.
    hugr_file = resources_dir / "example_gpu.hugr"
    hugr_envelope = hugr_file.read_bytes()
    ir = compile_to_llvm_ir(hugr_envelope, target_triple=target_triple)  # type: ignore[call-arg]
    snapshot.assert_match(ir, f"gpu_{target_triple}")
