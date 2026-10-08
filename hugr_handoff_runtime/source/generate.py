"""Generate deterministic, fixed-size HUGRs for optimizer handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import no_type_check

import numpy as np
import zixy.qubit.pauli as zqp
from guppyalgos.algorithms.block_encoding.lcu import (
    LCU,
    LCUData,
    build_unary_iteration_select,
)
from guppyalgos.algorithms.select.qrom import qrom_unary_iteration
from guppyalgos.algorithms.state_preparation import multiplexor_prep
from guppyalgos.algorithms.state_preparation.alias_sampling import (
    AliasSamplingRegs,
    alias_samp_prep,
)
from guppyalgos.primitives.arithmetic import (
    adder_ripple_cuccaro_carry_out,
    adder_ripple_gidney_carry_out,
    multiplier_ripple_gidney_mod,
)
from guppyalgos.primitives.gate_decompositions.and_op import (
    temp_and_compute,
    temp_and_uncompute,
)
from guppyalgos.primitives.subroutines.ladders.ccx_v_chain import ccx_v_chain
from guppyalgos.utils import apply_bitstring, int_to_bits, qarray
from guppylang import guppy
from guppylang.std.builtins import array, output
from guppylang.std.quantum import (
    collect_measurements,
    measure,
    measure_array,
    qubit,
    toffoli,
)
from hugr.package import Package

HERE = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = HERE / "artifacts"
MANIFEST_PATH = HERE / "manifest.json"


@dataclass(frozen=True)
class Artifact:
    """A named HUGR package and its handoff metadata."""

    name: str
    family: str
    description: str
    fixed_inputs: dict[str, object]
    named_outputs: dict[str, str]
    logical_qubits: int
    peak_qubit_budget: int
    build: Callable[[], Package]


def _qrom_artifact(n_address: int, n_data: int, *, plain: bool = False) -> Artifact:
    modulus = 2**n_data
    data = [
        int_to_bits((5 * address + 3) % modulus, n_data)
        for address in range(2**n_address)
    ]
    lookup = qrom_unary_iteration(
        data,
        comp_and_op=toffoli if plain else temp_and_compute,
        uncomp_and_op=toffoli if plain else temp_and_uncompute,
    )
    address_value = 2 ** (n_address - 1) + 1
    address_input = array(*int_to_bits(address_value, n_address))

    def build() -> Package:
        @guppy
        @no_type_check
        def main() -> None:
            address = qarray(n_address)
            target = qarray(n_data)
            apply_bitstring(address, address_input)
            lookup(address, target)
            output("address", collect_measurements(measure_array(address)))
            output("data", collect_measurements(measure_array(target)))

        return main.compile()

    return Artifact(
        name=f"qrom_unary_addr{n_address}_data{n_data}_"
        + ("toffoli" if plain else "temp_and"),
        family="qrom",
        description=(
            f"Unary-iteration lookup with {2**n_address} fixed {n_data}-bit words "
            + (
                "and ordinary Toffoli compute/uncompute."
                if plain
                else "and 4T temporary-AND compute / measurement uncompute."
            )
        ),
        fixed_inputs={"address": address_value},
        named_outputs={
            "address": f"bit[{n_address}]",
            "data": f"bit[{n_data}]",
        },
        logical_qubits=n_address + n_data,
        peak_qubit_budget=n_address + n_data + n_address - 1,
        build=build,
    )


def _adder_artifact(width: int) -> Artifact:
    a_value = 2**width - 3
    b_value = 5
    a_input = array(*int_to_bits(a_value, width))
    b_input = array(*int_to_bits(b_value, width))

    def build() -> Package:
        @guppy
        @no_type_check
        def main() -> None:
            a = qarray(width)
            b = qarray(width)
            carry = qubit()
            apply_bitstring(a, a_input)
            apply_bitstring(b, b_input)
            adder_ripple_gidney_carry_out(a, b, carry)
            output("a", collect_measurements(measure_array(a)))
            output("sum", collect_measurements(measure_array(b)))
            output("carry", measure(carry).read())

        return main.compile()

    return Artifact(
        name=f"arithmetic_gidney_adder_n{width}",
        family="arithmetic",
        description=(
            f"{width}-bit Gidney ripple-carry addition with a separate carry-out."
        ),
        fixed_inputs={"a": a_value, "b": b_value},
        named_outputs={
            "a": f"bit[{width}]",
            "sum": f"bit[{width}]",
            "carry": "bit",
        },
        logical_qubits=2 * width + 1,
        peak_qubit_budget=3 * width,
        build=build,
    )


def _multiplier_artifact(width: int) -> Artifact:
    a_value = 2 ** (width - 1) - 1
    b_value = 3
    a_input = array(*int_to_bits(a_value, width))
    b_input = array(*int_to_bits(b_value, width))

    def build() -> Package:
        @guppy
        @no_type_check
        def main() -> None:
            a = qarray(width)
            b = qarray(width)
            product = qarray(width)
            apply_bitstring(a, a_input)
            apply_bitstring(b, b_input)
            multiplier_ripple_gidney_mod(a, b, product)
            output("a", collect_measurements(measure_array(a)))
            output("b", collect_measurements(measure_array(b)))
            output("product", collect_measurements(measure_array(product)))

        return main.compile()

    return Artifact(
        name=f"arithmetic_gidney_multiplier_n{width}",
        family="arithmetic",
        description=f"{width}-bit Gidney shift-and-add multiplier modulo 2^{width}.",
        fixed_inputs={"a": a_value, "b": b_value},
        named_outputs={
            "a": f"bit[{width}]",
            "b": f"bit[{width}]",
            "product": f"bit[{width}]",
        },
        logical_qubits=3 * width,
        peak_qubit_budget=5 * width - 1,
        build=build,
    )


def _rotation_lcu_artifact(name: str, hamiltonian_text: str, n_state: int) -> Artifact:
    hamiltonian = zqp.RealTermSum.from_str(hamiltonian_text, n_state)
    data = LCUData.from_hamiltonian(hamiltonian)
    prepare = multiplexor_prep(data.amplitudes)
    select = build_unary_iteration_select(data)
    n_prep = data.n_prep_qubits
    n_state_qubits = data.n_state_qubits
    state_value = 1
    state_input = array(*int_to_bits(state_value, n_state_qubits))
    dagger = object()

    @guppy
    @no_type_check
    def unprepare(prep: array[qubit, n_prep]) -> None:
        with dagger:
            prepare(prep)

    def build() -> Package:
        @guppy
        @no_type_check
        def main() -> None:
            prep = qarray(n_prep)
            state = qarray(n_state_qubits)
            apply_bitstring(state, state_input)
            LCU(prepare, select, unprepare).compose(prep, state)
            output("prepare", collect_measurements(measure_array(prep)))
            output("state", collect_measurements(measure_array(state)))

        return main.compile()

    return Artifact(
        name=name,
        family="lcu",
        description=(
            f"{data.n_terms}-term, {data.n_state_qubits}-qubit LCU block encoding "
            "with default library rotation PREPARE and unary SELECT."
        ),
        fixed_inputs={"state": state_value},
        named_outputs={
            "prepare": f"bit[{data.n_prep_qubits}]",
            "state": f"bit[{data.n_state_qubits}]",
        },
        logical_qubits=data.n_prep_qubits + data.n_state_qubits,
        peak_qubit_budget=(
            data.n_prep_qubits + data.n_state_qubits + data.n_prep_qubits - 1
        ),
        build=build,
    )


def _alias_lcu_artifact() -> Artifact:
    hamiltonian = zqp.RealTermSum.from_str(
        "(0.25, Z0), (-0.125, X1), (0.125, Y0 Y1), (0.125, Z0 X1), "
        "(0.125, X0), (0.125, Z1), (0.0625, X0 Z1), (-0.0625, Z0 Z1)",
        2,
    )
    data = LCUData.from_hamiltonian(hamiltonian)
    n_index = data.n_prep_qubits
    n_state = data.n_state_qubits
    n_precision = 4
    probabilities = np.abs(data.coeffs) / data.l1_norm
    alias_prepare = alias_samp_prep(probabilities, precision=1 / 2**n_precision)
    unary_select = build_unary_iteration_select(data)
    state_value = 1
    state_input = array(*int_to_bits(state_value, n_state))

    @guppy
    @no_type_check
    def prepare(regs: AliasSamplingRegs[n_index, n_precision]) -> None:
        alias_prepare(
            regs.index,
            regs.alternative,
            regs.keep,
            regs.comparison,
            regs.comparison_result,
            False,
        )

    @guppy
    @no_type_check
    def select(
        regs: AliasSamplingRegs[n_index, n_precision],
        state: array[qubit, n_state],
    ) -> None:
        unary_select(regs.index, state)

    @guppy
    @no_type_check
    def unprepare(regs: AliasSamplingRegs[n_index, n_precision]) -> None:
        alias_prepare(
            regs.index,
            regs.alternative,
            regs.keep,
            regs.comparison,
            regs.comparison_result,
            True,
        )

    def build() -> Package:
        @guppy
        @no_type_check
        def main() -> None:
            regs = AliasSamplingRegs(
                qarray(n_index),
                qarray(n_index),
                qarray(n_precision),
                qarray(n_precision),
                qubit(),
            )
            state = qarray(n_state)
            apply_bitstring(state, state_input)
            LCU(prepare, select, unprepare).compose(regs, state)
            output("index", collect_measurements(measure_array(regs.index)))
            output("alternative", collect_measurements(measure_array(regs.alternative)))
            output("keep", collect_measurements(measure_array(regs.keep)))
            output("comparison", collect_measurements(measure_array(regs.comparison)))
            output("comparison_result", measure(regs.comparison_result).read())
            output("state", collect_measurements(measure_array(state)))

        return main.compile()

    logical_qubits = 2 * n_index + 2 * n_precision + 1 + data.n_state_qubits
    return Artifact(
        name="lcu_alias_8term_state2_precision4",
        family="lcu",
        description=(
            "Eight-term, two-qubit LCU using four-bit alias-sampling PREPARE, "
            "default library QROM, comparison, and unary SELECT."
        ),
        fixed_inputs={"state": state_value},
        named_outputs={
            "index": f"bit[{n_index}]",
            "alternative": f"bit[{n_index}]",
            "keep": f"bit[{n_precision}]",
            "comparison": f"bit[{n_precision}]",
            "comparison_result": "bit",
            "state": f"bit[{data.n_state_qubits}]",
        },
        logical_qubits=logical_qubits,
        # Two unary-iteration work qubits and one comparator work qubit are
        # sufficient in addition to the persistent registers.
        peak_qubit_budget=logical_qubits + n_index,
        build=build,
    )


def register_artifact(width: int, *, v_chain: bool = False) -> Artifact:
    """Call the default library Cuccaro adder or V-chain."""
    a_value = 1 if v_chain else 2**width - 1
    b_value = 2**width - 1 if v_chain else 1
    a_input = array(*int_to_bits(a_value, width))
    b_input = array(*int_to_bits(b_value, width))

    @guppy
    @no_type_check
    def adder_main() -> None:
        a = qarray(width)
        b = qarray(width)
        carry = qubit()
        apply_bitstring(a, a_input)
        apply_bitstring(b, b_input)
        adder_ripple_cuccaro_carry_out(a, b, carry)
        output("a", collect_measurements(measure_array(a)))
        output("sum", collect_measurements(measure_array(b)))
        output("carry", measure(carry).read())

    @guppy
    @no_type_check
    def chain_main() -> None:
        a = qarray(width)
        b = qarray(width)
        target = qubit()
        apply_bitstring(a, a_input)
        apply_bitstring(b, b_input)
        ccx_v_chain(a, b, target)
        output("a", collect_measurements(measure_array(a)))
        output("b", collect_measurements(measure_array(b)))
        output("target", measure(target).read())

    outputs = (
        {"a": f"bit[{width}]", "b": f"bit[{width}]", "target": "bit"}
        if v_chain
        else {
            "a": f"bit[{width}]",
            "sum": f"bit[{width}]",
            "carry": "bit",
        }
    )
    qubits = 2 * width + 1
    return Artifact(
        name=f"toffoli_v_chain_n{width}"
        if v_chain
        else f"arithmetic_cuccaro_adder_n{width}",
        family="toffoli_chain" if v_chain else "arithmetic",
        description="Ordinary-Toffoli runtime V-chain."
        if v_chain
        else "Default library Cuccaro carry-out adder.",
        fixed_inputs={"a": a_value, "b": b_value},
        named_outputs=outputs,
        logical_qubits=qubits,
        peak_qubit_budget=qubits if v_chain else qubits + 1,
        build=chain_main.compile if v_chain else adder_main.compile,
    )


def select_artifact(n_terms: int) -> Artifact:
    """Build ordinary-Toffoli SELECT using the unmodified library factory."""
    terms = [
        "(1, Z0)",
        "(-1, X1)",
        "(1, Y0 Y1)",
        "(1, Z0 X1)",
        "(1, X0)",
        "(1, Z1)",
        "(1, X0 Z1)",
        "(-1, Z0 Z1)",
    ]
    data = LCUData.from_hamiltonian(
        zqp.RealTermSum.from_str(", ".join(terms[:n_terms]), 2)
    )
    select = build_unary_iteration_select(data)
    n_index = data.n_prep_qubits
    address_input = array(*int_to_bits(1, n_index))

    @guppy
    @no_type_check
    def main() -> None:
        index = qarray(n_index)
        state = qarray(2)
        apply_bitstring(index, address_input)
        select(index, state)
        output("index", collect_measurements(measure_array(index)))
        output("state", collect_measurements(measure_array(state)))

    return Artifact(
        name=f"select_pauli_{n_terms}term_state2_toffoli",
        family="select",
        description="Runtime library Pauli SELECT with ordinary Toffolis.",
        fixed_inputs={"index": 1, "state": 0, "terms": terms[:n_terms]},
        named_outputs={"index": f"bit[{n_index}]", "state": "bit[2]"},
        logical_qubits=n_index + 2,
        peak_qubit_budget=2 * n_index + 1,
        build=main.compile,
    )


def artifact_specs() -> list[Artifact]:
    """Return the fixed artifact suite in stable manifest order."""
    return [
        _qrom_artifact(3, 3),
        _qrom_artifact(4, 4),
        _adder_artifact(4),
        _adder_artifact(8),
        _multiplier_artifact(3),
        _multiplier_artifact(4),
        _rotation_lcu_artifact(
            "lcu_rotation_4term_state2",
            "(0.4, Z0), (-0.2, X1), (0.3, Y0 Y1), (0.1, Z0 X1)",
            2,
        ),
        _rotation_lcu_artifact(
            "lcu_rotation_8term_state2",
            "(0.25, Z0), (-0.125, X1), (0.125, Y0 Y1), (0.125, Z0 X1), "
            "(0.125, X0), (0.125, Z1), (0.0625, X0 Z1), (-0.0625, Z0 Z1)",
            2,
        ),
        _alias_lcu_artifact(),
        _qrom_artifact(3, 3, plain=True),
        _qrom_artifact(4, 4, plain=True),
        select_artifact(4),
        select_artifact(8),
        *[register_artifact(n) for n in (4, 8, 16)],
        *[register_artifact(n, v_chain=True) for n in (4, 8, 16)],
    ]


def _sha256(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def _verify_package(
    path: Path, expected_sha256: str | None = None
) -> dict[str, object]:
    contents = path.read_bytes()
    digest = _sha256(contents)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(
            f"Checksum mismatch for {path.name}: {digest} != {expected_sha256}"
        )
    package = Package.from_str(contents.decode("utf-8"))
    # Serializing the parsed package forces schema/model validation as well as
    # checking that this is a complete HUGR envelope.
    package.to_bytes()
    return {"bytes": len(contents), "sha256": digest}


def generate() -> None:
    """Compile every fixed program and write its HUGR plus a manifest."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for artifact in artifact_specs():
        print(f"Compiling {artifact.name} ...", flush=True)
        package = artifact.build()
        path = ARTIFACT_DIR / f"{artifact.name}.hugr"
        path.write_text(package.to_str(), encoding="utf-8")
        file_metadata = _verify_package(path)
        records.append(
            {
                "file": f"artifacts/{path.name}",
                "family": artifact.family,
                "description": artifact.description,
                "entrypoint_inputs": {},
                "fixed_inputs": artifact.fixed_inputs,
                "named_outputs": artifact.named_outputs,
                "logical_qubits": artifact.logical_qubits,
                "peak_qubit_budget": artifact.peak_qubit_budget,
                **file_metadata,
            }
        )

    manifest = {
        "format": "HUGR envelope (text)",
        "schema_version": 1,
        "versions": {
            package_name: version(package_name)
            for package_name in ("guppyalgos", "guppylang", "hugr", "zixy")
        },
        "artifacts": records,
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(records)} artifacts and {MANIFEST_PATH.name}.")


def verify() -> None:
    """Verify every manifest checksum and parse every HUGR envelope."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    for record in manifest["artifacts"]:
        path = HERE / record["file"]
        _verify_package(path, record["sha256"])
        print(f"Verified {path.name}")


def main() -> None:
    """Generate the suite, or verify it when ``--verify-only`` is supplied."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Parse and checksum existing artifacts without regenerating them.",
    )
    args = parser.parse_args()
    if args.verify_only:
        verify()
    else:
        generate()


if __name__ == "__main__":
    main()
