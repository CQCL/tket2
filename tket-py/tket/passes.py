from pathlib import Path
from typing import Optional, Literal, Protocol
import json
from dataclasses import dataclass

from pytket import Circuit
from pytket.passes import (
    CustomPass,
    BasePass,
    CliffordSimp,
    KAKDecomposition,
    SquashRzPhasedX,
)
from pytket.circuit import OpType as PyTketOp

from tket import optimiser
from tket.circuit import Tk2Circuit

from hugr.passes._composable_pass import (
    ComposablePass,
    implement_pass_run,
    PassResult,
)


from hugr.hugr.base import Hugr

# Re-export native bindings.
from ._tket.passes import (
    CircuitChunks,
    greedy_depth_reduce,
    lower_to_pytket,
    badger_optimise,
    chunks,
    tket1_pass,
    normalize_guppy,
    PullForwardError,
)

__all__ = [
    "badger_pass",
    # Bindings.
    # TODO: Wrap these in Python classes.
    "CircuitChunks",
    "greedy_depth_reduce",
    "lower_to_pytket",
    "badger_optimise",
    "chunks",
    "clifford_simp",
    "two_qubit_squash",
    "squash_phasedx_rz",
    "normalize_guppy",
    "PullForwardError",
]


def badger_pass(
    rewriter: Optional[Path] = None,
    max_threads: Optional[int] = None,
    timeout: Optional[int] = None,
    progress_timeout: Optional[int] = None,
    max_circuit_count: Optional[int] = None,
    log_dir: Optional[Path] = None,
    rebase: bool = False,
    cost_fn: Literal["cx", "rz"] | None = None,
) -> BasePass:
    """Construct a Badger pass.

    The Badger optimiser requires a pre-compiled rewriter produced by the
    `compile-rewriter <https://github.com/CQCL/tket2/tree/main/badger-optimiser>`_
    utility. If `rewriter` is not specified, a default one will be used.

    The cost function to minimise can be specified by passing `cost_fn` as `'cx'`
    or `'rz'`. If not specified, the default is `'cx'`.

    The arguments `max_threads`, `timeout`, `progress_timeout`, `max_circuit_count`,
    `log_dir` and `rebase` are optional and will be passed on to the Badger
    optimiser if provided."""
    if rewriter is None:
        try:
            import tket_eccs
        except ImportError:
            raise ValueError(
                "The default rewriter is not available. Please specify a path to a rewriter or install tket-eccs."
            )

        rewriter = tket_eccs.nam_6_3()
    opt = optimiser.BadgerOptimiser.load_precompiled(rewriter, cost_fn=cost_fn)

    def apply(circuit: Circuit) -> Circuit:
        """Apply Badger optimisation to the circuit."""
        return badger_optimise(
            circuit,
            optimiser=opt,
            max_threads=max_threads,
            timeout=timeout,
            progress_timeout=progress_timeout,
            max_circuit_count=max_circuit_count,
            log_dir=log_dir,
            rebase=rebase,
        )

    return CustomPass(apply, label="tket.badger_pass")


class Tket1Pass(ComposablePass, Protocol):
    def _pytket_pass(self) -> BasePass: ...

    def run(self, hugr: Hugr, *, inplace: bool = False) -> PassResult:
        return implement_pass_run(
            self, hugr=hugr, inplace=False, inplace_call=self._run_pytket_pass_on_hugr
        )

    def _run_pytket_pass_on_hugr(self, hugr: Hugr) -> PassResult:
        pass_json = json.dumps(self._pytket_pass().to_dict())
        compiler_state: Tk2Circuit = Tk2Circuit.from_bytes(hugr.to_bytes())
        opt_program = tket1_pass(compiler_state, pass_json, traverse_subcircuits=True)
        hugr = Hugr.from_str(opt_program.to_str())
        return PassResult(hugr=hugr, inplace=False, results=[(self.name, hugr)])


@dataclass
class CliffordSimplification(Tket1Pass):
    allow_swaps: bool = True
    target_2qb_gate: PyTketOp = PyTketOp.CX

    def _pytket_pass(self) -> BasePass:
        return CliffordSimp(self.allow_swaps, self.target_2qb_gate)


def clifford_simp(
    circ: Circuit,
    *,
    allow_swaps: bool = True,
    target_2qb_gate: str = "CX",
) -> Circuit:
    """An optimisation pass that applies a number of rewrite rules for simplifying Clifford gate sequences, similar to Duncan & Fagan (https://arxiv.org/abs/1901.10114).

    Produces a circuit comprising TK1 gates and the two-qubit gate specified as the target.

    Parameters:
    :param allow_swaps: Whether the rewriting may introduce implicit wire swaps
    :param target_2qb_gate: Target two-qubit gate (either CX or TK2)
    """
    match target_2qb_gate:
        case "CX":
            gate = PyTketOp.CX
        case "TK2":
            gate = PyTketOp.TK2
        case _:
            raise ValueError(f"Invalid target two-qubit gate: {target_2qb_gate}")

    pass_json = json.dumps(
        CliffordSimp(allow_swaps=allow_swaps, target_2qb_gate=gate).to_dict()
    )
    return tket1_pass(circ, pass_json, traverse_subcircuits=True)


def two_qubit_squash(
    circ: Circuit,
    *,
    allow_swaps: bool = True,
    target_2qb_gate: str = "CX",
    cx_fidelity: float = 1.0,
) -> Circuit:
    """
    Squash sequences of two-qubit operations into minimal form.

    This pass squashes together sequences of single- and two-qubit gates into their minimal form.
    The sequence may be decomposed to either TK2 or CX gates.

    - Two-qubit operations can always be expressed in a minimal form using at most three CXs, or as a single TK2 gate (also known as the KAK or Cartan decomposition).
    - It is generally recommended to squash to TK2 gates, and then use the DecomposeTK2 pass for noise-aware decomposition to other gate sets.
    - For backward compatibility, decompositions to CX are also supported. In this case, `cx_fidelity` can be provided to perform approximate decompositions to CX gates.
    - When decomposing to TK2 gates, *any* sequence of two or more two-qubit gates on the same set of qubits is replaced by a single TK2 gate.
    - When decomposing to CX, the substitution is performed only if it results in a reduction in the number of CX gates, or if at least one of the two-qubit gates is not a CX.
    - With `allow_swaps=True` (default), qubits may be swapped when convenient to further reduce the two-qubit gate count (only applicable when decomposing to CX gates).
    - Gates containing symbolic parameters are not squashed.

    Parameters:
    :param allow_swaps: Whether to allow implicit wire swaps
    :param target_2qb_gate: Target two-qubit gate (either CX or TK2)
    :param cx_fidelity: Estimated CX gate fidelity, used when `target_2qb_gate` is CX
    """
    match target_2qb_gate:
        case "CX":
            gate = PyTketOp.CX
        case "TK2":
            gate = PyTketOp.TK2
        case _:
            raise ValueError(f"Invalid target two-qubit gate: {target_2qb_gate}")

    pass_json = json.dumps(
        KAKDecomposition(
            allow_swaps=allow_swaps, target_2qb_gate=gate, cx_fidelity=cx_fidelity
        ).to_dict()
    )
    return tket1_pass(circ, pass_json, traverse_subcircuits=True)


def squash_phasedx_rz(
    circ: Circuit,
) -> Circuit:
    """Squash single qubit gates into PhasedX and Rz gates. Also remove identity gates.

    Commute Rz gates to the back if possible.
    """
    pass_json = json.dumps(SquashRzPhasedX().to_dict())
    return tket1_pass(circ, pass_json, traverse_subcircuits=True)
