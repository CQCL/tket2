from pathlib import Path
from typing import Optional, Literal
import json
from dataclasses import dataclass

from pytket import Circuit
from pytket.passes import (
    CustomPass,
    BasePass,
)

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


@dataclass
class PytketPass(ComposablePass):
    pytket_pass: BasePass

    def __init__(self, pytket_pass: BasePass) -> None:
        self.pytket_pass = pytket_pass

    def __call__(self, hugr: Hugr, *, inplace: bool = False) -> Hugr:
        """Call the pass to transform a HUGR, returning a Hugr."""
        return self.run(hugr, inplace=inplace).hugr

    def run(self, hugr: Hugr, *, inplace: bool = False) -> PassResult:
        return implement_pass_run(
            self,
            hugr=hugr,
            inplace=inplace,
            copy_call=lambda h: self._run_pytket_pass_on_hugr(h),
        )

    def _run_pytket_pass_on_hugr(self, hugr: Hugr) -> PassResult:
        pass_json = json.dumps(self.pytket_pass.to_dict())
        compiler_state: Tk2Circuit = Tk2Circuit.from_bytes(hugr.to_bytes())
        opt_program = tket1_pass(compiler_state, pass_json, traverse_subcircuits=True)
        new_hugr = Hugr.from_str(opt_program.to_str())
        return PassResult(hugr=new_hugr, inplace=False, results=[(self.name, new_hugr)])
