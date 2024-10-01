from pathlib import Path
from typing import Optional, Literal

from pytket import Circuit
from pytket.passes import CustomPass, BasePass

from tket2 import optimiser

# Re-export native bindings
from ._tket2.passes import (
    CircuitChunks,
    greedy_depth_reduce,
    lower_to_pytket,
    badger_optimise,
    chunks,
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
            import tket2_eccs
        except ImportError:
            raise ValueError(
                "The default rewriter is not available. Please specify a path to a rewriter or install tket2-eccs."
            )

        rewriter = tket2_eccs.nam_6_3()
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

    return CustomPass(apply)
