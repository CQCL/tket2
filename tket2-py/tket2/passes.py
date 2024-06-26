from pathlib import Path
from typing import Optional
from importlib import resources

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
    log_dir: Optional[Path] = None,
    rebase: bool = False,
) -> BasePass:
    """Construct a Badger pass.

    The Badger optimiser requires a pre-compiled rewriter produced by the
    `compile-rewriter <https://github.com/CQCL/tket2/tree/main/badger-optimiser>`_
    utility. If `rewriter` is not specified, a default one will be used.

    The arguments `max_threads`, `timeout`, `log_dir` and `rebase` are optional
    and will be passed on to the Badger optimiser if provided."""
    if rewriter is None:
        with resources.as_file(
            resources.files("tket2").joinpath("data/nam_6_3.rwr")
        ) as r:
            rewriter = Path(r)
    opt = optimiser.BadgerOptimiser.load_precompiled(rewriter)

    def apply(circuit: Circuit) -> Circuit:
        """Apply Badger optimisation to the circuit."""
        return badger_optimise(
            circuit,
            optimiser=opt,
            max_threads=max_threads,
            timeout=timeout,
            progress_timeout=progress_timeout,
            log_dir=log_dir,
            rebase=rebase,
        )

    return CustomPass(apply)
