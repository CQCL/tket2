from pathlib import Path
from typing import Optional
from importlib import resources

from pytket import Circuit
from pytket.passes import CustomPass

from tket2 import optimiser

# Re-export native bindings
from .tket2._passes import *  # noqa: F403
from .tket2 import _passes

__all__ = [
    "badger_pass",
    *_passes.__all__,
]


def badger_pass(
    rewriter: Optional[Path] = None,
    max_threads: Optional[int] = None,
    timeout: Optional[int] = None,
    log_dir: Optional[Path] = None,
    rebase: Optional[bool] = None,
) -> CustomPass:
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
        return _passes.badger_optimise(
            circuit,
            opt,
            max_threads,
            timeout,
            log_dir,
            rebase,
        )

    return CustomPass(apply)
