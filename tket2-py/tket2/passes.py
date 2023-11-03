from pathlib import Path
from typing import Optional
import importlib

from pytket import Circuit
from pytket.passes import CustomPass
from tket2 import passes, optimiser


def taso_pass(
    rewriter: Optional[Path] = None,
    max_threads: Optional[int] = None,
    timeout: Optional[int] = None,
    log_dir: Optional[Path] = None,
    rebase: Optional[bool] = None,
) -> CustomPass:
    """Construct a TASO pass.

    The Taso optimiser requires a pre-compiled rewriter produced by the
    `compile-rewriter <https://github.com/CQCL/tket2/tree/main/taso-optimiser>`_
    utility. If `rewriter` is not specified, a default one will be used.

    The arguments `max_threads`, `timeout`, `log_dir` and `rebase` are optional
    and will be passed on to the TASO optimiser if provided."""
    if rewriter is None:
        rewriter = Path(importlib.resources.files("tket2").joinpath("data/nam_6_3.rwr"))
    opt = optimiser.TasoOptimiser.load_precompiled(rewriter)

    def apply(circuit: Circuit) -> Circuit:
        """Apply TASO optimisation to the circuit."""
        return passes.taso_optimise(
            circuit,
            opt,
            max_threads,
            timeout,
            log_dir,
            rebase,
        )

    return CustomPass(apply)
