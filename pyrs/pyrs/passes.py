from pathlib import Path

from pytket import Circuit
from pytket.passes import CustomPass
from pyrs.pyrs import passes, optimiser


def taso_pass(
    rewriter_dir=None,
    rewriter_file=None,
    max_threads=None,
    timeout=None,
    log_dir=None,
    rebase=None,
) -> CustomPass:
    """Construct a TASO pass.

    A Taso optimiser may be specified using either `rewriter_dir` or `rewriter_file`.
    If `rewriter_dir` is specified, the optimiser will be loaded from the file
    `rewriter_dir/nam_6_3.rwr`. By default, will search for a rewritier in
    the current directory. The rewriter must be precompiled.

    The arguments `max_threads`, `timeout`, `log_dir` and `rebase` are optional
    and will be passed on to the TASO optimiser if provided."""
    if rewriter_dir is None:
        rewriter_dir = "."
    if rewriter_file is None:
        rewriter_file = Path(rewriter_dir) / "nam_6_3.rwr"

    def apply(circuit: Circuit):
        """Apply TASO optimisation to the circuit."""
        return passes.taso_optimise(
            circuit,
            optimiser,
            max_threads,
            timeout,
            log_dir,
            rebase,
        )

    return CustomPass(apply)
