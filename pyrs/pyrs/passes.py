from pathlib import Path

from pytket import Circuit
from pytket.passes import BasePass
from pyrs.pyrs import passes, optimiser


class TasoPass(BasePass):
    """A pytket pass that applies TASO optimisation to a circuit."""
    def __init__(
        self,
        rewriter_dir=None,
        rewriter_file=None,
        max_threads=None,
        timeout=None,
        log_dir=None,
        rebase=None,
    ):
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
        self.optimiser = optimiser.TasoOptimiser.load_precompiled(rewriter_file)
        self.max_threads = max_threads
        self.timeout = timeout
        self.log_dir = log_dir
        self.rebase = rebase
        super().__init__()

    def apply(self, circuit: Circuit) -> Circuit:
        """Apply TASO optimisation to the circuit."""
        return passes.taso_optimise(
            circuit,
            self.optimiser,
            self.max_threads,
            self.timeout,
            self.log_dir,
            self.rebase,
        )
