from typing import TypeVar, Literal
from .circuit import Tk2Circuit
from pytket._tket.circuit import Circuit

from pathlib import Path

CircuitClass = TypeVar("CircuitClass", Circuit, Tk2Circuit)

class BadgerOptimiser:
    @staticmethod
    def load_precompiled(
        filename: Path, cost_fn: Literal["cx", "rz"] | None = None
    ) -> BadgerOptimiser:
        """
        Load a precompiled rewriter from a file.

        :param filename: The path to the file containing the precompiled rewriter.
        :param cost_fn: The cost function to use.
        """

    @staticmethod
    def compile_eccs(
        filename: Path, cost_fn: Literal["cx", "rz"] | None = None
    ) -> BadgerOptimiser:
        """
        Compile a set of ECCs and create a new rewriter.

        :param filename: The path to the file containing the ECCs.
        :param cost_fn: The cost function to use.
        """

    def optimise(
        self,
        circ: CircuitClass,
        timeout: int | None = None,
        progress_timeout: int | None = None,
        n_threads: int | None = None,
        split_circ: bool = False,
        queue_size: int | None = None,
        log_progress: Path | None = None,
    ) -> CircuitClass:
        """Optimise a circuit.

        :param circ: The circuit to optimise.
        :param timeout: Maximum time to spend on the optimisation.
        :param progress_timeout: Maximum time to wait between new best results.
        :param n_threads: Number of threads to use.
        :param split_circ: Split the circuit into subcircuits and optimise them separately.
        :param queue_size: Maximum number of circuits to keep in the queue of candidates.
        :param log_progress: Log progress to a CSV file.
        """
