from pathlib import Path

from .optimiser import BadgerOptimiser
from .circuit import Tk2Circuit
from pytket._tket.circuit import Circuit

class CircuitChunks:
    def reassemble(self) -> Circuit | Tk2Circuit:
        """Reassemble the circuit from its chunks."""

    def circuits(self) -> list[Circuit | Tk2Circuit]:
        """Returns clones of the split circuits."""

    def update_circuit(self, index: int, circ: Circuit | Tk2Circuit) -> None:
        """Replace a circuit chunk with a new version."""

class PullForwardError(Exception):
    """Error from a `PullForward` operation."""

def greedy_depth_reduce(circ: Circuit | Tk2Circuit) -> tuple[Circuit | Tk2Circuit, int]:
    """Greedy depth reduction of a circuit.

    Returns the reduced circuit and the depth reduction.
    """

def badger_optimise(
    circ: Circuit | Tk2Circuit,
    optimiser: BadgerOptimiser,
    max_threads: int | None = None,
    timeout: int | None = None,
    progress_timeout: int | None = None,
    log_dir: Path | None = None,
    rebase: bool = False,
) -> Circuit | Tk2Circuit:
    """Optimise a circuit using the Badger optimiser.

    HyperTKET's best attempt at optimising a circuit using circuit rewriting
    and the given Badger optimiser.

    By default, the input circuit will be rebased to Nam, i.e. CX + Rz + H before
    optimising. This can be deactivated by setting `rebase` to `false`, in which
    case the circuit is expected to be in the Nam gate set.

    Will use at most `max_threads` threads (plus a constant) and take at most
    `timeout` seconds (plus a constant). Default to the number of cpus and
    15min respectively.

    Log files will be written to the directory `log_dir` if specified.
    """

def chunks(c: Circuit | Tk2Circuit, max_chunk_size: int) -> CircuitChunks:
    """Split a circuit into chunks of at most `max_chunk_size` gates."""