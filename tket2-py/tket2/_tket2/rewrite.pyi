from pathlib import Path
from tket2._tket2.circuit import Node, Tk2Circuit

class ECCRewriter:
    @staticmethod
    def load_precompiled(filename: Path) -> ECCRewriter:
        """Load a precompiled rewriter from a file."""

    def get_rewrites(self, circ: Tk2Circuit) -> list[CircuitRewrite]:
        """Get rewrites for a circuit."""

class CircuitRewrite:
    """A rewrite rule for circuits."""

    def __init__(
        self,
        source_position: Subcircuit,
        source_circ: Tk2Circuit,
        replacement: Tk2Circuit,
    ) -> None:
        """Create a new circuit rewrite rule."""

    def node_count_delta(self) -> int:
        """The change in node count from the rewrite.

        A positive value indicates an increase in node count.
        """

    def replacement(self) -> Tk2Circuit:
        """The replacement circuit."""

class Subcircuit:
    """A subcircuit of a circuit."""

    def __init__(self, nodes: list[Node], circ: Tk2Circuit) -> None:
        """Create a new subcircuit."""
