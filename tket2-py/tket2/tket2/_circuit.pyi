from enum import Enum
from typing import Any, Callable
from pytket._tket.circuit import Circuit

class Tk2Circuit:
    """Rust representation of a TKET2 circuit."""

    def __init__(self, circ: Circuit) -> None:
        """Create a Tk2Circuit from a pytket Circuit."""
    def __hash__(self) -> int:
        """Compute the circuit hash by traversal."""
    def circuit_cost(self, cost_fn: Callable[[Tk2Op], Any]) -> int:
        """Compute the cost of the circuit. Return value must implement __add__."""
    def node_op(self, node: Node) -> CustomOp:
        """If the node corresponds to a custom op, return it. Otherwise, raise an error."""
    def to_tket1(self) -> Circuit:
        """Convert to pytket Circuit."""
    def apply_rewrite(self, rw) -> None:
        """Apply a rewrite to the circuit."""
    def node_inputs(self, node: Node) -> list[Wire]:
        """The incoming wires to a node."""
    def node_outputs(self, node: Node) -> list[Wire]:
        """The outgoing wires from a node."""
    def input_node(self) -> Node:
        """The input node of the circuit."""
    def output_node(self) -> Node:
        """The output node of the circuit."""

class Tk2Op(Enum):
    """A Tket2 built-in operation."""

    H = 1
    CX = 2

class TypeBound(Enum):
    """HUGR type bounds."""

    Any = 0  # Any type
    Copyable = 1  # Copyable type
    Eq = 2  # Equality-comparable type

class HugrType:
    """Value types in HUGR."""

    def __init__(self, extension: str, type_name: str, bound: TypeBound) -> None:
        """Create a new named Custom type."""
    @staticmethod
    def qubit() -> HugrType:
        """Qubit type from HUGR prelude."""
    @staticmethod
    def linear_bit() -> HugrType:
        """Linear bit type from TKET1 extension."""
    @staticmethod
    def bool() -> HugrType:
        """Boolean type (HUGR 2-ary unit sum)."""

class Node:
    """Handle to node in HUGR."""
    def outs(self, n: int) -> list[Wire]:
        """Generate n output wires from this node."""
    def __getitem__(self, i: int) -> Wire:
        """Get the i-th output wire from this node."""

class Wire:
    """An outgoing edge from a node in a HUGR, defined by the node and outgoing port."""
    def node(self) -> Node:
        """Source node of wire."""

class CustomOp:
    """A HUGR custom operation."""
    def __init__(
        self,
        extension: str,
        op_name: str,
        input_types: list[HugrType],
        output_types: list[HugrType],
    ) -> None:
        """Create a new custom operation from name and input/output types."""

    def to_custom(self) -> CustomOp:
        """Convert to a custom operation. Identity operation."""
    def name(self) -> str:
        """Fully qualified (include extension) name of the operation."""

class Dfg:
    """A builder for a HUGR dataflow graph."""
    def __init__(
        self,
        input_types: list[HugrType],
        output_types: list[HugrType],
    ) -> None:
        """Begin building a dataflow graph with specified input and output types."""
    def inputs(self) -> list[Wire]:
        """The output wires of the input node in the DFG, one for each input type."""
    def add_op(self, op: CustomOp, wires: list[Wire]) -> Node:
        """Add a custom operation to the DFG, wiring in input wires."""
    def finish(self, outputs: list[Wire]) -> Tk2Circuit:
        """Finish building the DFG by wiring in output wires to the output node
        (one per output type) and return the resulting circuit."""

def to_hugr_dot(hugr: Tk2Circuit | Circuit) -> str: ...
def to_hugr_mermaid(hugr: Tk2Circuit | Circuit) -> str: ...
def validate_hugr(hugr: Tk2Circuit | Circuit) -> None: ...
