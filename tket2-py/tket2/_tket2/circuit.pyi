from typing import Any, Callable
from pytket._tket.circuit import Circuit as Tk1Circuit

from tket2._tket2.ops import Tk2Op

class Tk2Circuit:
    """Rust representation of a TKET2 circuit."""

    def __init__(self, circ: Tk1Circuit) -> None:
        """Create a Tk2Circuit from a pytket Circuit."""

    def __hash__(self) -> int:
        """Compute the circuit hash by traversal."""

    def __copy__(self) -> Tk2Circuit:
        """Create a copy of the circuit."""

    def __deepcopy__(self) -> Tk2Circuit:
        """Create a deep copy of the circuit."""

    def hash(self) -> int:
        """Compute the circuit hash by traversal."""

    def circuit_cost(self, cost_fn: Callable[[Tk2Op], Any]) -> int:
        """Compute the cost of the circuit. Return value must implement __add__."""

    def num_operations(self) -> int:
        """The number of operations in the circuit.

        This includes [`Tk2Op`]s, pytket ops, and any other custom operations.

        Nested circuits are traversed to count their operations.
        """

    def node_op(self, node: Node) -> bytes:
        """If the node corresponds to a custom op, return it. Otherwise, raise an error."""

    def to_tket1(self) -> Tk1Circuit:
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

    def to_hugr_json(self) -> str:
        """Encode the circuit as a HUGR json."""

    def to_package_json(self) -> str:
        """Encode the circuit as a HUGR Package json."""

    @staticmethod
    def from_hugr_json(json: str) -> Tk2Circuit:
        """Decode a HUGR json string to a Tk2Circuit."""

    @staticmethod
    def from_package_json(json: str, function_name: str | None = None) -> Tk2Circuit:
        """Decode a HUGR Package json to a circuit.

        Traverses the package's modules in order until it finds one containing a
        function named `function_name`, and loads it as a circuit.

        If the json is a hugr json, it will be decoded as a `main` function in an empty module.

        When `function_name` is not given, it defaults to `main`.
        """

    def to_tket1_json(
        self,
    ) -> str:
        """Encode the circuit as a pytket json string."""

    @staticmethod
    def from_tket1_json(json: str) -> Tk2Circuit:
        """Decode a pytket json string to a Tk2Circuit."""

class Node:
    """Handle to node in HUGR."""

    def __init__(self, idx: int) -> None:
        """Create a new node handle."""

class Wire:
    """An outgoing edge from a node in a HUGR, defined by the node and outgoing port."""

    def node(self) -> Node:
        """Source node of wire."""

    def port(self) -> int:
        """Source port of wire."""

class CircuitCost:
    """A cost function for circuits."""

    def __init__(self, cost: Any) -> None:
        """Create a new circuit cost.

        The cost object must implement __add__, __sub__, __eq__, and __lt__."""

def render_circuit_dot(hugr: Tk2Circuit | Tk1Circuit) -> str: ...
def render_circuit_mermaid(hugr: Tk2Circuit | Tk1Circuit) -> str: ...
def validate_circuit(hugr: Tk2Circuit | Tk1Circuit) -> None: ...

class HugrError(Exception): ...
class BuildError(Exception): ...
class ValidationError(Exception): ...
class HUGRSerializationError(Exception): ...
class TK1ConvertError(Exception): ...
