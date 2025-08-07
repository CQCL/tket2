"""
Python protocols for circuit matching and replacement.

These protocols define the interfaces that Python classes must implement
to work with the circuit rewriting system.
"""

from typing import Protocol, List, Any, Dict
from typing_extensions import runtime_checkable

from .circuit import Tk2Circuit
from ._tket.ops import TketOp
from .matcher import OpArg, MatchOutcome


@runtime_checkable
class CircuitMatcher(Protocol):
    """Protocol for circuit pattern matchers.

    Classes implementing this protocol can be used to define custom
    circuit matching logic that integrates with the Rust rewriting system.
    """

    def match_tket_op(
        self, op: TketOp, op_args: List[OpArg], context: Dict[str, Any]
    ) -> MatchOutcome:
        """
        Match a TKET operation and return the match outcome.

        Args:
            op: The TKET operation to match
            op_args: Arguments to the operation (OpArg instances)
            context: Context of the current partial match containing:
                - match_info: Current partial match information
                - op_node: Node index of the current operation

        Returns:
            A MatchOutcome object indicating whether to:
            - complete: Pattern was fully matched
            - proceed: Pattern was partially matched and should continue
            - skip: Current operation should be skipped
            - stop: Stop matching (no actions)
        """
        ...


@runtime_checkable
class CircuitReplacer(Protocol):
    """Protocol for match replacements.

    Classes implementing this protocol can generate replacement circuits
    for pattern matches found by CircuitMatcher implementations.
    """

    def replace_match(self, circuit: Tk2Circuit, match_info: Any) -> List[Tk2Circuit]:
        """
        Generate replacement circuits for a pattern match.

        Args:
            circuit: Tk2Circuit representing the matched subgraph
            match_info: Information about the match returned by the matcher

        Returns:
            List of possible replacement circuits
        """
        ...
