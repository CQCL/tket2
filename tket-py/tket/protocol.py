"""
Python protocols for circuit matching and replacement.

These protocols define the interfaces that Python classes must implement
to work with the circuit rewriting system.
"""

from typing import Protocol, List, Any
from typing_extensions import runtime_checkable

from .circuit import Tk2Circuit
from ._tket.ops import TketOp
from .matcher import MatchOutcome, MatchContext, CircuitUnit


@runtime_checkable
class CircuitMatcher(Protocol):
    """Protocol for circuit pattern matchers.

    Classes implementing this protocol can be used to define custom
    circuit matching logic that integrates with the Rust rewriting system.
    """

    def match_tket_op(
        self, op: TketOp, op_args: List[CircuitUnit], context: MatchContext
    ) -> MatchOutcome:
        """
        Match a TKET operation and return the match outcome.

        Args:
            op: The TKET operation to match
            op_args: Arguments to the operation as CircuitUnit instances
            context: Context of the current partial match containing:
                - match_info: Current partial match information
                - op_node: Node index of the current operation
                - subcircuit: The partial circuit matched so far

        Return:
            A dict object with any subset of the following keys:
            - "complete": If this is set, the pattern was fully matched and
              the match will be reported to the user (or optimiser), along with
              the value stored in the dict.
            - proceed: Either a boolean or an arbitrary (non-boolean) value. If
              this is set to anything other than `False`, the pattern was
              partially matched and the matching should continue.
              If the dict value is `True`, the match_data is left unchanged.
              Otherwise, the match_data is updated to the value in the dict.
            - skip: Either a boolean or an arbitrary (non-boolean) value.
              If set to anything other than `False`, the current operation
              should be skipped and matching should continue without it. If the
              dict value is `True`, the match_data is left unchanged. Otherwise,
              the match_data is updated to the value in the dict.
            - stop: If this is set to any value, or if no other key is set,
              the current match will be abandonned without reporting a match.
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
