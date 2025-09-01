"""Type stubs for the matcher module."""

from typing import Any, List, Optional, Tuple, Literal
from ..protocol import CircuitMatcher, CircuitReplacer
from .circuit import Tk2Circuit
from .ops import TketOp
from .rewrite import CircuitRewrite
from ..matcher import MatchContext, MatchOutcome

class MatchReplaceRewriter:
    """Rewriter that uses pattern matching and replacement to generate circuit rewrites."""

    def __init__(
        self,
        matcher: CircuitMatcher,
        replacement: CircuitReplacer,
    ) -> None:
        """
        Create a new rewriter.

        Args:
            matcher: An object implementing CircuitMatcher protocol
            replacement: An object implementing CircuitReplacer protocol
        """
        ...

    def get_rewrites(self, circuit: Tk2Circuit) -> List[CircuitRewrite]:
        """Get all possible rewrites for a circuit."""
        ...

class CombineMatchReplaceRewriter:
    """
    A rewriter that combines multiple `CircuitMatcher`s before passing the
    combined match to a `CircuitReplacer`.

    The [`CircuitMatcher`]s are used to find matches in the circuit. All
    cartesian products of the matches that are convex are then passed to the
    [`CircuitReplacer`] to create [`CircuitRewrite`]s.
    """

    def __init__(
        self,
        matchers: List[CircuitMatcher],
        replacement: CircuitReplacer,
    ) -> None:
        """
        Create a new combine rewriter.

        Args:
            matchers: A list of objects implementing CircuitMatcher protocol
            replacement: An object implementing CircuitReplacer protocol
        """
        ...

    def get_rewrites(self, circuit: Tk2Circuit) -> List[CircuitRewrite]:
        """Get all possible rewrites for a circuit."""
        ...

def hadamard_cnot_rewriter() -> MatchReplaceRewriter:
    """Create a rewriter for Hadamard-CNOT pattern optimization."""
    ...

def rotation_rewriter() -> MatchReplaceRewriter:
    """Create a rewriter for rotation gate optimization."""
    ...

class RotationMatcher:
    """Dummy placeholder matcher for rotation gate optimization."""

    def __init__(self) -> None: ...
    def match_tket_op(
        self, op: TketOp, op_args: List[CircuitUnit], context: MatchContext
    ) -> MatchOutcome: ...

class ReplaceWithIdentity:
    """Replace any n-qubit circuit with the n-qubit identity circuit."""

    def __init__(self) -> None: ...
    def replace_match(self, circuit: Tk2Circuit, match_info: Any) -> List[Tk2Circuit]:
        """Get the identity replacement for the match."""
        ...

class CircuitUnit:
    linear_index: Optional[int]
    linear_pos: Optional[Literal["before", "after"]]
    copyable_wire: Optional[Tuple[str, int]]
    constant_float: Optional[float]
