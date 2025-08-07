"""Type stubs for the matcher module."""

from typing import Any, List, Union, Dict
from ..protocol import CircuitMatcher, CircuitReplacer
from .circuit import Tk2Circuit
from .ops import TketOp
from .rewrite import CircuitRewrite
from ..matcher import OpArg

# Individual OpArg variant types
class QubitOpArg:
    """Qubit operation argument."""

    def __init__(self, index: int) -> None: ...
    @property
    def index(self) -> int: ...

class QubitOpBeforeArg:
    """Qubit operation argument positioned before operations."""

    def __init__(self, index: int) -> None: ...
    @property
    def index(self) -> int: ...

class QubitOpAfterArg:
    """Qubit operation argument positioned after operations."""

    def __init__(self, index: int) -> None: ...
    @property
    def index(self) -> int: ...

class ConstF64Arg:
    """Constant float argument."""

    def __init__(self, value: float) -> None: ...
    @property
    def value(self) -> float: ...

class MatchOutcome:
    """Match outcome for pattern matching."""

    def __init__(self) -> None:
        """Create a new empty match outcome."""
        ...

    @staticmethod
    def stop() -> MatchOutcome:
        """Create a match outcome that stops matching."""
        ...

    @staticmethod
    def skip(partial_match: Any) -> MatchOutcome:
        """Create a match outcome that skips the current operation."""
        ...

    @staticmethod
    def complete(match_info: Any) -> MatchOutcome:
        """Create a match outcome that completes a match."""
        ...

    @staticmethod
    def proceed(partial_match: Any) -> MatchOutcome:
        """Create a match outcome that proceeds with partial matching."""
        ...

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
        self, op: TketOp, op_args: List[OpArg], context: Dict[str, Any]
    ) -> MatchOutcome: ...

class ReplaceWithIdentity:
    """Replace any n-qubit circuit with the n-qubit identity circuit."""

    def __init__(self) -> None: ...
    def replace_match(self, circuit: Tk2Circuit, _match_info: Any) -> List[Tk2Circuit]:
        """Get the identity replacement for the match."""
        ...
