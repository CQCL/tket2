"""Circuit matching and replacement system.

This module provides functionality for pattern matching and replacement
in quantum circuits, enabling circuit optimization through rewrite rules.
"""

from typing import Any, Union, TypedDict
from .circuit import Tk2Circuit

# Import the actual implementation from the Rust extension
from ._tket.matcher import (
    MatchReplaceRewriter,
    RotationMatcher,
    ReplaceWithIdentity,
    CircuitUnit,
)


class MatchContext(TypedDict):
    """
    Context of a partial match.
    """

    match_info: Any
    subcircuit: Tk2Circuit
    op_node: str


class MatchOutcome(TypedDict, total=False):
    """
    Outcome of a pattern match.
    """

    complete: Any
    proceed: Union[Any, bool]
    skip: Union[Any, bool]


# Re-export the main classes and functions
__all__ = [
    "MatchContext",
    "MatchOutcome",
    "MatchReplaceRewriter",
    "RotationMatcher",
    "ReplaceWithIdentity",
    "CircuitUnit",
]
