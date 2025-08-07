"""Circuit matching and replacement system.

This module provides functionality for pattern matching and replacement
in quantum circuits, enabling circuit optimization through rewrite rules.
"""

from typing import Union

# Import the actual implementation from the Rust extension
from ._tket.matcher import (
    QubitOpArg,
    QubitOpBeforeArg,
    QubitOpAfterArg,
    ConstF64Arg,
    MatchOutcome,
    MatchReplaceRewriter,
    RotationMatcher,
    ReplaceWithIdentity,
)

# Type alias for any OpArg variant
OpArg = Union[QubitOpArg, QubitOpBeforeArg, QubitOpAfterArg, ConstF64Arg]

# Re-export the main classes and functions
__all__ = [
    "OpArg",
    "QubitOpArg",
    "QubitOpBeforeArg",
    "QubitOpAfterArg",
    "ConstF64Arg",
    "MatchOutcome",
    "MatchReplaceRewriter",
    "RotationMatcher",
    "ReplaceWithIdentity",
]

# Define the type alias for any OpArg variant
# This is the runtime definition that corresponds to the stub file
OpArgVariant = Union[QubitOpArg, QubitOpBeforeArg, QubitOpAfterArg, ConstF64Arg]
