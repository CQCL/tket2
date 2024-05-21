from typing import Iterator, Optional
from .circuit import Node, Tk2Circuit
from .rewrite import CircuitRewrite
from pytket._tket.circuit import Circuit

class Rule:
    """A rewrite rule defined by a left hand side and right hand side of an equation."""

    def __init__(
        self,
        l: Circuit | Tk2Circuit,  # noqa: E741
        r: Circuit | Tk2Circuit,
    ) -> None:
        """Create a new rewrite rule."""

class RuleMatcher:
    """A matcher for multiple rewrite rule."""

    def __init__(self, rules: list[Rule]) -> None:
        """Create a new rule matcher."""

    def find_matches(self, circ: Tk2Circuit) -> Optional[CircuitRewrite]:
        """Find a match of the rules in the circuit."""

class CircuitPattern:
    """A pattern that matches a circuit exactly."""

    def __init__(self, circ: Circuit | Tk2Circuit) -> None:
        """Create a new circuit pattern."""

class PatternMatcher:
    """A matcher object for fast pattern matching on circuits."""

    def __init__(self, patterns: Iterator[CircuitPattern]) -> None:
        """Create a new pattern matcher."""

    def find_matches(self, circ: Circuit | Tk2Circuit) -> list[PatternMatch]:
        """Find a match of the patterns in the circuit."""

class PatternMatch:
    """A convex pattern match in a circuit"""

    def pattern_id(self) -> PatternID:
        """The id of the matched pattern."""

    def root(self) -> Node:
        """The root node for the pattern in the matched circuit."""

class PatternID:
    """An identifier for a pattern in a pattern matcher."""

    def __int__(self) -> int:
        """Get the integer value of the pattern id."""

class InvalidPatternError(Exception):
    """Conversion error between a pattern and a circuit."""

class InvalidReplacementError(Exception):
    """An error occurred while constructing a pattern match replacement."""
