from enum import Enum
from typing import Any, Iterable

class Tk2Op(Enum):
    """A rust-backed Tket2 built-in operation."""

    def __init__(self, op: str) -> None:
        """Create a new Tk2Op from a string name."""

    @staticmethod
    def values() -> Iterable[Tk2Op]:
        """Iterate over all operation variants."""

    @property
    def name(self) -> str:
        """Get the string name of the operation."""

    @property
    def qualified_name(self) -> str:
        """Return the fully qualified name of the operation, including the extension."""

    def __str__(self) -> str:
        """Get the string name of the operation."""

    def __eq__(self, value: object) -> bool: ...

class Pauli(Enum):
    """Simple enum representation of Pauli matrices."""

    def __init__(self, pauli: str) -> None:
        """Create a new Pauli from a string name."""

    @property
    def name(self) -> str:
        """Get the string name of the Pauli."""

    @staticmethod
    def values() -> Iterable[Pauli]:
        """Iterate over all Pauli matrix variants."""

    @staticmethod
    def I() -> Pauli:  # noqa: E743
        """Identity Pauli matrix."""

    @staticmethod
    def X() -> Pauli:
        """Pauli-X matrix."""

    @staticmethod
    def Y() -> Pauli:
        """Pauli-Y matrix."""

    @staticmethod
    def Z() -> Pauli:
        """Pauli-Z matrix."""

    def __str__(self) -> str:
        """Get the string name of the Pauli."""

    def __eq__(self, value: Any) -> bool: ...
