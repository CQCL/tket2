"""Rotation extension operations."""

import functools
from typing import List

from hugr.ext import Extension
from hugr.ops import ExtOp
from hugr.tys import ExtType
from ._util import TketExtension, load_extension


class RotationExtension(TketExtension):
    """Rotation type for TKET's quantum operations."""

    @functools.cache
    def __call__(self) -> Extension:
        """Returns the rotation extension"""
        return load_extension("tket.rotation")

    def TYPES(self) -> List[ExtType]:
        """Return the types defined by this extension"""
        return [self.rotation]

    def OPS(self) -> List[ExtOp]:
        """Return the operations defined by this extension"""
        return [
            self.from_halfturns,
            self.from_halfturns_unchecked,
            self.radd,
            self.to_halfturns,
        ]

    @functools.cached_property
    def rotation(self) -> ExtType:
        """Rotation type expressed as number of half-turns."""
        return self().get_type("rotation").instantiate([])

    @functools.cached_property
    def from_halfturns(self) -> ExtOp:
        """Construct rotation from number of half-turns; returns None if non-finite."""
        return self().get_op("from_halfturns").instantiate()

    @functools.cached_property
    def from_halfturns_unchecked(self) -> ExtOp:
        """Construct rotation from number of half-turns; panics if non-finite."""
        return self().get_op("from_halfturns_unchecked").instantiate()

    @functools.cached_property
    def radd(self) -> ExtOp:
        """Add two angles together."""
        return self().get_op("radd").instantiate()

    @functools.cached_property
    def to_halfturns(self) -> ExtOp:
        """Convert rotation to a number of half-turns."""
        return self().get_op("to_halfturns").instantiate()
