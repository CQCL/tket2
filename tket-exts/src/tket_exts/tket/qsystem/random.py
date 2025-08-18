"""QSystem random extension operations."""

import functools
from typing import List

from hugr.ext import Extension
from hugr.ops import ExtOp
from hugr.tys import ExtType
from .._util import TketExtension, load_extension


class QSystemRandomExtension(TketExtension):
    """QSystem's random operations."""

    @functools.cache
    def __call__(self) -> Extension:
        """Returns the qsystem random extension"""
        return load_extension("tket.qsystem.random")

    def TYPES(self) -> List[ExtType]:
        """Return the types defined by this extension"""
        return [self.context]

    def OPS(self) -> List[ExtOp]:
        """Return the operations defined by this extension"""
        return [
            self.deleteRNGContext,
            self.newRNGContext,
            self.randomFloat,
            self.randomInt,
            self.randomIntBounded,
        ]

    @functools.cached_property
    def context(self) -> ExtType:
        """Linear RNG context type."""
        return self().get_type("context").instantiate([])

    @functools.cached_property
    def deleteRNGContext(self) -> ExtOp:
        """Discard the given RNG context."""
        return self().get_op("DeleteRNGContext").instantiate()

    @functools.cached_property
    def newRNGContext(self) -> ExtOp:
        """Seed the RNG and return a new RNG context (call once)."""
        return self().get_op("NewRNGContext").instantiate()

    @functools.cached_property
    def randomFloat(self) -> ExtOp:
        """Generate a random float in [0, 1)."""
        return self().get_op("RandomFloat").instantiate()

    @functools.cached_property
    def randomInt(self) -> ExtOp:
        """Generate a random 32-bit unsigned integer."""
        return self().get_op("RandomInt").instantiate()

    @functools.cached_property
    def randomIntBounded(self) -> ExtOp:
        """Generate a random 32-bit unsigned integer less than bound."""
        return self().get_op("RandomIntBounded").instantiate()
