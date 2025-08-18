"""Guppy extension operations."""

import functools
from typing import List

from hugr.ext import Extension
from hugr.ops import ExtOp
from hugr.tys import ExtType
from ._util import TketExtension, load_extension


class GuppyExtension(TketExtension):
    """Guppy-specific operations."""

    @functools.cache
    def __call__(self) -> Extension:
        """Returns the guppy extension"""
        return load_extension("tket.guppy")

    def TYPES(self) -> List[ExtType]:
        """Return the types defined by this extension"""
        return []

    def OPS(self) -> List[ExtOp]:
        """Return the operations defined by this extension"""
        return [
            self.drop,
        ]

    @functools.cached_property
    def drop(self) -> ExtOp:
        """Drop the input wire. Applicable to guppy affine types only."""
        return self().get_op("drop").instantiate()
