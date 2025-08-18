"""Futures extension operations."""

import functools
from typing import List

from hugr.ext import Extension
from hugr.ops import ExtOp
from hugr.tys import ExtType
from ._util import TketExtension, load_extension


class FuturesExtension(TketExtension):
    """Future type and handling operations."""

    @functools.cache
    def __call__(self) -> Extension:
        """Returns the futures extension"""
        return load_extension("tket.futures")

    def TYPES(self) -> List[ExtType]:
        """Return the types defined by this extension"""
        return [self.future_t]

    def OPS(self) -> List[ExtOp]:
        """Return the operations defined by this extension"""
        return [
            self.dup,
            self.free,
            self.read,
        ]

    @functools.cached_property
    def future_t(self) -> ExtType:
        """A value that is computed asynchronously."""
        return self().get_type("Future").instantiate([])

    @functools.cached_property
    def dup(self) -> ExtOp:
        """Duplicate a Future. The original is consumed and two Futures are returned."""
        return self().get_op("Dup").instantiate()

    @functools.cached_property
    def free(self) -> ExtOp:
        """Consume a Future without reading it."""
        return self().get_op("Free").instantiate()

    @functools.cached_property
    def read(self) -> ExtOp:
        """Read a value from a Future, consuming it."""
        return self().get_op("Read").instantiate()
