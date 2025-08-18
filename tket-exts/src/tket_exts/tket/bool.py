"""Opaque boolean operations."""

import functools
from typing import List

from hugr.ext import Extension, OpDef, TypeDef
from hugr.ops import ExtOp
from hugr.tys import ExtType
from ._util import TketExtension, load_extension


class BoolExtension(TketExtension):
    """Opaque boolean extension.

    This extension operates with an opaque `bool_t` type that
    is equivalent to Hugr's `[] + []` sum type representation.
    """

    @functools.cache
    def __call__(self) -> Extension:
        """Return the bool extension"""
        return load_extension("tket.bool")

    def TYPES(self) -> List[TypeDef]:
        """Return the types defined by this extension"""
        return [self.bool_t.type_def]

    def OPS(self) -> List[OpDef]:
        """Return the operations defined by this extension"""
        return [
            self.and_op.op_def(),
            self.eq.op_def(),
            self.make_opaque.op_def(),
            self.not_op.op_def(),
            self.or_op.op_def(),
            self.read.op_def(),
            self.xor.op_def(),
        ]

    @functools.cached_property
    def bool_t(self) -> ExtType:
        """An opaque boolean type"""
        return self().get_type("bool").instantiate([])

    @functools.cached_property
    def and_op(self) -> ExtOp:
        """Logical AND between two tket.bools"""
        return self().get_op("and").instantiate()

    @functools.cached_property
    def eq(self) -> ExtOp:
        """Equality between two tket.bools"""
        return self().get_op("eq").instantiate()

    @functools.cached_property
    def make_opaque(self) -> ExtOp:
        """Convert a Hugr `bool_t` (a unit sum) into an tket.bool."""
        return self().get_op("make_opaque").instantiate()

    @functools.cached_property
    def not_op(self) -> ExtOp:
        """Negation of a tket.bool"""
        return self().get_op("not").instantiate()

    @functools.cached_property
    def or_op(self) -> ExtOp:
        """Logical OR between two tket.bools"""
        return self().get_op("or").instantiate()

    @functools.cached_property
    def read(self) -> ExtOp:
        """Convert a tket.bool into a Hugr bool_t (a unit sum)"""
        return self().get_op("read").instantiate()

    @functools.cached_property
    def xor(self) -> ExtOp:
        """Logical XOR between two tket.bools"""
        return self().get_op("xor").instantiate()
