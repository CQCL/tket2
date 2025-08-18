"""Result extension operations."""

import functools
from typing import List

from hugr.ext import Extension
from hugr.ops import ExtOp
from hugr.tys import ExtType
from ._util import TketExtension, load_extension


class ResultExtension(TketExtension):
    """Result reporting operations."""

    @functools.cache
    def __call__(self) -> Extension:
        """Returns the result extension"""
        return load_extension("tket.result")

    def TYPES(self) -> List[ExtType]:
        """Return the types defined by this extension"""
        return []

    def OPS(self) -> List[ExtOp]:
        """Return the operations defined by this extension"""
        return [
            self.result_array_bool,
            self.result_array_f64,
            self.result_array_int,
            self.result_array_uint,
            self.result_bool,
            self.result_f64,
            self.result_int,
            self.result_uint,
        ]

    @functools.cached_property
    def result_array_bool(self) -> ExtOp:
        """Report an array of boolean results."""
        return self().get_op("result_array_bool").instantiate()

    @functools.cached_property
    def result_array_f64(self) -> ExtOp:
        """Report an array of floating-point results."""
        return self().get_op("result_array_f64").instantiate()

    @functools.cached_property
    def result_array_int(self) -> ExtOp:
        """Report an array of signed integer results."""
        return self().get_op("result_array_int").instantiate()

    @functools.cached_property
    def result_array_uint(self) -> ExtOp:
        """Report an array of unsigned integer results."""
        return self().get_op("result_array_uint").instantiate()

    @functools.cached_property
    def result_bool(self) -> ExtOp:
        """Report a boolean result."""
        return self().get_op("result_bool").instantiate()

    @functools.cached_property
    def result_f64(self) -> ExtOp:
        """Report a floating-point result."""
        return self().get_op("result_f64").instantiate()

    @functools.cached_property
    def result_int(self) -> ExtOp:
        """Report a signed integer result."""
        return self().get_op("result_int").instantiate()

    @functools.cached_property
    def result_uint(self) -> ExtOp:
        """Report an unsigned integer result."""
        return self().get_op("result_uint").instantiate()
