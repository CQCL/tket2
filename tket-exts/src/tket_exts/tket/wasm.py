"""WASM extension operations."""

import functools
from typing import List

from hugr.ext import Extension
from hugr.ops import ExtOp
from hugr.tys import ExtType
from ._util import TketExtension, load_extension


class WasmExtension(TketExtension):
    """WASM interop operations."""

    @functools.cache
    def __call__(self) -> Extension:
        """Returns the wasm extension"""
        return load_extension("tket.wasm")

    def TYPES(self) -> List[ExtType]:
        """Return the types defined by this extension"""
        return [self.context, self.func, self.module]

    def OPS(self) -> List[ExtOp]:
        """Return the operations defined by this extension"""
        return [
            self.call,
            self.dispose_context,
            self.get_context,
            self.lookup,
        ]

    @functools.cached_property
    def context(self) -> ExtType:
        """WASM context."""
        return self().get_type("context").instantiate([])

    @functools.cached_property
    def func(self) -> ExtType:
        """WASM function type signature."""
        return self().get_type("func").instantiate([])

    @functools.cached_property
    def module(self) -> ExtType:
        """WASM module."""
        return self().get_type("module").instantiate([])

    @functools.cached_property
    def call(self) -> ExtOp:
        """Call a function in a context, returning a Future of the result."""
        return self().get_op("call").instantiate()

    @functools.cached_property
    def dispose_context(self) -> ExtOp:
        """Dispose a WASM context."""
        return self().get_op("dispose_context").instantiate()

    @functools.cached_property
    def get_context(self) -> ExtOp:
        """Retrieve a context by handle."""
        return self().get_op("get_context").instantiate()

    @functools.cached_property
    def lookup(self) -> ExtOp:
        """Lookup a function in a module by name and signature."""
        return self().get_op("lookup").instantiate()
