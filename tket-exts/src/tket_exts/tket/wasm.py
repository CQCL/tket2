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
        return []

    def OPS(self) -> List[ExtOp]:
        """Return the operations defined by this extension"""
        return []
