"""Quantum extension operations."""

import functools
from typing import List

from hugr.ext import Extension
from hugr.ops import ExtOp
from hugr.tys import ExtType
from ._util import TketExtension, load_extension


class QuantumExtension(TketExtension):
    """TKET's standard quantum operations."""

    @functools.cache
    def __call__(self) -> Extension:
        """Returns the quantum extension"""
        return load_extension("tket.quantum")

    def TYPES(self) -> List[ExtType]:
        """Return the types defined by this extension"""
        return []

    def OPS(self) -> List[ExtOp]:
        """Return the operations defined by this extension"""
        return []
