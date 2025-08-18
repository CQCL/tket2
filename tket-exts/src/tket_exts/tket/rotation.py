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
        return []

    def OPS(self) -> List[ExtOp]:
        """Return the operations defined by this extension"""
        return []
