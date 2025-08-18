"""QSystem utils extension operations."""

import functools
from typing import List

from hugr.ext import Extension
from hugr.ops import ExtOp
from hugr.tys import ExtType
from .._util import TketExtension, load_extension


class QSystemUtilsExtension(TketExtension):
    """QSystem's utility operations."""

    @functools.cache
    def __call__(self) -> Extension:
        """Returns the qsystem utils extension"""
        return load_extension("tket.qsystem.utils")

    def TYPES(self) -> List[ExtType]:
        """Return the types defined by this extension"""
        return []

    def OPS(self) -> List[ExtOp]:
        """Return the operations defined by this extension"""
        return []
