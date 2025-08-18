"""QSystem extension operations."""

import functools
from typing import List

from .random import QSystemRandomExtension
from .utils import QSystemUtilsExtension

from hugr.ext import Extension
from hugr.ops import ExtOp
from hugr.tys import ExtType
from .._util import TketExtension, load_extension

__all__ = ["QSystemRandomExtension", "QSystemUtilsExtension", "QSystemExtension"]


class QSystemExtension(TketExtension):
    """QSystem extension operations."""

    @functools.cache
    def __call__(self) -> Extension:
        """Returns the qsystem extension"""
        return load_extension("tket.qsystem")

    def TYPES(self) -> List[ExtType]:
        """Return the types defined by this extension"""
        return []

    def OPS(self) -> List[ExtOp]:
        """Return the operations defined by this extension"""
        return []
