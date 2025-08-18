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
        return []
