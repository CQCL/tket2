"""Global Phase extension operations."""

import functools
from typing import List

from hugr.ext import Extension, OpDef, TypeDef
from ._util import TketExtension, load_extension


class GlobalPhaseExtension(TketExtension):
    """TKET's standard quantum operations."""

    @functools.cache
    def __call__(self) -> Extension:
        """Returns the quantum extension"""
        return load_extension("tket.global_phase")

    def TYPES(self) -> List[TypeDef]:
        """Return the types defined by this extension"""
        return []

    def OPS(self) -> List[OpDef]:
        """Return the operations defined by this extension"""
        # todo
        return []
