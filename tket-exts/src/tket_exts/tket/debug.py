"""Debug extension operations."""

import functools
from typing import List

from hugr.ext import Extension
from hugr.ops import ExtOp
from hugr.tys import ExtType
from ._util import TketExtension, load_extension


class DebugExtension(TketExtension):
    """Extension for debugging operations."""

    @functools.cache
    def __call__(self) -> Extension:
        """Return the debug extension"""
        return load_extension("tket.debug")

    def TYPES(self) -> List[ExtType]:
        """Return the types defined by this extension"""
        return []

    def OPS(self) -> List[ExtOp]:
        """Return the operations defined by this extension"""
        return [
            self.stateResult,
        ]

    @functools.cached_property
    def stateResult(self) -> ExtOp:
        """Report the state of given qubits in the given order."""
        return self().get_op("StateResult").instantiate()
