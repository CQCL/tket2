"""Utility functions for tket extensions."""

import pkgutil
from typing import List, Protocol

from hugr.ext import Extension
from hugr.ops import ExtOp
from hugr.tys import ExtType


def load_extension(name: str) -> Extension:
    import tket_exts

    replacement = name.replace(".", "/")
    json_str = pkgutil.get_data(tket_exts.__name__, f"data/{replacement}.json")
    assert json_str is not None, f"Could not load json for extension {name}"
    return Extension.from_json(json_str.decode())


class TketExtension(Protocol):
    def TYPES(self) -> List[ExtType]: ...
    def OPS(self) -> List[ExtOp]: ...
    def __call__(self) -> Extension: ...
