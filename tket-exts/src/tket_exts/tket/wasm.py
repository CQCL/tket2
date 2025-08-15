"""WASM extension operations."""

import functools

from hugr.ext import Extension
from ._util import load_extension


@functools.cache
def extension() -> Extension:
    """WASM interop"""
    return load_extension("tket.wasm")
