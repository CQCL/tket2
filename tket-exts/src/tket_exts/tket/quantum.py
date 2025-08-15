"""Quantum extension operations."""

import functools

from hugr.ext import Extension
from ._util import load_extension


@functools.cache
def extension() -> Extension:
    """TKET's standard quantum operations"""
    return load_extension("tket.quantum")
