"""Rotation extension operations."""

import functools

from hugr.ext import Extension
from ._util import load_extension


@functools.cache
def extension() -> Extension:
    """Rotation type for TKET's quantum operations"""
    return load_extension("tket.rotation")
