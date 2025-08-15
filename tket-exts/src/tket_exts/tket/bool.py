"""Opaque boolean operations."""

import functools

from hugr.ext import Extension
from ._util import load_extension


@functools.cache
def extension() -> Extension:
    """Opaque boolean extension"""
    return load_extension("tket.bool")
