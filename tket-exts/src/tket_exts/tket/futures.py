"""Futures extension operations."""

import functools

from hugr.ext import Extension
from ._util import load_extension


@functools.cache
def extension() -> Extension:
    """Future type and handling operations"""
    return load_extension("tket.futures")
