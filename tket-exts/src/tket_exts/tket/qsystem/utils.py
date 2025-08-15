"""QSystem utils extension operations."""

import functools

from hugr.ext import Extension
from .._util import load_extension


@functools.cache
def extension() -> Extension:
    return load_extension("tket.qsystem.utils")
