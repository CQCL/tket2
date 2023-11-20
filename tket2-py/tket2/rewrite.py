# Re-export native bindings
from .tket2._rewrite import *  # noqa: F403
from .tket2 import _rewrite

from pathlib import Path
import importlib

__all__ = [
    "default_ecc_rewriter",
    *_rewrite.__all__,
]


def default_ecc_rewriter() -> _rewrite.ECCRewriter:
    """Load the default ecc rewriter."""
    # TODO: Cite, explain what this is
    rewriter = Path(importlib.resources.files("tket2").joinpath("data/nam_6_3.rwr"))
    return _rewrite.ECCRewriter.load_precompiled(rewriter)
