# Re-export native bindings
from ._tket2.rewrite import ECCRewriter, CircuitRewrite, Subcircuit

from importlib import resources

__all__ = [
    "default_ecc_rewriter",
    # Bindings.
    # TODO: Wrap these in Python classes.
    "ECCRewriter",
    "CircuitRewrite",
    "Subcircuit",
]


def default_ecc_rewriter() -> ECCRewriter:
    """Load the default ecc rewriter."""
    # TODO: Cite, explain what this is
    with resources.as_file(
        resources.files("tket2").joinpath("data/nam_6_3.rwr")
    ) as rewriter:
        return ECCRewriter.load_precompiled(rewriter)
