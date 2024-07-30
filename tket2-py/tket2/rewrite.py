# Re-export native bindings
from ._tket2.rewrite import ECCRewriter, CircuitRewrite, Subcircuit

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
    try:
        import tket2_eccs
    except ImportError:
        raise ValueError(
            "The default rewriter is not available. Please specify a path to a rewriter or install tket2-eccs."
        )

    rewriter = tket2_eccs.nam_6_3()
    return ECCRewriter.load_precompiled(rewriter)
