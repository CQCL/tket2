# Re-export native bindings
from ._tket.rewrite import ECCRewriter, CircuitRewrite, Subcircuit
from .matcher import MatchReplaceRewriter

__all__ = [
    "default_ecc_rewriter",
    "clifford_t_ecc_rewriter",
    # Bindings.
    # TODO: Wrap these in Python classes.
    "ECCRewriter",
    "CircuitRewrite",
    "Subcircuit",
    "Rewriter",
]

Rewriter = ECCRewriter | MatchReplaceRewriter | list["Rewriter"]


def default_ecc_rewriter() -> ECCRewriter:
    """Load the default ecc rewriter."""
    try:
        import tket_eccs
    except ImportError:
        raise ValueError(
            "The default rewriter is not available. Please specify a path to a rewriter or install tket-eccs."
        )

    rewriter = tket_eccs.nam_6_3()
    return ECCRewriter.load_precompiled(rewriter)


def clifford_t_ecc_rewriter() -> ECCRewriter:
    """Load the ECC rewriter on the Clifford T gateset."""
    try:
        import tket_eccs
    except ImportError:
        raise ValueError(
            "The default rewriter is not available. Please specify a path to a rewriter or install tket-eccs."
        )

    rewriter = tket_eccs.clifford_t_6_3()
    return ECCRewriter.load_precompiled(rewriter)
