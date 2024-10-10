"""TKET2 is an open source quantum compiler developed by Quantinuum.

Central to TKET2's design is its hardware agnosticism which allows researchers
and quantum software developers to take advantage of its powerful compilation
for many different quantum architectures.

This is the Python front-end for TKET2, providing a high-level interface for
working with quantum circuits. See also the Rust library with the same name on
[crates.io](https://crates.io/crates/tket2).
"""

from . import circuit, ops, optimiser, passes, pattern, rewrite

__all__ = ["circuit", "ops", "optimiser", "passes", "pattern", "rewrite"]


# This is updated by our release-please workflow, triggered by this
# annotation: x-release-please-version
__version__ = "0.4.1"
