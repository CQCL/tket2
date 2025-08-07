"""TKET2 is an open source quantum compiler developed by Quantinuum.

> [!WARNING]
> This package has been renamed to `tket`. Please update your dependencies accordingly.
> The legacy `pytket` compiler will be incorporated into `tket`.

Central to TKET2's design is its hardware agnosticism which allows researchers
and quantum software developers to take advantage of its powerful compilation
for many different quantum architectures.

This is the Python front-end for TKET2, providing a high-level interface for
working with quantum circuits. See also the Rust library with the same name on
[crates.io](https://crates.io/crates/tket2).
"""

# This is updated by our release-please workflow, triggered by this
# annotation: x-release-please-version
__version__ = "0.12.0"

raise RuntimeError(
    "The `tket2` package has been renamed to `tket`. Please update your dependencies accordingly."
)
