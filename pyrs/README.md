## pyrs

This package uses [pyo3](https://pyo3.rs/v0.16.4/) and
[maturin](https://github.com/PyO3/maturin) to bind TKET2 functionality to
python as the `pyrs` package.

Recommended:

A clean python 3.10 environment with `maturin` installed. At which point running
`maturin develop` in this directory should build and install the package in the
environment. Run `pytest` in this directory to test everything is working.
