# tket: The Hardware Agnostic Quantum Compiler

[![build_status][]](https://github.com/CQCL/tket2/actions)
[![codecov][]](https://codecov.io/gh/CQCL/tket)

TKET is an open source quantum compiler developed by Quantinuum. Central to
TKET's design is its hardware agnosticism which allows researchers and quantum
software developers to take advantage of its powerful compilation for many
different quantum architectures.

`tket` is available as a rust crate on [crates.io](https://crates.io/crates/tket) and as
a python package on [PyPI](https://pypi.org/project/tket/).

See the respective
[Rust](https://github.com/CQCL/tket2/blob/main/tket) and
[Python](https://github.com/CQCL/tket2/blob/main/tket-py) READMEs for
more information.

## Usage

The rust crate documentation is available at [docs.rs](https://docs.rs/tket).

See the [Getting Started][getting-started] notebook for a quick introduction to using `tket` in Python.

  [getting-started]: https://github.com/CQCL/tket2/blob/main/tket-py/examples/1-Getting-Started.ipynb

## Developing TKET

See [DEVELOPMENT.md][] for instructions on setting up the development environment.

## License

This project is licensed under Apache License, Version 2.0 ([LICENSE][] or http://www.apache.org/licenses/LICENSE-2.0).

  [build_status]: https://github.com/CQCL/tket2/workflows/Continuous%20integration/badge.svg?branch=main
  [codecov]: https://img.shields.io/codecov/c/gh/CQCL/tket?logo=codecov
  [LICENSE]: https://github.com/CQCL/tket2/blob/main/LICENCE
  [DEVELOPMENT.md]: https://github.com/CQCL/tket2/blob/main/DEVELOPMENT.md
