# tket2

[![build_status][]](https://github.com/CQCL-DEV/tket2/actions)
![msrv][]
[![codecov][]](https://codecov.io/gh/CQCL/tket2)

Version 2 of the TKET compiler.

  [build_status]: https://github.com/CQCL-DEV/hugr/workflows/Continuous%20integration/badge.svg?branch=main
  [msrv]: https://img.shields.io/badge/rust-1.71.0%2B-blue.svg
  [codecov]: https://img.shields.io/codecov/c/gh/CQCL/tket2?logo=codecov

## Features

- `pyo3`
  Enables some python bindings via pyo3. See the `tket2-py` folder for more.

- `portmatching`
  Enables pattern matching using the `portmatching` crate.

- `rewrite-tracing`
  Adds opt-in tracking of the rewrites applied to a circuit.

## Developing TKET2

See [DEVELOPMENT.md](DEVELOPMENT.md) for instructions on setting up the development environment.

## License

This project is licensed under Apache License, Version 2.0 ([LICENSE][] or http://www.apache.org/licenses/LICENSE-2.0).

  [LICENSE]: LICENCE
