# tket2

[![build_status][]](https://github.com/CQCL-DEV/tket2/actions)
![msrv][]

Version 2 of the TKET compiler.

  [build_status]: https://github.com/CQCL-DEV/hugr/workflows/Continuous%20integration/badge.svg?branch=main
  [msrv]: https://img.shields.io/badge/rust-1.70.0%2B-blue.svg

## Features

- `pyo3`
This optional feature enables some python bindings via pyo3. See the `pyrs` folder for more.

- `tkcxx`
  This enables binding to TKET-1 code using [cxx](https://cxx.rs/). For this you will to set up an environment with conan. See the [tket-rs README](https://github.com/CQCL-DEV/tket-rs#readme) for more details.

## Developing TKET2

See [DEVELOPMENT.md](DEVELOPMENT.md) for instructions on setting up the development environment.

## License

This project is licensed under Apache License, Version 2.0 ([LICENSE][] or http://www.apache.org/licenses/LICENSE-2.0).

  [LICENSE]: LICENCE
