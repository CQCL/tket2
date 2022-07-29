# tket2proto
Prototype development of TKET-2 in rust

You will need rust >= 1.61. 
You will need access to thet [tket-json-rs repository](https://github.com/CQCL/tket-json-rs) as that is a git dependency.

With that in place `cargo build`, `cargo test` should work.

## Optional features

### pyo3
This optional feature enables some python bindings via pyo3. See the `pyrs` folder for more.

### cxx
This enables binding to TKET-1 code using [cxx](https://cxx.rs/). For this you will need access to the [tket-rs](https://github.com/CQCL-DEV/tket-rs) repository, 
which in turn will require `conan` installed and the tket remote conan repository added. See the [TKET readme](https://github.com/CQCL/tket) for more on conan.
