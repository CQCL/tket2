# tket2: The Hardware Agnostic Quantum Compiler

[![build_status][]](https://github.com/CQCL-DEV/tket2/actions)
![msrv][]
[![codecov][]](https://codecov.io/gh/CQCL/tket2)

  [build_status]: https://github.com/CQCL-DEV/hugr/workflows/Continuous%20integration/badge.svg?branch=main
  [msrv]: https://img.shields.io/badge/rust-1.75.0%2B-blue.svg
  [codecov]: https://img.shields.io/codecov/c/gh/CQCL/tket2?logo=codecov

TKET2 is an open source quantum compiler developed by Quantinuum. Central to
TKET2's design is its hardware agnosticism which allows researchers and
quantum software developers to take advantage of its state of the art
compilation for many different quantum architectures.

Circuits are represented using the HUGR IR defined in the
[hugr] crate. TKET2 augments Hugr with
* The [`Circuit`] trait, providing a high-level interface for working with HUGRs representing quantum circuits
* a HUGR extension with quantum operations
* A composable pass system for optimising circuits
* A number of built-in rewrite utilities and passes for common optimisations

This crate is interoperable with [`tket1`] circuits via its
serial encoding.

  [hugr]: https://lib.rs/crates/hugr
  [`Circuit`]: https://docs.rs/tket2/latest/tket2/trait.Circuit.html
  [`tket1`]: https://github.com/CQCL/tket

# Using TKET2

Defining a circuit in TKET2 is currently done by using the low-level [hugr Builder] API, or by loading tket1 circuits from JSON files.

  [hugr Builder]: https://docs.rs/hugr/latest/hugr/builder/index.html

```rust
use tket2::{Circuit, Hugr};

// Load a tket1 circuit.
let mut circ: Hugr = tket2::json::load_tk1_json_file("test_files/barenco_tof_5.json").unwrap();

assert_eq!(circ.qubit_count(), 9);
assert_eq!(circ.num_gates(), 170);

// Traverse the circuit and print the gates.
for command in circ.commands() {
    println!("{:?}", command.optype());
}

// Render the circuit as a mermaid diagram.
println!("{}", circ.mermaid_string());

// Optimise the circuit.
tket2::passes::apply_greedy_commutation(&mut circ);
```

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
