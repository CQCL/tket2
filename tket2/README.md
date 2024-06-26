# tket2: The Hardware Agnostic Quantum Compiler

[![build_status][]](https://github.com/CQCL/tket2/actions)
![msrv][]
[![codecov][]](https://codecov.io/gh/CQCL/tket2)

TKET2 is an open source quantum compiler developed by Quantinuum. Central to
TKET2's design is its hardware agnosticism which allows researchers and quantum
software developers to take advantage of its powerful compilation for many
different quantum architectures.

Circuits are represented using the HUGR IR defined in the
[hugr] crate. TKET2 augments Hugr with
* The [`Circuit`] trait, providing a high-level interface for working with HUGRs representing quantum circuits
* a HUGR extension with quantum operations
* A composable pass system for optimising circuits
* A number of built-in rewrite utilities and passes for common optimisations

This crate is interoperable with [`tket1`] circuits via its
serial encoding.

# Using TKET2

Defining a circuit in TKET2 is currently done by using the low-level [hugr Builder] API, or by loading tket1 circuits from JSON files.

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

Please read the [API documentation here][].

## Features

- `portmatching`
  Enables pattern matching using the [`portmatching`][] crate.

- `rewrite-tracing`
  Adds opt-in tracking of the rewrites applied to a circuit.

## Recent Changes

See [CHANGELOG][] for a list of changes. The minimum supported rust
version will only change on major releases.

## Developing TKET2

See [DEVELOPMENT.md][] for instructions on setting up the development environment.

## License

This project is licensed under Apache License, Version 2.0 ([LICENSE][] or http://www.apache.org/licenses/LICENSE-2.0).

  [build_status]: https://github.com/CQCL/tket2/workflows/Continuous%20integration/badge.svg?branch=main
  [msrv]: https://img.shields.io/badge/rust-1.75.0%2B-blue.svg
  [codecov]: https://img.shields.io/codecov/c/gh/CQCL/tket2?logo=codecov
  [hugr]: https://lib.rs/crates/hugr
  [hugr Builder]: https://docs.rs/hugr/latest/hugr/builder/index.html
  [API documentation here]: https://docs.rs/tket2/
  [`Circuit`]: https://docs.rs/tket2/latest/tket2/trait.Circuit.html
  [`tket1`]: https://github.com/CQCL/tket
  [`portmatching`]: https://lib.rs/crates/portmatching
  [LICENSE]: https://github.com/CQCL/tket2/blob/main/LICENCE
  [CHANGELOG]: https://github.com/CQCL/tket2/blob/main/tket2/CHANGELOG.md
  [DEVELOPMENT.md]: https://github.com/CQCL/tket2/blob/main/DEVELOPMENT.md
