# tket2proto
Prototype development of TKET-2 in rust

You will need rust >= 1.67.
You will need access to the [tket-json-rs repository](https://github.com/CQCL/tket-json-rs) as that is a git dependency.

With that in place `cargo build`, `cargo test` should work.

## Optional features

### pyo3
This optional feature enables some python bindings via pyo3. See the `pyrs` folder for more.

### cxx
This enables binding to TKET-1 code using [cxx](https://cxx.rs/). For this you will need access to the [tket-rs](https://github.com/CQCL-DEV/tket-rs) repository,
which in turn will require `conan` installed and the tket remote conan repository added. See the [TKET readme](https://github.com/CQCL/tket) for more on conan.


## Code Structure

A brief outline of how the code is currently structured

* `ext/portgraph`: a standalone graph library hosting the graph data structure
  and rewrite logic
    - `graph.rs`: core graph data structure and impls.
    - `substitute.rs`: graph rewriting impls and data structures
    - `toposort.rs`: topological traversal
    - `py_graph.rs`: pyo3 bindings

* `benches`: criterion benchmarks, limited, should be added to.
* `pyrs`: python bindings and tests using pyo3
* `src/circuit`: Circuit data structure and conversion impls
    - `circuit.rs`: core data structure and impls
    - `dag.rs`: type specialisations for graph
    - `operation.rs`: Operation enum, constants, WireType, Signature.
    - `py_circuit.rs`: python bindings and conversions
    - `tk1ops.rs`: CustomOp implementation for serialised tket1 `Operation`
    - `unitarybox.rs`: `CustomOp` implementation for a ffi-bound TKET1 SU2 box.
* `src/json`: Conversion to and from TKET1 serialised JSON format
* `src/passes`: Compilation passes and associated infrastructure
    - `classical.rs`: Constant folding pass
    - `pattern.rs`: Naive lazy fixed pattern matching implementation
    - `redundancy.rs`: **vestigial** redundancy removal implementation
    - `squash.rs`: Single qubit rotation squashing, all rotations are converted
      to quaternion rotations, and the angles constant folded.
    - `mod.rs`: generic strategies and iterators
* `src/validate`: simple validity checking for circuits
* `lib.rs`: integration tests
