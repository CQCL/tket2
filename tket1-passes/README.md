# TKET1-Passes

A bridge between C++ TKET1 and Rust land. The folder `cpp` provides a minimal
C-interface to a subset of TKET1's functionality. The folder `rust` then wraps
this interface into a Rust-friendly API.

## Building the C++ library

This requires standard modern C++ tooling:

- Conan 2.0+
- CMake 3.23+
- C++20 compiler

as well as the GMP (GNU Multiple Precision Arithmetic Library). This is usually
already installed on Linux; on macos you can use `brew install gmp`. You must
further set up `conan` to use the artifactory containing the tket package:

```
conan remote add tket-libs https://quantinuumsw.jfrog.io/artifactory/api/conan/tket1-libs --index 0
```

Finally, if you have not used `conan` before, you will need to set up a profile:

```
conan profile detect
```

Building the `tket1-passes` library is then as simple as running the following script:

```bash
cd cpp
./build.sh
```

## Building the Rust library

Upon successfully building the C++ library, the compiled library should be found
in the `lib` directory. The Rust library can then be built using:

```bash
cd rust
cargo build
```

## Currently supported TKET1 features

This library is currently limited to TKET passes (a small subset of them, in fact).
Specifically, the following passes are supported:

- **`two_qubit_squash`** - Squash sequences of two-qubit operations using KAK decomposition
- **`clifford_resynthesis`** - Resynthesise Clifford subcircuits and simplify using Clifford rules
