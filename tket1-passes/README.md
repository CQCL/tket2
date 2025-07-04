# TKET1-Passes

A bridge between C++ TKET1 and Rust land. The folder `tket-c-api`
provides a minimal C-interface to a subset of TKET1's functionality. The
crate in this directory then wraps this interface into a Rust-friendly
API.

## Building instructions

Building `tket1-passes` is done using `conan >= 2.0`. You must set up `conan` to
use the artifactory containing the tket package:

```
conan remote add tket-libs https://quantinuumsw.jfrog.io/artifactory/api/conan/tket1-libs --index 0
```

If conan is unable to fetch all dependencies as pre-compiled binaries, you will
also need standard C++ tooling to compile the dependencies (i.e. a reasonably
recent version of cmake and a C++ compiler).

## Currently supported TKET1 features

This library is currently limited to TKET passes (a small subset of them, in fact).
Specifically, the following passes are supported:

- **`two_qubit_squash`** - Squash sequences of two-qubit operations using KAK decomposition
- **`clifford_resynthesis`** - Resynthesise Clifford subcircuits and simplify using Clifford rules
