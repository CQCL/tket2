# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.15.1](https://github.com/CQCL/tket2/compare/tket2-hseries-v0.15.0...tket2-hseries-v0.15.1) - 2025-06-18

### Bug Fixes

- Revert "fix: Change array result ops signature to return array resul… ([#921](https://github.com/CQCL/tket2/pull/921))

## [0.15.0](https://github.com/CQCL/tket2/compare/tket2-hseries-v0.14.1...tket2-hseries-v0.15.0) - 2025-06-16

### Bug Fixes

- *(tket2-hseries)* unicode-aware prefix in `emit_global_string` ([#902](https://github.com/CQCL/tket2/pull/902))
- [**breaking**] Change array result ops signature to return array result ([#888](https://github.com/CQCL/tket2/pull/888))

### New Features

- Add llvm lowering for debug extension ([#900](https://github.com/CQCL/tket2/pull/900))

### Refactor

- [**breaking**] More flexible pytket encoding ([#849](https://github.com/CQCL/tket2/pull/849))

## [0.14.1](https://github.com/CQCL/tket2/compare/tket2-hseries-v0.14.0...tket2-hseries-v0.14.1) - 2025-06-03

### New Features

- Add V and Vdg to quantum extension. ([#889](https://github.com/CQCL/tket2/pull/889))
- LLVM codegen for extensions ([#898](https://github.com/CQCL/tket2/pull/898))

## [0.14.0](https://github.com/CQCL/tket2/compare/tket2-hseries-v0.13.0...tket2-hseries-v0.14.0) - 2025-05-22

### ⚠ BREAKING CHANGES

- BoolOp::bool_to_sum / BoolOp::sum_to_bool renamed to BoolOp::read / BoolOp::make_opaque
- QSystemOp:Measure and QSystemOp:MeasureReset now return tket2.bools

### Bug Fixes

- *(tket2-hseries)* ensure deterministic lowering using maps ([#884](https://github.com/CQCL/tket2/pull/884))

### New Features

- *(tket2-hseries)* [**breaking**] insert RuntimeBarrier across qubits in a Barrier ([#866](https://github.com/CQCL/tket2/pull/866))
- [**breaking**] Add `ReplaceBoolPass` ([#854](https://github.com/CQCL/tket2/pull/854))
- *(tket2-hseries)* Remove `static_array<tket2.bool>` before `replace_bool`ing.   ([#885](https://github.com/CQCL/tket2/pull/885))

### Refactor

- *(tket2-hseries)* use smaller angle decompositions for CZ and CCX ([#883](https://github.com/CQCL/tket2/pull/883))

## [0.13.0](https://github.com/CQCL/tket2/compare/tket2-hseries-v0.12.0...tket2-hseries-v0.13.0) - 2025-05-16

### Bug Fixes

- [**breaking**] Do not use SimpleReplacement in lazify ([#873](https://github.com/CQCL/tket2/pull/873))

### New Features

- [**breaking**] bump msrv to 1.85 ([#868](https://github.com/CQCL/tket2/pull/868))

## [0.12.0](https://github.com/CQCL/tket2/compare/tket2-hseries-v0.11.0...tket2-hseries-v0.12.0) - 2025-05-06

### New Features

- Add `tket2.bool` extension ([#823](https://github.com/CQCL/tket2/pull/823))
- *(hseries)* [**breaking**] remove ZZMax operation from Qsystem extension ([#852](https://github.com/CQCL/tket2/pull/852))
- Add debug extension with state result op ([#843](https://github.com/CQCL/tket2/pull/843))

### Refactor

- Better error message on allocation failure. ([#827](https://github.com/CQCL/tket2/pull/827))

## [0.11.0](https://github.com/CQCL/tket2/compare/tket2-hseries-v0.10.0...tket2-hseries-v0.11.0) - 2025-03-17

### Bug Fixes

- [**breaking**] Remove `OrderInZones` operation. Make `UtilsOp` enum `non_exhaustive` ([#797](https://github.com/CQCL/tket2/pull/797))

### New Features

- Lower tk2 ops using function calls ([#812](https://github.com/CQCL/tket2/pull/812))

## [0.10.0](https://github.com/CQCL/tket2/compare/tket2-hseries-v0.9.1...tket2-hseries-v0.10.0) - 2025-03-06

### Bug Fixes

- [**breaking**] remove type argument from `RNGContext` type, swap returns ([#786](https://github.com/CQCL/tket2/pull/786))

### New Features

- *(tket2-hseries)* [**breaking**] Add order_in_zones extension op ([#792](https://github.com/CQCL/tket2/pull/792))

## [0.9.1](https://github.com/CQCL/tket2/compare/tket2-hseries-v0.9.0...tket2-hseries-v0.9.1) - 2025-02-25

### New Features

- add a `tket2.qsystem.utils` extension with `GetCurrentShot` (#772)
- add "tket2.qsystem.random" extension (#779)

## [0.9.0](https://github.com/CQCL/tket2/compare/tket2-hseries-v0.8.0...tket2-hseries-v0.9.0) - 2025-02-12

### Bug Fixes

- Use `RemoveDeadFuncsPass` instead of deprecated `remove_polyfuncs()` (#759)
- nondeterminism in lazify-measure (#766)

### New Features

- *(tket2-hseries)* Add `tket2.wasm` extension (#737)
- force-order qfree early and qalloc late. (#762)

## [0.8.0](https://github.com/CQCL/tket2/compare/tket2-hseries-v0.7.1...tket2-hseries-v0.8.0) - 2025-01-10

### New Features

- *(tket2-hseries)* [**breaking**] Redefine `QSystemOp::LazyMeasure` and introduce `QSystemOp::LazyMeasureReset` (#741)
- *(tket2-hseries)* Lazify more flavours of measure ops (#742)

## [0.7.1](https://github.com/CQCL/tket2/compare/tket2-hseries-v0.7.0...tket2-hseries-v0.7.1) - 2024-12-18

### New Features

- Add monomorphization and constant folding to QSystemPass (#730)

## [0.7.0](https://github.com/CQCL/tket2/compare/tket2-hseries-v0.6.0...tket2-hseries-v0.7.0) - 2024-12-16

### ⚠ BREAKING CHANGES

- Replaced the hseries `qalloc` op with a fallible `TryQalloc`
- Extension definitions and registries now use `Arc`s for sharing

### New Features

- [**breaking**] update measurement and alloc operations (#702)

### Refactor

- [**breaking**] update to hugr 0.14 (#700)
- [**breaking**] rename hseries to qsystem (#703)

## [0.4.0](https://github.com/CQCL/tket2/compare/tket2-hseries-v0.3.0...tket2-hseries-v0.4.0) - 2024-09-16

### New Features

- [**breaking**] `HSeriesPass` lowers `Tk2Op`s into `HSeriesOp`s ([#602](https://github.com/CQCL/tket2/pull/602))
- [**breaking**] simplify angle extension in to a half turns rotation type ([#611](https://github.com/CQCL/tket2/pull/611))

## [0.3.0](https://github.com/CQCL/tket2/compare/tket2-hseries-v0.2.0...tket2-hseries-v0.3.0) - 2024-09-09

### Bug Fixes

- extension ops checking against incorrect name ([#593](https://github.com/CQCL/tket2/pull/593))
- [**breaking**] remove TryFrom for extension ops use `cast` ([#592](https://github.com/CQCL/tket2/pull/592))

### New Features

- lowering tk2ops -> hseriesops ([#579](https://github.com/CQCL/tket2/pull/579))
- *(tket2-hseries)* cli extension dumping ([#584](https://github.com/CQCL/tket2/pull/584))

## [0.2.0](https://github.com/CQCL/tket2/compare/tket2-hseries-v0.1.1...tket2-hseries-v0.2.0) - 2024-09-04

### New Features
- [**breaking**] Update rust hugr dependency to `0.12.0`, and python hugr to `0.8.0` ([#568](https://github.com/CQCL/tket2/pull/568))
- [**breaking**] HSeries ops ([#573](https://github.com/CQCL/tket2/pull/573))
- [**breaking**] replace f64 with angle type for tk2 ops ([#578](https://github.com/CQCL/tket2/pull/578))

## [0.1.1](https://github.com/CQCL/tket2/compare/tket2-hseries-v0.1.0...tket2-hseries-v0.1.1) - 2024-08-15

### New Features
- *(tket2-hseries)* make result operation internals public ([#542](https://github.com/CQCL/tket2/pull/542))

## [0.1.0](https://github.com/CQCL/tket2/releases/tag/tket2-hseries-v0.1.0) - 2024-08-01

### New Features
- [**breaking**] init tket2-hseries ([#368](https://github.com/CQCL/tket2/pull/368))
- *(tket2-hseries)* Add `tket2.futures` Hugr extension ([#471](https://github.com/CQCL/tket2/pull/471))
- Add lazify-measure pass ([#482](https://github.com/CQCL/tket2/pull/482))
- add results extensions ([#494](https://github.com/CQCL/tket2/pull/494))
- *(tket2-hseries)* [**breaking**] Add `HSeriesPass` ([#487](https://github.com/CQCL/tket2/pull/487))
