# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
