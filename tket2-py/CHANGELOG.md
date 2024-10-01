# Changelog

## [0.4.0](https://github.com/CQCL/tket2/compare/tket2-py-v0.3.0...tket2-py-v0.4.0) (2024-10-01)


### ⚠ BREAKING CHANGES

* Made all errors `non_exhaustive`, and renamed some fields for clarity.
* "tket2.angle" extension replaced with "tket2.rotation" extension with rotation type and simplified set of operations.
* TryFrom implementations for extension op structs removed, use `cast`

### Features

* `BadgerOptimiser.load_precompiled`, `BadgerOptimiser.compile_eccs` and `passes.badger_pass` now take an optional `cost_fn` parameter to specify the cost function to minimise. Supported values are `'cx'` (default behaviour) and `'rz'`. ([83ebfcb](https://github.com/CQCL/tket2/commit/83ebfcb9156fb5516f877155939062d11c7196d5))
* simplify angle extension in to a half turns rotation type ([#611](https://github.com/CQCL/tket2/issues/611)) ([0723937](https://github.com/CQCL/tket2/commit/0723937a8aed69302359fbd2383a01a77adc6b36))
* Support encoding float and sympy ops ([#618](https://github.com/CQCL/tket2/issues/618)) ([74dcbf7](https://github.com/CQCL/tket2/commit/74dcbf757c1c2ae57bf313c61296df270167ee7d))
* **tket2-hseries:** cli extension dumping ([#584](https://github.com/CQCL/tket2/issues/584)) ([abf292f](https://github.com/CQCL/tket2/commit/abf292f82b858840eeb067394105dbc62feb3230))


### Bug Fixes

* remove TryFrom for extension ops use `cast` ([#592](https://github.com/CQCL/tket2/issues/592)) ([5ca29af](https://github.com/CQCL/tket2/commit/5ca29af3a7cc6ce6d8b3261b342015da94f4aab0))
* Support hugr packages, fix the notebooks ([#622](https://github.com/CQCL/tket2/issues/622)) ([1cf9dcb](https://github.com/CQCL/tket2/commit/1cf9dcb7ba80dd236916bcf86a1fa0f5459fd349))


### Documentation

* Add tket2-py module docstring ([#539](https://github.com/CQCL/tket2/issues/539)) ([8ef7a57](https://github.com/CQCL/tket2/commit/8ef7a5736294cf462b0694c235f0d10316c68f68))


### Miscellaneous Chores

* Replace thiserror with derive_more 1.0 ([#624](https://github.com/CQCL/tket2/issues/624)) ([2250ce7](https://github.com/CQCL/tket2/commit/2250ce76f3019deee1d7e1206aa7843a745fe988))

## [0.3.0](https://github.com/CQCL/tket2/compare/tket2-py-v0.2.1...tket2-py-v0.3.0) (2024-09-04)


### ⚠ BREAKING CHANGES

* quantum extension renamed from "quantum.tket2" to "tket2.quantum"
* Parametric Tk2Ops take angle type rather than float.
* Rename lazy extension to "hseries" and add hseries ops.
* `Tk2Op::AngleAdd` removed, use `fadd` from the hugr float ops standard extension
* Updated to latest hugr version (`hugr-rs 0.12` / `hugr-py 0.8`). ECCs need to be recompiled.

### Features

* HSeries ops ([#573](https://github.com/CQCL/tket2/issues/573)) ([e6acc88](https://github.com/CQCL/tket2/commit/e6acc881d7ab67c584e6622d387bf2025209f8b8))
* remove Tk2Op::AngleAdd ([#567](https://github.com/CQCL/tket2/issues/567)) ([42cc82f](https://github.com/CQCL/tket2/commit/42cc82f0ee6e77dffb2f55c53613a7c4c8687824))
* replace f64 with angle type for tk2 ops ([#578](https://github.com/CQCL/tket2/issues/578)) ([d14631f](https://github.com/CQCL/tket2/commit/d14631f762f7ad2cf7db65e66b11cc38a54966ef))
* Update rust hugr dependency to `0.12.0`, and python hugr to `0.8.0` ([#568](https://github.com/CQCL/tket2/issues/568)) ([258a7c5](https://github.com/CQCL/tket2/commit/258a7c5ec25ee2665c524a174704944f0c19729e))


### Bug Fixes

* quantum extension name wrong way round ([#582](https://github.com/CQCL/tket2/issues/582)) ([06a6838](https://github.com/CQCL/tket2/commit/06a68386ce6d0b1376ed369da2a42d3b3eaa056a))

## [0.2.1](https://github.com/CQCL/tket2/compare/tket2-py-v0.2.0...tket2-py-v0.2.1) (2024-08-14)

### ⚠ BREAKING CHANGES

* increased minimum required version of hugr to 0.7.0

### Documentation

* Add tket2-py module docstring ([#539](https://github.com/CQCL/tket2/issues/539)) ([8ef7a57](https://github.com/CQCL/tket2/commit/8ef7a5736294cf462b0694c235f0d10316c68f68))

## [0.2.0](https://github.com/CQCL/tket2/compare/tket2-py-v0.1.0...tket2-py-v0.2.0) (2024-08-01)


### ⚠ BREAKING CHANGES

* `.rwr` ECC files generated with older versions are no longer supported. Please recompile them, or compress the file with `zstd`.

### Features

* Add timeout options and stats to Badger ([#496](https://github.com/CQCL/tket2/issues/496)) ([32a9885](https://github.com/CQCL/tket2/commit/32a98853db1415c6934fa26c966cb430c74023aa))
* Compress binary ECCs using zlib ([#498](https://github.com/CQCL/tket2/issues/498)) ([d9a713c](https://github.com/CQCL/tket2/commit/d9a713c068077dfea8c12301c1973575ddc8ca2c))
* Expose advanced Badger timeout options to tket2-py ([#506](https://github.com/CQCL/tket2/issues/506)) ([fe7d40e](https://github.com/CQCL/tket2/commit/fe7d40e4a176375c22fae5e52d42aeaf9ab38d33))
* Move the compiled eccs to a separate package ([#517](https://github.com/CQCL/tket2/issues/517)) ([7247cc6](https://github.com/CQCL/tket2/commit/7247cc65f4c4e679fd5b680d1e53e630f06d94a1))


### Bug Fixes

* Recompile ecc sets after [#441](https://github.com/CQCL/tket2/issues/441) ([#484](https://github.com/CQCL/tket2/issues/484)) ([1122fa4](https://github.com/CQCL/tket2/commit/1122fa453790fe5fd3433d4e028fb327949d9619))


### Miscellaneous Chores

* bump hugr version to 0.10.0 ([#508](https://github.com/CQCL/tket2/issues/508)) ([eca258b](https://github.com/CQCL/tket2/commit/eca258bfbef5fcd82a0d3b3d70cb736e275b3487))

## [0.1.0a4](https://github.com/CQCL/tket2/compare/tket2-py-v0.1.0...tket2-py-v0.1.0a4) (2024-07-15)


### Bug Fixes

* Recompile ecc sets after [#441](https://github.com/CQCL/tket2/issues/441) ([#484](https://github.com/CQCL/tket2/issues/484)) ([1122fa4](https://github.com/CQCL/tket2/commit/1122fa453790fe5fd3433d4e028fb327949d9619))

## [0.1.0a3](https://github.com/CQCL/tket2/compare/tket2-py-v0.1.0...tket2-py-v0.1.0a3) (2024-07-12)


### Documentation

* simplify the getting started notebook ([#466](https://github.com/CQCL/tket2/issues/466)) ([10639b9](https://github.com/CQCL/tket2/commit/10639b9ea6766cf9a91deb556dc1852881c7cd96))

## [0.1.0a2](https://github.com/CQCL/tket2/compare/tket2-py-v0.0.0a1...tket2-py-v0.1.0a2) (2024-07-10)


### ⚠ BREAKING CHANGES

* `Dfg` binding removed, replaced with builder from `hugr` package.

### Features

* Add `lhs` and `rhs` bindings to `Rule` ([#440](https://github.com/CQCL/tket2/issues/440)) ([49b1c89](https://github.com/CQCL/tket2/commit/49b1c89aeeaff73c6ebf66d3f81dddc4ad81f7bb))
* get pauli propagation + examples working with new hugr builder [#465](https://github.com/CQCL/tket2/issues/465) ([cab0d87](https://github.com/CQCL/tket2/commit/cab0d8750c5bc92c4f1284e07146add371db233f))

## 0.1.0a1 (2024-06-28)

Initial alpha release of the Python bindings for TKET2.

Includes compatibility with pytket circuits and guppy definitions, as well as
some basic circuit passes. See the included examples for more information.
