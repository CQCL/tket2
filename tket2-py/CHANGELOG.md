# Changelog

## [0.8.0](https://github.com/CQCL/tket2/compare/tket2-py-v0.7.0...tket2-py-v0.8.0) (2025-03-17)


### ⚠ BREAKING CHANGES

* pre-envelope `Circuit` serialisation functions are deprecated.

### Features

* Add methods to en/decode from bytes ([#807](https://github.com/CQCL/tket2/issues/807)) ([3e76fd2](https://github.com/CQCL/tket2/commit/3e76fd2fb981889542b06cd218dfce9ad925cf29))
* Use hugr envelopes to store/load circuits ([#813](https://github.com/CQCL/tket2/issues/813)) ([2940b2e](https://github.com/CQCL/tket2/commit/2940b2e0c9b270259259690b83dbdf261543d26d))


### Miscellaneous Chores

* bump to hugr-rs 0.15 and hugr-py 0.11 ([#806](https://github.com/CQCL/tket2/issues/806)) ([f3bfaae](https://github.com/CQCL/tket2/commit/f3bfaae0b6e4b8fd934a62343317b85ccb8f96ee))

## [0.7.0](https://github.com/CQCL/tket2/compare/tket2-py-v0.6.1...tket2-py-v0.7.0) (2025-03-06)


### ⚠ BREAKING CHANGES

* **tket2-py:** Adds `OrderInZones` to `tket2.qsystem.utils`

### Features

* **tket2-py:** bump tket2_exts constraint ([#794](https://github.com/CQCL/tket2/issues/794)) ([520e950](https://github.com/CQCL/tket2/commit/520e9505630d1bc166472fb1af1095bb39e8b414))

## [0.6.1](https://github.com/CQCL/tket2/compare/tket2-py-v0.6.0...tket2-py-v0.6.1) (2025-02-21)


### Features

* add a `tket2.qsystem.utils` extension with `GetCurrentShot` ([#772](https://github.com/CQCL/tket2/issues/772)) ([175a02d](https://github.com/CQCL/tket2/commit/175a02da2ce8a0065c265cdae7518c1b5284cff3)), closes [#767](https://github.com/CQCL/tket2/issues/767)


### Bug Fixes

* include RNG extension in tket2-py, bump tket2-exts constraint ([#781](https://github.com/CQCL/tket2/issues/781)) ([9eb8897](https://github.com/CQCL/tket2/commit/9eb8897fe5eed96070700b2f43461da1fd1346ee))

## [0.6.0](https://github.com/CQCL/tket2/compare/tket2-py-v0.5.1...tket2-py-v0.6.0) (2024-12-16)


### ⚠ BREAKING CHANGES

* Updated `hugr-rs` to `0.14.0` / `hugr-py 0.10.0`. Extension are now defined in `hugr 0.14`-compatible format.

### Features

* update measurement and alloc operations ([#702](https://github.com/CQCL/tket2/issues/702)) ([a7a0201](https://github.com/CQCL/tket2/commit/a7a020116f42bfeb89c356d08816a2f3ce1b5226))


### Miscellaneous Chores

* Update to next version of hugr ([#720](https://github.com/CQCL/tket2/issues/720)) ([4a3a5a5](https://github.com/CQCL/tket2/commit/4a3a5a5e38252d4ee709e7e97bb5a1e90bd9fff4))


## [0.5.1](https://github.com/CQCL/tket2/compare/tket2-py-v0.5.0...tket2-py-v0.5.1) (2024-11-29)


### Bug Fixes

* Remove use of removed auto_rebase_pass ([#708](https://github.com/CQCL/tket2/issues/708)) ([ea122a7](https://github.com/CQCL/tket2/commit/ea122a76444da0e94d745b02e2b475719cfd7bf7))

## [0.5.0](https://github.com/CQCL/tket2/compare/tket2-py-v0.4.1...tket2-py-v0.5.0) (2024-11-11)


### ⚠ BREAKING CHANGES

* Removed `load_guppy_*` methods. Use Circuit::load_function_reader instead.

### Features

* move extensions to auxillary package ([#682](https://github.com/CQCL/tket2/issues/682)) ([dd78f9a](https://github.com/CQCL/tket2/commit/dd78f9a571905407bcf74131a54f4aefbca740c4))
* Track circuit extensions and read/write packages ([#680](https://github.com/CQCL/tket2/issues/680)) ([5e87dd9](https://github.com/CQCL/tket2/commit/5e87dd94f216a87f4d27dee44d178578d40e7ace))


## [0.4.1](https://github.com/CQCL/tket2/compare/tket2-py-v0.4.0...tket2-py-v0.4.1) (2024-10-10)


### Features

* Add `tket2.rotation.from_halfturns_unchecked` op ([#640](https://github.com/CQCL/tket2/issues/640)) ([86ffe64](https://github.com/CQCL/tket2/commit/86ffe64fa455fc19a60999a7edb71803d615d77b))

## [0.4.0](https://github.com/CQCL/tket2/compare/tket2-py-v0.3.0...tket2-py-v0.4.0) (2024-10-01)


### ⚠ BREAKING CHANGES

* "tket2.angle" extension replaced with "tket2.rotation" extension with rotation type and simplified set of operations.

### Features

* `BadgerOptimiser.load_precompiled`, `BadgerOptimiser.compile_eccs` and `passes.badger_pass` now take an optional `cost_fn` parameter to specify the cost function to minimise. Supported values are `'cx'` (default behaviour) and `'rz'`. ([83ebfcb](https://github.com/CQCL/tket2/commit/83ebfcb9156fb5516f877155939062d11c7196d5))
* simplify angle extension in to a half turns rotation type ([#611](https://github.com/CQCL/tket2/issues/611)) ([0723937](https://github.com/CQCL/tket2/commit/0723937a8aed69302359fbd2383a01a77adc6b36))


### Bug Fixes

* Support hugr packages, fix the notebooks ([#622](https://github.com/CQCL/tket2/issues/622)) ([1cf9dcb](https://github.com/CQCL/tket2/commit/1cf9dcb7ba80dd236916bcf86a1fa0f5459fd349))


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
