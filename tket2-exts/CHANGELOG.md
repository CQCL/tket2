# Changelog

## [0.4.0](https://github.com/CQCL/tket2/compare/tket2-exts-v0.3.0...tket2-exts-v0.4.0) (2025-02-20)


### ⚠ BREAKING CHANGES

* **tket2-hseries:** The signature of `QSystemOp::LazyMeasure` is changed to consume its qubit.

### Features

* add "tket2.qsystem.random" extension ([#779](https://github.com/CQCL/tket2/issues/779)) ([f4901ee](https://github.com/CQCL/tket2/commit/f4901eed22f8e66ca5ea3ccb8d72ead134ff8001))
* add a `tket2.qsystem.utils` extension with `GetCurrentShot` ([#772](https://github.com/CQCL/tket2/issues/772)) ([175a02d](https://github.com/CQCL/tket2/commit/175a02da2ce8a0065c265cdae7518c1b5284cff3)), closes [#767](https://github.com/CQCL/tket2/issues/767)
* **tket2-hseries:** Add `tket2.wasm` extension ([#737](https://github.com/CQCL/tket2/issues/737)) ([34bdc21](https://github.com/CQCL/tket2/commit/34bdc218b5e9bf334830873e847935dea0053242))
* **tket2-hseries:** Redefine `QSystemOp::LazyMeasure` and introduce `QSystemOp::LazyMeasureReset` ([#741](https://github.com/CQCL/tket2/issues/741)) ([1f126c0](https://github.com/CQCL/tket2/commit/1f126c0a4f7686fa6941a05aa28228786baac6d1))
* update measurement and alloc operations ([#702](https://github.com/CQCL/tket2/issues/702)) ([a7a0201](https://github.com/CQCL/tket2/commit/a7a020116f42bfeb89c356d08816a2f3ce1b5226))


### Bug Fixes

* remove unicode pi symbols in rotation extension ([#743](https://github.com/CQCL/tket2/issues/743)) ([b3ed351](https://github.com/CQCL/tket2/commit/b3ed35108d5fe93c3aa8101084b695470c488a30))


### Documentation

* docstring capitalisation ([#686](https://github.com/CQCL/tket2/issues/686)) ([e18f921](https://github.com/CQCL/tket2/commit/e18f921903953dc6a033ef697092f80a99a142b0))


## [0.3.0](https://github.com/CQCL/tket2/compare/tket2-exts-v0.2.0...tket2-exts-v0.3.0) (2024-12-16)


### ⚠ BREAKING CHANGES

* Updated `hugr` rust dependency to `0.14.0`. Extension are now defined in `hugr 0.14` style.

### Miscellaneous Chores

* Update to next version of hugr ([#720](https://github.com/CQCL/tket2/issues/720)) ([4a3a5a5](https://github.com/CQCL/tket2/commit/4a3a5a5e38252d4ee709e7e97bb5a1e90bd9fff4))

## [0.2.0](https://github.com/CQCL/tket2/compare/tket2-exts-v0.1.1...tket2-exts-v0.2.0) (2024-12-02)


### ⚠ BREAKING CHANGES

* all references to "hseries" including extension, modules and structs renamed to "qsystem"
* hseries qalloc op replaced with fallible TryQalloc

### Features

* update measurement and alloc operations ([#702](https://github.com/CQCL/tket2/issues/702)) ([a7a0201](https://github.com/CQCL/tket2/commit/a7a020116f42bfeb89c356d08816a2f3ce1b5226))


### Code Refactoring

* rename hseries to qsystem ([#703](https://github.com/CQCL/tket2/issues/703)) ([1e90173](https://github.com/CQCL/tket2/commit/1e90173872e73c44a6321fe400ae6f2e4e115220))

## [0.1.1](https://github.com/CQCL/tket2/compare/tket2-exts-v0.1.0...tket2-exts-v0.1.1) (2024-11-05)


### Documentation

* docstring capitalisation ([#686](https://github.com/CQCL/tket2/issues/686)) ([e18f921](https://github.com/CQCL/tket2/commit/e18f921903953dc6a033ef697092f80a99a142b0))

## 0.1.0 (2024-11-06)


### Features

* Add tket2 extension definitions.
