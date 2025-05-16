# Changelog

## [0.7.0](https://github.com/CQCL/tket2/compare/tket2-exts-v0.6.0...tket2-exts-v0.7.0) (2025-05-16)


### ⚠ BREAKING CHANGES

* bump to hugr 0.20
* **hseries:** ZZMax removed from Qsystem extension. Use ZZPhase(pi/2).

### Features

* Add `tket2.bool` extension ([#823](https://github.com/CQCL/tket2/issues/823)) ([8818d2f](https://github.com/CQCL/tket2/commit/8818d2f122ec3911366d02c18e347752a427fad9))
* Add debug extension with state result op ([#843](https://github.com/CQCL/tket2/issues/843)) ([64bbe88](https://github.com/CQCL/tket2/commit/64bbe88ddd0a214044d2636f3d4bd8262d6b05f5)), closes [#832](https://github.com/CQCL/tket2/issues/832)
* **hseries:** remove ZZMax operation from Qsystem extension ([#852](https://github.com/CQCL/tket2/issues/852)) ([b488125](https://github.com/CQCL/tket2/commit/b4881256b2d6a5c21c1d7a69d91384c5d2cc9905))


### Miscellaneous Chores

* Bump to hugr-0.20 ([#862](https://github.com/CQCL/tket2/issues/862)) ([652a7d0](https://github.com/CQCL/tket2/commit/652a7d0b039bca62407f16f7548204e97f92ef5a))

## [0.6.0](https://github.com/CQCL/tket2/compare/tket2-exts-v0.5.1...tket2-exts-v0.6.0) (2025-03-18)


### ⚠ BREAKING CHANGES

* Removed `OrderInZones` operation from `tket2.qsystem.utils`.


### Removed

* Removed `OrderInZones` operation from `tket2.qsystem.utils`. ([#797](https://github.com/CQCL/tket2/issues/797)) ([8c3ee89](https://github.com/CQCL/tket2/commit/8c3ee8971b9f095ccdb0522cf0751a2ff20b89a5))


## [0.5.1](https://github.com/CQCL/tket2/compare/tket2-exts-v0.5.0...tket2-exts-v0.5.1) (2025-03-06)


### Features

* Add order_in_zones extension op ([3ec7f5d](https://github.com/CQCL/tket2/commit/3ec7f5d5e0a7d07254e1b09976cddea98cd83702))

## [0.5.0](https://github.com/CQCL/tket2/compare/tket2-exts-v0.4.0...tket2-exts-v0.5.0) (2025-03-04)


### ⚠ BREAKING CHANGES

* To be compatible with Guppy's convention of implicitly returning `self` as the second value of the tuple, the following signatures are updated: ```diff
    -    /// `fn random_int(RNGContext) -> (RNGContext, u32)`
    +   /// `fn random_int(RNGContext) -> (u32, RNGContext)`

### Bug Fixes

* remove type argument from `RNGContext` type, swap returns ([#786](https://github.com/CQCL/tket2/issues/786)) ([633ebd7](https://github.com/CQCL/tket2/commit/633ebd74d71ba81f5b71d6db757b08ea3c959a5d))

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
