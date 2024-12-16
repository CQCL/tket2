# Changelog

## [0.3.0](https://github.com/CQCL/tket2/compare/tket2-exts-v0.2.0...tket2-exts-v0.3.0) (2024-12-16)


### ⚠ BREAKING CHANGES

* Updated `hugr` rust dependency to `0.14.0` Extension are now defined in `hugr 0.14` style. Replaced the `ROTATION_TYPE` with a method call, and dropped the per-extension registry definitions.

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
