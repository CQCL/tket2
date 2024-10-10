# Changelog

## [0.2.0](https://github.com/CQCL/tket2/compare/tket2-eccs-v0.1.0...tket2-eccs-v0.2.0) (2024-10-10)


### ⚠ BREAKING CHANGES

* Updated compiled ECCs to `hugr 0.13`'s serialization version
* "tket2.angle" extension replaced with "tket2.rotation" extension with rotation type and simplified set of operations.
* "tket2.quantum" ops no longer contains angle type/ops. Use "tket2.angle".
* quantum extension renamed from "quantum.tket2" to "tket2.quantum"
* Parametric Tk2Ops take angle type rather than float.
* Rename lazy extension to "hseries" and add hseries ops.
* `Tk2Op::AngleAdd` removed, use `fadd` from the hugr float ops standard extension
* Updated to latest hugr version (`hugr-rs 0.12` / `hugr-py 0.8`). ECCs need to be recompiled.
* Update hugr dependency

### Features

* HSeries ops ([#573](https://github.com/CQCL/tket2/issues/573)) ([e6acc88](https://github.com/CQCL/tket2/commit/e6acc881d7ab67c584e6622d387bf2025209f8b8))
* move angle types + and ops to new "tket2.angle" extension ([#591](https://github.com/CQCL/tket2/issues/591)) ([f0884cc](https://github.com/CQCL/tket2/commit/f0884cc605730dc3dfa5217e7217ecc207e9db9d))
* remove Tk2Op::AngleAdd ([#567](https://github.com/CQCL/tket2/issues/567)) ([42cc82f](https://github.com/CQCL/tket2/commit/42cc82f0ee6e77dffb2f55c53613a7c4c8687824))
* replace f64 with angle type for tk2 ops ([#578](https://github.com/CQCL/tket2/issues/578)) ([d14631f](https://github.com/CQCL/tket2/commit/d14631f762f7ad2cf7db65e66b11cc38a54966ef))
* simplify angle extension in to a half turns rotation type ([#611](https://github.com/CQCL/tket2/issues/611)) ([0723937](https://github.com/CQCL/tket2/commit/0723937a8aed69302359fbd2383a01a77adc6b36))
* Update compiled ECCs to hugr-rs 0.13 ([#647](https://github.com/CQCL/tket2/issues/647)) ([6919ab5](https://github.com/CQCL/tket2/commit/6919ab50893001798936b2f2f2c914f738542d38))
* Update rust hugr dependency to `0.12.0`, and python hugr to `0.8.0` ([#568](https://github.com/CQCL/tket2/issues/568)) ([258a7c5](https://github.com/CQCL/tket2/commit/258a7c5ec25ee2665c524a174704944f0c19729e))


### Bug Fixes

* quantum extension name wrong way round ([#582](https://github.com/CQCL/tket2/issues/582)) ([06a6838](https://github.com/CQCL/tket2/commit/06a68386ce6d0b1376ed369da2a42d3b3eaa056a))


### Miscellaneous Chores

* Update hugr dependency ([0196fdd](https://github.com/CQCL/tket2/commit/0196fdd60ee0cae8b85c80ec5662f6023fe89617))

## 0.1.0 (2024-08-01)


### Features

* Move the compiled eccs to a separate package ([#517](https://github.com/CQCL/tket2/issues/517)) ([7247cc6](https://github.com/CQCL/tket2/commit/7247cc65f4c4e679fd5b680d1e53e630f06d94a1))

## Changelog
