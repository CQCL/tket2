# Changelog

## [0.3.0](https://github.com/CQCL/tket2/compare/tket2-eccs-v0.2.0...tket2-eccs-v0.3.0) (2024-12-16)


### ⚠ BREAKING CHANGES

* Updated `hugr` rust dependency to `0.14.0` Extension are now defined in `hugr 0.14` style. Replaced the `ROTATION_TYPE` with a method call, and dropped the per-extension registry definitions.

### Features

* move extensions to auxillary package ([#682](https://github.com/CQCL/tket2/issues/682)) ([dd78f9a](https://github.com/CQCL/tket2/commit/dd78f9a571905407bcf74131a54f4aefbca740c4))


### Reverts

* "Support python 3.13 ([#653](https://github.com/CQCL/tket2/issues/653))" ([#657](https://github.com/CQCL/tket2/issues/657)) ([3fb67ac](https://github.com/CQCL/tket2/commit/3fb67acf5e860fba8f8e1b9e6c5836846a0fcb7d))


### Miscellaneous Chores

* Update to next version of hugr ([#720](https://github.com/CQCL/tket2/issues/720)) ([4a3a5a5](https://github.com/CQCL/tket2/commit/4a3a5a5e38252d4ee709e7e97bb5a1e90bd9fff4))

## [0.2.0](https://github.com/CQCL/tket2/compare/tket2-eccs-v0.1.0...tket2-eccs-v0.2.0) (2024-10-10)


### ⚠ BREAKING CHANGES

* Recompiled eccs with `hugr 0.13.0`

## 0.1.0 (2024-08-01)


### Features

* Move the compiled eccs to a separate package ([#517](https://github.com/CQCL/tket2/issues/517)) ([7247cc6](https://github.com/CQCL/tket2/commit/7247cc65f4c4e679fd5b680d1e53e630f06d94a1))

## Changelog
