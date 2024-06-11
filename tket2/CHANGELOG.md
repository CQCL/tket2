# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0-alpha.2](https://github.com/CQCL/tket2/compare/tket2-v0.1.0-alpha.1...tket2-v0.1.0-alpha.2) - 2024-06-11

### Bug Fixes
- Commands iterator ignoring the hierarchy. ([#381](https://github.com/CQCL/tket2/pull/381))

### New Features
- Replace `Circuit::num_gates` with `num_operations` ([#384](https://github.com/CQCL/tket2/pull/384))
- Utilities for loading compiled guppy circuits ([#393](https://github.com/CQCL/tket2/pull/393))

### Refactor
- [**breaking**] Replace Circuit trait with a struct ([#370](https://github.com/CQCL/tket2/pull/370))
- [**breaking**] Rename `tket2::json` into `tket2::serialize::pytket` ([#392](https://github.com/CQCL/tket2/pull/392))

## [0.1.0-alpha.1](https://github.com/CQCL/tket2/releases/tag/tket2-v0.1.0-alpha.1) - 2024-05-24

Initial alpha release of the library
