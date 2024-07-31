# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0-alpha.3](https://github.com/CQCL/tket2/compare/tket2-v0.1.0-alpha.2...tket2-v0.1.0-alpha.3) - 2024-07-31

### Bug Fixes
- Single source of truth for circuit names, and better circuit errors ([#390](https://github.com/CQCL/tket2/pull/390))
- Support non-DFG circuits ([#391](https://github.com/CQCL/tket2/pull/391))
- Portmatching not matching const edges ([#444](https://github.com/CQCL/tket2/pull/444))
- Pattern matcher discriminating on opaqueOp description ([#441](https://github.com/CQCL/tket2/pull/441))
- `extract_dfg` inserting the output node with an invalid child order ([#442](https://github.com/CQCL/tket2/pull/442))
- Recompile ecc sets after [#441](https://github.com/CQCL/tket2/pull/441) ([#484](https://github.com/CQCL/tket2/pull/484))

### Documentation
- Update tket2-py readme ([#431](https://github.com/CQCL/tket2/pull/431))
- Better error reporting in portmatching ([#437](https://github.com/CQCL/tket2/pull/437))
- Improved multi-threading docs for Badger ([#495](https://github.com/CQCL/tket2/pull/495))

### New Features
- `Circuit::operations` ([#395](https://github.com/CQCL/tket2/pull/395))
- tuple unpack rewrite ([#406](https://github.com/CQCL/tket2/pull/406))
- guppy → pytket conversion ([#407](https://github.com/CQCL/tket2/pull/407))
- Drop linear bits, improve pytket encoding/decoding ([#420](https://github.com/CQCL/tket2/pull/420))
- *(py)* Allow using `Tk2Op`s in the builder ([#436](https://github.com/CQCL/tket2/pull/436))
- Initial support for `TailLoop` as circuit parent ([#417](https://github.com/CQCL/tket2/pull/417))
- Support tuple unpacking with multiple unpacks ([#470](https://github.com/CQCL/tket2/pull/470))
- Partial tuple unpack ([#475](https://github.com/CQCL/tket2/pull/475))
- [**breaking**] Compress binary ECCs using zlib ([#498](https://github.com/CQCL/tket2/pull/498))
- Add timeout options and stats to Badger ([#496](https://github.com/CQCL/tket2/pull/496))
- Expose advanced Badger timeout options to tket2-py ([#506](https://github.com/CQCL/tket2/pull/506))

### Refactor
- [**breaking**] Simplify tket1 conversion errors ([#408](https://github.com/CQCL/tket2/pull/408))
- Cleanup tket1 serialized op structures ([#419](https://github.com/CQCL/tket2/pull/419))

### Testing
- Add coverage for Badger split circuit multi-threading ([#505](https://github.com/CQCL/tket2/pull/505))

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
