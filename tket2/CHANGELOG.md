# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.0-alpha.1](https://github.com/CQCL/tket2/releases/tag/tket2-v0.0.0-alpha.1) - 2024-05-24

### Bug Fixes
- Check for rewrite composition in badger ([#255](https://github.com/CQCL/tket2/pull/255))
- induced cycles in depth optimisation ([#264](https://github.com/CQCL/tket2/pull/264))
- Encode opaque symbolic constants ([#273](https://github.com/CQCL/tket2/pull/273))
- Correctly detect custom ops by name ([#281](https://github.com/CQCL/tket2/pull/281))
- Track input linear units in `Command` ([#310](https://github.com/CQCL/tket2/pull/310))
- Don't convert tket2 measurements into tket1 ops ([#331](https://github.com/CQCL/tket2/pull/331))

### Documentation
- Expand the main module and README docs ([#298](https://github.com/CQCL/tket2/pull/298))

### New Features
- add angle type to tket2 extension  ([#231](https://github.com/CQCL/tket2/pull/231))
- bindings for circuit cost and hash ([#252](https://github.com/CQCL/tket2/pull/252))
- Implement `PyErr` conversion locally in `tket2-py` ([#258](https://github.com/CQCL/tket2/pull/258))
- Add a "progress timeout" to badger ([#259](https://github.com/CQCL/tket2/pull/259))
- [**breaking**] Add lexicographic cost ([#270](https://github.com/CQCL/tket2/pull/270))
- rewrite tracing ([#267](https://github.com/CQCL/tket2/pull/267))
- Move pre/post rewrite cost to the RewriteStrategy API ([#276](https://github.com/CQCL/tket2/pull/276))
- [**breaking**] Lexicographic cost fn ([#277](https://github.com/CQCL/tket2/pull/277))
- Return rewrite strategies as a generator ([#275](https://github.com/CQCL/tket2/pull/275))
- add qalloc, qfree, reset ops ([#284](https://github.com/CQCL/tket2/pull/284))
- [**breaking**] Support any ops in portmatching ([#293](https://github.com/CQCL/tket2/pull/293))
- Add `PatternMatch::nodes` and `subcircuit` + matching example ([#299](https://github.com/CQCL/tket2/pull/299))
- Use `IncomingPort` and `OutgoingPort` instead of `Port` where possible. ([#296](https://github.com/CQCL/tket2/pull/296))
- expose Tk2Op name ([#307](https://github.com/CQCL/tket2/pull/307))

### Refactor
- Move tket2 code to a workspace member ([#210](https://github.com/CQCL/tket2/pull/210))
- Restructure the python code ([#211](https://github.com/CQCL/tket2/pull/211))
- s/taso/badger/ ([#228](https://github.com/CQCL/tket2/pull/228))
- Move python bindings from `tket2` to `tket2-py` ([#235](https://github.com/CQCL/tket2/pull/235))
- rename t2op ([#256](https://github.com/CQCL/tket2/pull/256))

### Testing
- Add small parallel badger test ([#237](https://github.com/CQCL/tket2/pull/237))
- fix non-deterministic badger test ([#245](https://github.com/CQCL/tket2/pull/245))
