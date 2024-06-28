# Changelog

## 0.1.0 (2024-06-28)


### ⚠ BREAKING CHANGES

* require `hugr-0.5.0`
* Replaced `tket2.circuit.OpConvertError` with `tket2.circuit.TK1ConvertError` in the python lib.
* Moved `tket2::json` to `tket2::serialize::pytket`
* Replaced the `Circuit` trait with a wrapper struct.
* This is a breaking change to the compiled rewriter serialisation format.

### Features

* Add a "progress timeout" to badger ([#259](https://github.com/CQCL/tket2/issues/259)) ([556cf64](https://github.com/CQCL/tket2/commit/556cf64063a080fd8b52d9bdfb9ee911f4c691f0))
* Add missing typing hints ([#352](https://github.com/CQCL/tket2/issues/352)) ([4990613](https://github.com/CQCL/tket2/commit/49906139b2758899b9cc9a3205b55d193a570950))
* bindings for circuit cost and hash ([#252](https://github.com/CQCL/tket2/issues/252)) ([85ce5f9](https://github.com/CQCL/tket2/commit/85ce5f9aa1233d0837d93f7666e58a318aba01ae))
* drop pyo3 core dep ([#355](https://github.com/CQCL/tket2/issues/355)) ([9f7d415](https://github.com/CQCL/tket2/commit/9f7d415c93a7db62c548587975265d29beca49e4))
* EccRewriter bindings ([#251](https://github.com/CQCL/tket2/issues/251)) ([97e2e0a](https://github.com/CQCL/tket2/commit/97e2e0ae57d7bf2029831f2a6404e62bf2d62251))
* guppy → pytket conversion ([#407](https://github.com/CQCL/tket2/issues/407)) ([8c5a487](https://github.com/CQCL/tket2/commit/8c5a48772384d8f9f6b2c75cae020c3d3e4cdc6e))
* Implement `PyErr` conversion locally in `tket2-py` ([#258](https://github.com/CQCL/tket2/issues/258)) ([3e1a68d](https://github.com/CQCL/tket2/commit/3e1a68dd153439f4d25d0bc18b6ba7721f1c49ce))
* init tket2-hseries ([#368](https://github.com/CQCL/tket2/issues/368)) ([61e7535](https://github.com/CQCL/tket2/commit/61e7535dc909faf72d10967001725dc3a52bde08))
* pauli propagation use case example ([#333](https://github.com/CQCL/tket2/issues/333)) ([f46973c](https://github.com/CQCL/tket2/commit/f46973c0bc482121df41e48d1d94fc40d99cdf94))
* **py:** Allow using `Tk2Op`s in the builder ([#436](https://github.com/CQCL/tket2/issues/436)) ([aed8651](https://github.com/CQCL/tket2/commit/aed8651a111a3b1feccac318469850094e660a95))
* Support any ops in portmatching ([#293](https://github.com/CQCL/tket2/issues/293)) ([6b05a05](https://github.com/CQCL/tket2/commit/6b05a05a3251715cf19f6f08b4b7a8a8e1558f1c))
* **tket2-py:** Bind the `lower_to_pytket` pass in python ([#439](https://github.com/CQCL/tket2/issues/439)) ([8208324](https://github.com/CQCL/tket2/commit/82083247d7a5e38eceb2c1649a10ea83c189440f))
* Use tket1 and tket2 circuits interchangeably everywhere ([#243](https://github.com/CQCL/tket2/issues/243)) ([eac7acf](https://github.com/CQCL/tket2/commit/eac7acf9fd4cee513db4a75dab22f0600f7e0cc6))
* Utilities for loading compiled guppy circuits ([#393](https://github.com/CQCL/tket2/issues/393)) ([028779a](https://github.com/CQCL/tket2/commit/028779af447b6302e31b338a294e465817c8bfc0))


### Bug Fixes

* failed importlib import ([#254](https://github.com/CQCL/tket2/issues/254)) ([b077660](https://github.com/CQCL/tket2/commit/b07766011e5c2fd8c4e8984335858c6d50a9f37b))
* induced cycles in depth optimisation ([#264](https://github.com/CQCL/tket2/issues/264)) ([68c9ff2](https://github.com/CQCL/tket2/commit/68c9ff2002c36fe1000e9ee1388599f5c08c3e8d)), closes [#253](https://github.com/CQCL/tket2/issues/253)
* Make native py modules behave like python's ([#212](https://github.com/CQCL/tket2/issues/212)) ([4220038](https://github.com/CQCL/tket2/commit/422003820136d08ccf315586be5faa2fac6b26ee)), closes [#209](https://github.com/CQCL/tket2/issues/209)
* pytest failing to find `tket2` in CI ([#367](https://github.com/CQCL/tket2/issues/367)) ([a9df8e6](https://github.com/CQCL/tket2/commit/a9df8e6e087e41707e35080fa3feb6a29b10faeb))
* **tket2-py:** Replace `with_hugr` helpers with `with_circ`, keeping the circuit parent. ([#438](https://github.com/CQCL/tket2/issues/438)) ([b77b3cb](https://github.com/CQCL/tket2/commit/b77b3cbe72fb935a2e70656c60bdda86bfa15a6d))


### Documentation

* Add some example notebooks for the python package. ([#443](https://github.com/CQCL/tket2/issues/443)) ([4ed276c](https://github.com/CQCL/tket2/commit/4ed276c619d3afd9ea69971b2eb05c23e0bd032d)), closes [#434](https://github.com/CQCL/tket2/issues/434)
* Update tket2-py readme ([6c8f18a](https://github.com/CQCL/tket2/commit/6c8f18a08da2c98776e619b16096a062caafe8ad))


### Code Refactoring

* Rename `tket2::json` into `tket2::serialize::pytket` ([#392](https://github.com/CQCL/tket2/issues/392)) ([93e611c](https://github.com/CQCL/tket2/commit/93e611c2fe414d46e129974fcf0edb0c3c004e97))
* Replace Circuit trait with a struct ([#370](https://github.com/CQCL/tket2/issues/370)) ([ec5dd22](https://github.com/CQCL/tket2/commit/ec5dd2269d5b28a0f3399ad00549643d35502c0d))
* Simplify tket1 conversion errors ([#408](https://github.com/CQCL/tket2/issues/408)) ([b0b8aff](https://github.com/CQCL/tket2/commit/b0b8aff269d7572940e3dd67a20537e768890f07))
