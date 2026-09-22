# pg-greedy-synth

`pg-greedy-synth` provides `GreedySynthPass`, which uses a greedy heuristic to
synthesize a canonical Pauli graph. It processes commuting operation sets in
order and selects TQEs that reduce the current operations while taking nearby
operations into account.

The pass can output:

1. Single qubit Clifford gates
2. TQE gates
3. Single qubit rotation gates
4. Measurements
5. Resets
6. Conditional gates
7. Black boxes, which are preserved

See [ARCHITECTURE.md](ARCHITECTURE.md) for an overview of the internal design.
See [TUNING.md](TUNING.md) for internal tuning constants.

## Configured Clifford actions

`GreedySynthPass::with_clifford_actions` replaces the built-in TQE candidates
with target-specific Clifford actions. Each `CliffordAction` contains an
arbitrary-width local tableau, one concrete ordered qubit placement, and its
target cost and duration. Selected actions are emitted as tableau operations;
single-qubit basis and sign corrections retain their existing gate forms.

The supplied actions must contain a strictly reducing choice for every
non-local Pauli operation and trailing tableau state reached by synthesis. The
pass reports an incomplete action set rather than silently emitting a TQE.
