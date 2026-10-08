# HUGR handoff: current-package benchmark

19 deterministic entrypoints call the unchanged current guppyalgos public implementations. No added `@guppy.comptime`, loop peeling, gate rewrites, comparator replacements, or manual region extraction. The three Cuccaro examples use the package's public carry-out adder, not the historical handoff-local plain MAJ/UMA construction. QROM includes its default and public Toffoli compute/uncompute configurations. SELECT uses its library defaults.

Both variants run `PauliGraphResynthesis::default()` on independent copies of each same generated HUGR. The enabled variant sets only `with_t_optimization(true)`. Ancilla budget is unset (`None`, automatic); seed and scope retain their defaults.

| Run | Completed | Failed | Failure |
|---|---:|---:|---|
| Resynthesis only | 0 | 19 | Unsupported `SetBits` |
| Resynthesis + T optimisation | 0 | 6 | Unsupported `SetBits` |
| Resynthesis + T optimisation | 0 | 13 | Unsupported black boxes |

**No before/after optimisation comparison is available: failed transformations have no accepted output and are not counted as unchanged circuits.** Diagnostic counts in results.json are static gate nodes, not execution-weighted costs; loops, calls and control flow remain. Seven T gates per Toffoli is only an explicitly labelled diagnostic lowering estimate and excludes synthesis of arbitrary rotations.

All 19 generated envelopes loaded and validated in the Rust loader; Python envelope/checksum verification passed. No guppyalgos package files were edited.

| Circuit | Resynthesis only | With T optimisation |
|---|---|---|
| qrom_unary_addr3_data3_temp_and | SetBits | SetBits |
| qrom_unary_addr4_data4_temp_and | SetBits | SetBits |
| arithmetic_gidney_adder_n4 | SetBits | Black boxes |
| arithmetic_gidney_adder_n8 | SetBits | Black boxes |
| arithmetic_gidney_multiplier_n3 | SetBits | Black boxes |
| arithmetic_gidney_multiplier_n4 | SetBits | Black boxes |
| lcu_rotation_4term_state2 | SetBits | Black boxes |
| lcu_rotation_8term_state2 | SetBits | Black boxes |
| lcu_alias_8term_state2_precision4 | SetBits | Black boxes |
| qrom_unary_addr3_data3_toffoli | SetBits | SetBits |
| qrom_unary_addr4_data4_toffoli | SetBits | SetBits |
| select_pauli_4term_state2_toffoli | SetBits | SetBits |
| select_pauli_8term_state2_toffoli | SetBits | SetBits |
| arithmetic_cuccaro_adder_n4 | SetBits | Black boxes |
| arithmetic_cuccaro_adder_n8 | SetBits | Black boxes |
| arithmetic_cuccaro_adder_n16 | SetBits | Black boxes |
| toffoli_v_chain_n4 | SetBits | Black boxes |
| toffoli_v_chain_n8 | SetBits | Black boxes |
| toffoli_v_chain_n16 | SetBits | Black boxes |

Reproduce from the tket2 root:

```sh
/Users/silas.dilkes/Code/guppyalgos/.venv/bin/python hugr_handoff_runtime/source/generate.py
cargo run --locked --release -p tket --example handoff_benchmark -- hugr_handoff_runtime
```
