# HUGR handoff: current-package benchmark

19 deterministic entrypoints call the unchanged current guppyalgos public implementations. No added `@guppy.comptime`, loop peeling, gate rewrites, comparator replacements, or manual region extraction. The three Cuccaro examples use the package's public carry-out adder, not the historical handoff-local plain MAJ/UMA construction. QROM includes its default and public Toffoli compute/uncompute configurations. SELECT uses its library defaults.

Both variants run `PauliGraphResynthesis::default()` on independent copies of each same generated HUGR. The enabled variant sets only `with_t_optimization(true)`. Ancilla budget is unset (`None`, automatic); seed and scope retain their defaults.

| Run | Completed | Failed |
|---|---:|---:|
| Resynthesis only | 19 | 0 |
| Resynthesis + T optimisation | 19 | 0 |

All 38 outputs validated. All 19 input hashes still match the original manifest. No guppyalgos package files were edited.

Counts are static gate nodes, not execution-weighted costs; loops, calls and control flow remain. The prepared column estimates seven T gates per remaining Toffoli and excludes synthesis of arbitrary rotations. The other columns count actual T/Tdg nodes after the same resynthesis pipeline, with only the extra T optimisation switched on or off.

| Circuit | Prepared 7T estimate | Resynthesis T/Tdg | With T optimisation | Extra T reduction |
|---|---:|---:|---:|---:|
| qrom_unary_addr3_data3_temp_and | 8 | 8 | 8 | 0 |
| qrom_unary_addr4_data4_temp_and | 8 | 8 | 8 | 0 |
| arithmetic_gidney_adder_n4 | 12 | 12 | 12 | 0 |
| arithmetic_gidney_adder_n8 | 12 | 12 | 12 | 0 |
| arithmetic_gidney_multiplier_n3 | 12 | 12 | 12 | 0 |
| arithmetic_gidney_multiplier_n4 | 12 | 12 | 12 | 0 |
| lcu_rotation_4term_state2 | 28 | 28 | 28 | 0 |
| lcu_rotation_8term_state2 | 28 | 28 | 28 | 0 |
| lcu_alias_8term_state2_precision4 | 170 | 166 | 160 | 6 |
| qrom_unary_addr3_data3_toffoli | 28 | 28 | 28 | 0 |
| qrom_unary_addr4_data4_toffoli | 28 | 28 | 28 | 0 |
| select_pauli_4term_state2_toffoli | 28 | 28 | 28 | 0 |
| select_pauli_8term_state2_toffoli | 28 | 28 | 28 | 0 |
| arithmetic_cuccaro_adder_n4 | 48 | 48 | 48 | 0 |
| arithmetic_cuccaro_adder_n8 | 48 | 48 | 48 | 0 |
| arithmetic_cuccaro_adder_n16 | 48 | 48 | 48 | 0 |
| toffoli_v_chain_n4 | 21 | 21 | 21 | 0 |
| toffoli_v_chain_n8 | 21 | 21 | 21 | 0 |
| toffoli_v_chain_n16 | 21 | 21 | 21 | 0 |
| **Total** | **609** | **605** | **599** | **6** |

Reproduce from the tket2 root:

```sh
/Users/silas.dilkes/Code/guppyalgos/.venv/bin/python hugr_handoff_runtime/source/generate.py
cargo run --locked --release -p tket --example handoff_benchmark -- hugr_handoff_runtime
```
