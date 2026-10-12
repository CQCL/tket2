# HUGR handoff: compile-time circuit construction

The same 19 constructions and fixed inputs as `hugr_handoff_runtime`, using the same guppyalgos version. Circuit-building loops, indices and static dispatch execute during `@guppy.comptime` tracing. The original gate bodies, library defaults, comparator, ladders and temporary-AND decompositions are retained. Neither guppyalgos checkout was edited.

The generator adapts the existing source bodies for Python tracing, rather than using the historical handoff kernels. Type parameters are inferred from array lengths; array ownership bookkeeping runs in Python. Original and adapted bodies plus source-file hashes are saved in `source_audit.json`. Measurement feed-forward remains runtime code. The remaining two loops in QROM/SELECT/LCU inputs only measure and read the terminal work-register zero checks; no computational gates or calls occur inside them.

Both variants run the same `PauliGraphResynthesis::default()` pipeline on independent copies of each identical expanded HUGR, changing only `with_t_optimization(false/true)`. Ancilla budget is unset (`None`, automatic), seed is 0 and scope is default.

Both modes completed **19/19**; all **38 output HUGRs validated**. Paired execution checks passed for 13 fixed-output circuits, four shots each in the runtime and comptime forms (104 shots). LCU circuits and the three cases requiring more than 20 simulation qubits were not included in those execution checks.

Counts are static T/Tdg nodes and exclude approximation of arbitrary rotations. Computational loops have been expanded. The prepared column estimates seven T gates per remaining Toffoli; it is not a separately synthesised baseline. Runtime-suite static totals count loop bodies once and cannot be compared directly with these expanded totals.

| Circuit | Prepared 7T estimate | Resynthesis T/Tdg | With T optimisation | Extra T reduction |
|---|---:|---:|---:|---:|
| qrom_unary_addr3_data3_temp_and | 24 | 24 | 24 | 0 |
| qrom_unary_addr4_data4_temp_and | 56 | 56 | 56 | 0 |
| arithmetic_gidney_adder_n4 | 16 | 16 | 16 | 0 |
| arithmetic_gidney_adder_n8 | 32 | 32 | 32 | 0 |
| arithmetic_gidney_multiplier_n3 | 48 | 48 | 48 | 0 |
| arithmetic_gidney_multiplier_n4 | 88 | 88 | 88 | 0 |
| lcu_rotation_4term_state2 | 28 | 16 | 16 | 0 |
| lcu_rotation_8term_state2 | 84 | 48 | 48 | 0 |
| lcu_alias_8term_state2_precision4 | 252 | 188 | 166 | 22 |
| qrom_unary_addr3_data3_toffoli | 84 | 48 | 48 | 0 |
| qrom_unary_addr4_data4_toffoli | 196 | 112 | 112 | 0 |
| select_pauli_4term_state2_toffoli | 28 | 16 | 16 | 0 |
| select_pauli_8term_state2_toffoli | 84 | 48 | 48 | 0 |
| arithmetic_cuccaro_adder_n4 | 36 | 24 | 22 | 2 |
| arithmetic_cuccaro_adder_n8 | 92 | 56 | 56 | 0 |
| arithmetic_cuccaro_adder_n16 | 204 | 120 | 120 | 0 |
| toffoli_v_chain_n4 | 49 | 31 | 31 | 0 |
| toffoli_v_chain_n8 | 105 | 63 | 63 | 0 |
| toffoli_v_chain_n16 | 217 | 127 | 127 | 0 |
| **Total** | **1723** | **1161** | **1137** | **24** |

Extra T optimisation saves **24/1,161 T gates (2.1%)**: 22 in alias LCU and 2 in the 4-bit Cuccaro adder. The other 17 circuits have unchanged T counts.

Reproduce from the tket2 root, using the same Guppy environment:

```sh
/Users/silas.dilkes/Code/guppyalgos/.venv/bin/python hugr_handoff_comptime/source/generate.py
cargo run --locked --release -p tket --example handoff_benchmark -- hugr_handoff_comptime
/Users/silas.dilkes/Code/guppyalgos/.venv/bin/python hugr_handoff_comptime/source/verify.py
```

Use `verify.py --structure-only` for checksum, input-matching and loop checks without simulation. `summary.csv`, `results.json`, `provenance.json` and `execution_checks.json` contain the results and verification records. Optimised output HUGRs are retained locally in `outputs/` and excluded from git.
