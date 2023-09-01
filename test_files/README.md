## Generating the pattern matching dataset

The dataset that is necessary for the pattern matching dataset
can be obtained by unzipping the `T_Tdg_H_X_C~X_complete_ECC.zip`
file.

The data in this folder was obtained as follows:

- Using Quartz, create the complete ECC set with gate set T, Tdg, H, X and CX, 3 qubits, up to 7 gates. This creates the file `T_Tdg_H_X_CX_complete_ECC_set.json`.
- From the ECC extract all circuits in QASM format using the `benches/quartz/ecc_to_qasm.cpp` file (requires the `quartz_runtime` library).
- Run `python qasm_to_json.py .` from this folder to obtain TK1 JSON circuits of each QASM circuit
- Run `cargo run --bin=sanitise_patterns_dataset --features="portmatching" .` from this folder to remove empty and disconnected circuits. This step reduces the number of circuits from 4942 to 4913.