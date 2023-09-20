//! General tests.

use std::collections::HashSet;

use hugr::Hugr;
use rstest::rstest;
use tket_json_rs::circuit_json::{self, SerialCircuit};

use crate::circuit::Circuit;
use crate::json::TKETDecode;

const SIMPLE_JSON: &str = r#"{
        "phase": "0",
        "bits": [],
        "qubits": [["q", [0]], ["q", [1]]],
        "commands": [
            {"args": [["q", [0]]], "op": {"type": "H"}},
            {"args": [["q", [0]], ["q", [1]]], "op": {"type": "CX"}}
        ],
        "implicit_permutation": [[["q", [0]], ["q", [0]]], [["q", [1]], ["q", [1]]]]
    }"#;

const UNKNOWN_OP: &str = r#"{
        "phase": "1/2",
        "bits": [["c", [0]], ["c", [1]]],
        "qubits": [["q", [0]], ["q", [1]], ["q", [2]]],
        "commands": [
            {"args": [["q", [0]], ["q", [1]], ["q", [2]]], "op": {"type": "CSWAP"}},
            {"args": [["q", [1]], ["c", [1]]], "op": {"type": "Measure"}}
        ],
        "created_qubits": [],
        "discarded_qubits": [],
        "implicit_permutation": [[["q", [0]], ["q", [0]]], [["q", [1]], ["q", [1]]], [["q", [2]], ["q", [2]]]]
    }"#;

const PARAMETRIZED: &str = r#"{
        "phase": "0.0",
        "bits": [],
        "qubits": [["q", [0]], ["q", [1]]],
        "commands": [
            {"args":[["q",[0]]],"op":{"type":"H"}},
            {"args":[["q",[1]],["q",[0]]],"op":{"type":"CX"}},
            {"args":[["q",[0]]],"op":{"params":["0.1"],"type":"Rz"}},
            {"args": [["q", [0]]], "op": {"params": ["3.141596/pi", "alpha", "0.3"], "type": "TK1"}}
        ],
        "created_qubits": [],
        "discarded_qubits": [],
        "implicit_permutation": [[["q", [0]], ["q", [0]]], [["q", [1]], ["q", [1]]]]
    }"#;

#[rstest]
#[case::simple(SIMPLE_JSON, 2, 2)]
#[case::unknown_op(UNKNOWN_OP, 2, 3)]
#[case::parametrized(PARAMETRIZED, 4, 2)]
fn json_roundtrip(#[case] circ_s: &str, #[case] num_commands: usize, #[case] num_qubits: usize) {
    let ser: circuit_json::SerialCircuit = serde_json::from_str(circ_s).unwrap();
    assert_eq!(ser.commands.len(), num_commands);

    let circ: Hugr = ser.clone().decode().unwrap();

    assert_eq!(circ.qubit_count(), num_qubits);

    let reser: SerialCircuit = SerialCircuit::encode(&circ).unwrap();
    compare_serial_circs(&ser, &reser);
}

fn compare_serial_circs(a: &SerialCircuit, b: &SerialCircuit) {
    assert_eq!(a.name, b.name);
    assert_eq!(a.phase, b.phase);

    let qubits_a: HashSet<_> = a.qubits.iter().collect();
    let qubits_b: HashSet<_> = b.qubits.iter().collect();
    assert_eq!(qubits_a, qubits_b);

    let bits_a: HashSet<_> = a.bits.iter().collect();
    let bits_b: HashSet<_> = b.bits.iter().collect();
    assert_eq!(bits_a, bits_b);

    assert_eq!(a.implicit_permutation, b.implicit_permutation);

    assert_eq!(a.commands.len(), b.commands.len());

    // the below only works if both serial circuits share a topological ordering
    // of commands.
    for (a, b) in a.commands.iter().zip(b.commands.iter()) {
        assert_eq!(a.op.op_type, b.op.op_type);
        assert_eq!(a.args, b.args);
        assert_eq!(a.op.params, b.op.params);
    }
    // TODO: Check commands equality (they only implement PartialEq)
}
