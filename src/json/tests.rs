//! General tests.

use std::collections::HashSet;

use hugr::hugr::views::{HierarchyView, SiblingGraph};
use hugr::ops::handle::DfgID;
use hugr::{Hugr, HugrView};
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
            {"args": [["q", [1]], ["c", [1]]], "op": {"type": "Measure"}},
            {"args": [["q", [2]], ["c", [0]]], "op": {"type": "Measure"}}
        ],
        "created_qubits": [],
        "discarded_qubits": [],
        "implicit_permutation": [[["q", [0]], ["q", [0]]], [["q", [1]], ["q", [1]]], [["q", [2]], ["q", [2]]]]
    }"#;

#[rstest]
#[case::simple(SIMPLE_JSON, 2, 2)]
#[case::unknown_op(UNKNOWN_OP, 3, 3)]
fn json_roundtrip(#[case] circ_s: &str, #[case] num_commands: usize, #[case] num_qubits: usize) {
    let ser: circuit_json::SerialCircuit = serde_json::from_str(circ_s).unwrap();
    assert_eq!(ser.commands.len(), num_commands);

    let hugr: Hugr = ser.clone().decode().unwrap();
    let circ: SiblingGraph<'_, DfgID> = SiblingGraph::new(&hugr, hugr.root());

    assert_eq!(circ.qubits().len(), num_qubits);

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

    // TODO: Check commands equality (they only implement PartialEq)
}
