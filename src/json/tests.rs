//! General tests.

use std::collections::HashSet;

use hugr::hugr::region::{FlatRegionView, Region};
use hugr::{Hugr, HugrView};
use tket_json_rs::circuit_json::{self, SerialCircuit};

use crate::circuit::Circuit;
use crate::json::TKETDecode;

#[test]
fn read_json_simple() {
    let circ_s = r#"{
        "phase": "0",
        "bits": [],
        "qubits": [["q", [0]], ["q", [1]]],
        "commands": [
            {"args": [["q", [0]]], "op": {"type": "H"}},
            {"args": [["q", [0]], ["q", [1]]], "op": {"type": "CX"}}
        ],
        "implicit_permutation": [[["q", [0]], ["q", [0]]], [["q", [1]], ["q", [1]]]]
    }"#;

    let ser: circuit_json::SerialCircuit = serde_json::from_str(circ_s).unwrap();
    assert_eq!(ser.commands.len(), 2);

    let hugr: Hugr = ser.clone().decode().unwrap();
    let circ = FlatRegionView::new(&hugr, hugr.root());

    assert_eq!(circ.qubits().len(), 2);

    let reser: SerialCircuit = SerialCircuit::encode(&circ).unwrap();
    compare_serial_circs(&ser, &reser);
}

#[test]
fn read_json_unknown_op() {
    // test ops that are not native to tket-2 are correctly captured as
    // custom and output

    let circ_s = r#"{
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

    let ser: SerialCircuit = serde_json::from_str(circ_s).unwrap();
    assert_eq!(ser.commands.len(), 3);

    let hugr: Hugr = ser.clone().decode().unwrap();
    let circ = FlatRegionView::new(&hugr, hugr.root());

    assert_eq!(circ.qubits().len(), 3);

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
