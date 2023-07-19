//! General tests.

use hugr::hugr::region::{FlatRegionView, Region};
use hugr::{Hugr, HugrView};
use tket_json_rs::circuit_json::{self, SerialCircuit};

use crate::circuit::Circuit;
use crate::json::TKET1Decode;

#[test]
fn read_json_simple() {
    let circ_s = r#"{"bits": [["c", [0]], ["c", [1]]], "commands": [{"args": [["q", [0]]], "op": {"type": "H"}}, {"args": [["q", [0]], ["q", [1]]], "op": {"type": "CX"}}, {"args": [["q", [0]], ["c", [0]]], "op": {"type": "Measure"}}, {"args": [["q", [1]], ["c", [1]]], "op": {"type": "Measure"}}], "implicit_permutation": [[["q", [0]], ["q", [0]]], [["q", [1]], ["q", [1]]]], "phase": "0", "qubits": [["q", [0]], ["q", [1]]]}"#;
    let ser: circuit_json::SerialCircuit = serde_json::from_str(circ_s).unwrap();
    assert_eq!(ser.commands.len(), 4);

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

    let circ_s = r#"{"bits": [], "commands": [{"args": [["q", [0]], ["q", [1]], ["q", [2]]], "op": {"type": "CSWAP"}}], "created_qubits": [], "discarded_qubits": [], "implicit_permutation": [[["q", [0]], ["q", [0]]], [["q", [1]], ["q", [1]]], [["q", [2]], ["q", [2]]]], "phase": "0", "qubits": [["q", [0]], ["q", [1]], ["q", [2]]]}"#;
    let ser: SerialCircuit = serde_json::from_str(circ_s).unwrap();
    assert_eq!(ser.commands.len(), 1);

    let hugr: Hugr = ser.clone().decode().unwrap();
    let circ = FlatRegionView::new(&hugr, hugr.root());

    assert_eq!(circ.qubits().len(), 3);

    let reser: SerialCircuit = SerialCircuit::encode(&circ).unwrap();
    compare_serial_circs(&ser, &reser);
}

fn compare_serial_circs(a: &SerialCircuit, b: &SerialCircuit) {
    assert_eq!(a.name, b.name);
    assert_eq!(a.phase, b.phase);
    // TODO: Make sure registers don't get reordered
    //assert_eq!(a.qubits, b.qubits);
    //assert_eq!(a.bits, b.bits);
    assert_eq!(a.implicit_permutation, b.implicit_permutation);
    // TODO: Implement Hash for Command, and compare the `commands` as sets
}
