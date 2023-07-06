//! General tests.

use hugr::hugr::region::FlatRegionView;
use hugr::{Hugr, HugrView};
use tket_json_rs::circuit_json;

use crate::circuit::Circuit;
use crate::json::json_convert::TKET1Decode;

#[test]
fn read_json() {
    // let expr = symengine::Expression::new("a + b + 3");
    let circ_s = r#"{"bits": [["c", [0]], ["c", [1]]], "commands": [{"args": [["q", [0]]], "op": {"type": "H"}}, {"args": [["q", [0]], ["q", [1]]], "op": {"type": "CX"}}, {"args": [["q", [0]], ["c", [0]]], "op": {"type": "Measure"}}, {"args": [["q", [1]], ["c", [1]]], "op": {"type": "Measure"}}], "implicit_permutation": [[["q", [0]], ["q", [0]]], [["q", [1]], ["q", [1]]]], "phase": "0", "qubits": [["q", [0]], ["q", [1]]]}"#;
    let ser: circuit_json::SerialCircuit = serde_json::from_str(circ_s).unwrap();
    assert_eq!(ser.commands.len(), 4);

    let hugr: Hugr = ser.decode().unwrap();
    let circ = FlatRegionView::new(&hugr, hugr.root());

    assert_eq!(circ.qubits().len(), 2);

    //check_soundness(&circ).unwrap();

    //let _reser: SerialCircuit = circ.into();
    //assert_eq!(&ser, &_reser);

    // ser and reser cannot be compared because they will be different up to
    // topsort ordering of parallel commands
}

#[test]
fn read_json_unknown_op() {
    // test ops that are not native to tket-2 are correctly captured as
    // custom and output

    let circ_s = r#"{"bits": [], "commands": [{"args": [["q", [0]], ["q", [1]], ["q", [2]]], "op": {"type": "CSWAP"}}], "created_qubits": [], "discarded_qubits": [], "implicit_permutation": [[["q", [0]], ["q", [0]]], [["q", [1]], ["q", [1]]], [["q", [2]], ["q", [2]]]], "phase": "0", "qubits": [["q", [0]], ["q", [1]], ["q", [2]]]}"#;
    let ser: circuit_json::SerialCircuit = serde_json::from_str(circ_s).unwrap();
    assert_eq!(ser.commands.len(), 1);

    let hugr: Hugr = ser.decode().unwrap();
    let circ = FlatRegionView::new(&hugr, hugr.root());

    assert_eq!(circ.qubits().len(), 3);

    //let mut coms = circ.to_commands();
    //coms.next(); // skip input
    //let com = coms.next().unwrap();
    //assert!(matches!(com.op, &Op::Custom(_)));

    //let _reser: SerialCircuit = circ.into();
    //assert_eq!(&ser, &_reser);
}
