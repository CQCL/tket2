//! General tests.

use std::io::BufReader;

use hugr::builder::{DFGBuilder, Dataflow, DataflowHugr};
use hugr::extension::prelude::QB_T;

use hugr::std_extensions::arithmetic::float_types::{ConstF64, FLOAT64_TYPE};
use hugr::types::FunctionType;
use hugr::Hugr;
use rstest::{fixture, rstest};
use tket_json_rs::circuit_json::{self, SerialCircuit};
use tket_json_rs::optype;

use crate::circuit::Circuit;
use crate::extension::REGISTRY;
use crate::json::TKETDecode;
use crate::Tk2Op;

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

#[rstest]
#[cfg_attr(miri, ignore)] // Opening files is not supported in (isolated) miri
#[case::barenco_tof_10("../test_files/barenco_tof_10.json")]
fn json_file_roundtrip(#[case] circ: impl AsRef<std::path::Path>) {
    let reader = BufReader::new(std::fs::File::open(circ).unwrap());
    let ser: circuit_json::SerialCircuit = serde_json::from_reader(reader).unwrap();
    let circ: Hugr = ser.clone().decode().unwrap();
    let reser: SerialCircuit = SerialCircuit::encode(&circ).unwrap();
    compare_serial_circs(&ser, &reser);
}

#[fixture]
fn circ_add_angles_symbolic() -> Hugr {
    let input_t = vec![QB_T, FLOAT64_TYPE, FLOAT64_TYPE];
    let output_t = vec![QB_T];
    let mut h = DFGBuilder::new(FunctionType::new(input_t, output_t)).unwrap();

    let mut inps = h.input_wires();
    let qb = inps.next().unwrap();
    let f1 = inps.next().unwrap();
    let f2 = inps.next().unwrap();

    let res = h.add_dataflow_op(Tk2Op::AngleAdd, [f1, f2]).unwrap();
    let f12 = res.outputs().next().unwrap();
    let res = h.add_dataflow_op(Tk2Op::RxF64, [qb, f12]).unwrap();
    let qb = res.outputs().next().unwrap();

    h.finish_hugr_with_outputs([qb], &REGISTRY).unwrap()
}

#[fixture]
fn circ_add_angles_constants() -> Hugr {
    let qb_row = vec![QB_T];
    let mut h = DFGBuilder::new(FunctionType::new(qb_row.clone(), qb_row)).unwrap();

    let qb = h.input_wires().next().unwrap();

    let point2 = h.add_load_value(ConstF64::new(0.2));
    let point3 = h.add_load_value(ConstF64::new(0.3));
    let point5 = h
        .add_dataflow_op(Tk2Op::AngleAdd, [point2, point3])
        .unwrap()
        .out_wire(0);

    let qbs = h
        .add_dataflow_op(Tk2Op::RxF64, [qb, point5])
        .unwrap()
        .outputs();
    h.finish_hugr_with_outputs(qbs, &REGISTRY).unwrap()
}

#[rstest]
#[case::symbolic(circ_add_angles_symbolic(), "f0 + f1")]
#[case::constants(circ_add_angles_constants(), "0.2 + 0.3")]
fn test_add_angle_serialise(#[case] circ_add_angles: Hugr, #[case] param_str: &str) {
    let ser: SerialCircuit = SerialCircuit::encode(&circ_add_angles).unwrap();
    assert_eq!(ser.commands.len(), 1);
    assert_eq!(ser.commands[0].op.op_type, optype::OpType::Rx);
    assert_eq!(ser.commands[0].op.params, Some(vec![param_str.into()]));

    // Note: this is not a proper roundtrip as the symbols f0 and f1 are not
    // converted back to circuit inputs. This would require parsing symbolic
    // expressions.
    let deser: Hugr = ser.clone().decode().unwrap();
    let reser = SerialCircuit::encode(&deser).unwrap();
    compare_serial_circs(&ser, &reser);
}

fn compare_serial_circs(a: &SerialCircuit, b: &SerialCircuit) {
    assert_eq!(a.name, b.name);
    assert_eq!(a.phase, b.phase);

    let qubits_a: Vec<_> = a.qubits.iter().collect();
    let qubits_b: Vec<_> = b.qubits.iter().collect();
    assert_eq!(qubits_a, qubits_b);

    let bits_a: Vec<_> = a.bits.iter().collect();
    let bits_b: Vec<_> = b.bits.iter().collect();
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
