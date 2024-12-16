//! General tests.

use std::collections::{HashMap, HashSet};
use std::io::BufReader;

use hugr::builder::{DFGBuilder, Dataflow, DataflowHugr};
use hugr::extension::prelude::{bool_t, qb_t};

use hugr::hugr::hugrmut::HugrMut;
use hugr::std_extensions::arithmetic::float_ops::FloatOps;
use hugr::types::Signature;
use hugr::HugrView;
use rstest::{fixture, rstest};
use tket_json_rs::circuit_json::{self, SerialCircuit};
use tket_json_rs::optype;
use tket_json_rs::register;

use super::{TKETDecode, METADATA_INPUT_PARAMETERS, METADATA_Q_OUTPUT_REGISTERS};
use crate::circuit::Circuit;
use crate::extension::rotation::{rotation_type, ConstRotation, RotationOp};
use crate::extension::sympy::SympyOpDef;
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

const MULTI_REGISTER: &str = r#"{
        "phase": "0",
        "bits": [],
        "qubits": [["q", [2]], ["q", [1]], ["my_qubits", [2]]],
        "commands": [
            {"args": [["my_qubits", [2]]], "op": {"type": "H"}},
            {"args": [["q", [2]], ["q", [1]]], "op": {"type": "CX"}}
        ],
        "implicit_permutation": []
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

const PARAMETERIZED: &str = r#"{
        "phase": "0.0",
        "bits": [],
        "qubits": [["q", [0]], ["q", [1]]],
        "commands": [
            {"args":[["q",[0]]],"op":{"type":"H"}},
            {"args":[["q",[1]],["q",[0]]],"op":{"type":"CX"}},
            {"args":[["q",[0]]],"op":{"params":["(1.5707963267948966 / pi)"],"type":"Rz"}},
            {"args": [["q", [0]]], "op": {"params": ["(3.141596 / pi)", "alpha", "(0.7853981633974483 / pi)"], "type": "TK1"}}
        ],
        "created_qubits": [],
        "discarded_qubits": [],
        "implicit_permutation": [[["q", [0]], ["q", [0]]], [["q", [1]], ["q", [1]]]]
    }"#;

/// Check some properties of the serial circuit.
fn validate_serial_circ(circ: &SerialCircuit) {
    // Check that all commands have valid arguments.
    for command in &circ.commands {
        for arg in &command.args {
            assert!(
                circ.qubits.contains(&register::Qubit::from(arg.clone()))
                    || circ.bits.contains(&register::Bit::from(arg.clone())),
                "Circuit command {command:?} has an invalid argument '{arg:?}'"
            );
        }
    }

    // Check that the implicit permutation is valid.
    let perm: HashMap<register::ElementId, register::ElementId> = circ
        .implicit_permutation
        .iter()
        .map(|p| (p.0.clone().id, p.1.clone().id))
        .collect();
    for (key, value) in &perm {
        let valid_qubits = circ.qubits.contains(&register::Qubit::from(key.clone()))
            && circ.qubits.contains(&register::Qubit::from(value.clone()));
        let valid_bits = circ.bits.contains(&register::Bit::from(key.clone()))
            && circ.bits.contains(&register::Bit::from(value.clone()));
        assert!(
            valid_qubits || valid_bits,
            "Circuit has an invalid permutation '{key:?} -> {value:?}'"
        );
    }
    assert_eq!(
        perm.len(),
        circ.implicit_permutation.len(),
        "Circuit has duplicate permutations",
    );
    assert_eq!(
        HashSet::<&register::ElementId>::from_iter(perm.values()).len(),
        perm.len(),
        "Circuit has duplicate values in permutations"
    );
}

fn compare_serial_circs(a: &SerialCircuit, b: &SerialCircuit) {
    assert_eq!(a.name, b.name);
    assert_eq!(a.phase, b.phase);
    assert_eq!(&a.qubits, &b.qubits);
    assert_eq!(&a.bits, &b.bits);
    assert_eq!(a.commands.len(), b.commands.len());

    // This comparison only works if both serial circuits share a topological
    // ordering of commands.
    //
    // We also cannot compare the arguments directly, since we may permute them
    // internally.
    //
    // TODO: Do a proper comparison independent of the toposort ordering, and
    // track register reordering.
    for (a, b) in a.commands.iter().zip(b.commands.iter()) {
        assert_eq!(a.op.op_type, b.op.op_type);
        assert_eq!(a.op.params, b.op.params);
        assert_eq!(a.args.len(), b.args.len());
    }
}

/// A simple circuit with some preset qubit registers
#[fixture]
fn circ_preset_qubits() -> Circuit {
    let input_t = vec![qb_t()];
    let output_t = vec![qb_t(), qb_t()];
    let mut h = DFGBuilder::new(Signature::new(input_t, output_t)).unwrap();

    let [qb0] = h.input_wires_arr();
    let [qb1] = h.add_dataflow_op(Tk2Op::QAlloc, []).unwrap().outputs_arr();

    let [qb0, qb1] = h
        .add_dataflow_op(Tk2Op::CZ, [qb0, qb1])
        .unwrap()
        .outputs_arr();

    let mut hugr = h.finish_hugr_with_outputs([qb0, qb1]).unwrap();

    // A preset register for the first qubit output
    hugr.set_metadata(
        hugr.root(),
        METADATA_Q_OUTPUT_REGISTERS,
        serde_json::json!([["q", [1]]]),
    );

    hugr.into()
}

/// A simple circuit with some input parameters
#[fixture]
fn circ_parameterized() -> Circuit {
    let input_t = vec![qb_t(), rotation_type(), rotation_type(), rotation_type()];
    let output_t = vec![qb_t()];
    let mut h = DFGBuilder::new(Signature::new(input_t, output_t)).unwrap();

    let [q, r0, r1, r2] = h.input_wires_arr();

    let [q] = h.add_dataflow_op(Tk2Op::Rx, [q, r0]).unwrap().outputs_arr();
    let [q] = h.add_dataflow_op(Tk2Op::Ry, [q, r1]).unwrap().outputs_arr();
    let [q] = h.add_dataflow_op(Tk2Op::Rz, [q, r2]).unwrap().outputs_arr();

    let mut hugr = h.finish_hugr_with_outputs([q]).unwrap();

    // Preset names for some of the inputs
    hugr.set_metadata(
        hugr.root(),
        METADATA_INPUT_PARAMETERS,
        serde_json::json!(["alpha", "beta"]),
    );

    hugr.into()
}

/// A simple circuit with ancillae
#[fixture]
fn circ_measure_ancilla() -> Circuit {
    let input_t = vec![qb_t()];
    let output_t = vec![bool_t(), bool_t()];
    let mut h = DFGBuilder::new(Signature::new(input_t, output_t)).unwrap();

    let [qb] = h.input_wires_arr();
    let [anc] = h.add_dataflow_op(Tk2Op::QAlloc, []).unwrap().outputs_arr();

    let [qb, meas_qb] = h
        .add_dataflow_op(Tk2Op::Measure, [qb])
        .unwrap()
        .outputs_arr();
    let [anc, meas_anc] = h
        .add_dataflow_op(Tk2Op::Measure, [anc])
        .unwrap()
        .outputs_arr();

    let [] = h.add_dataflow_op(Tk2Op::QFree, [qb]).unwrap().outputs_arr();
    let [] = h
        .add_dataflow_op(Tk2Op::QFree, [anc])
        .unwrap()
        .outputs_arr();

    h.finish_hugr_with_outputs([meas_qb, meas_anc])
        .unwrap()
        .into()
}

#[fixture]
fn circ_add_angles_symbolic() -> (Circuit, String) {
    let input_t = vec![qb_t(), rotation_type(), rotation_type()];
    let output_t = vec![qb_t()];
    let mut h = DFGBuilder::new(Signature::new(input_t, output_t)).unwrap();

    let [qb, f1, f2] = h.input_wires_arr();
    let [f12] = h
        .add_dataflow_op(RotationOp::radd, [f1, f2])
        .unwrap()
        .outputs_arr();
    let [qb] = h
        .add_dataflow_op(Tk2Op::Rx, [qb, f12])
        .unwrap()
        .outputs_arr();

    let circ = h.finish_hugr_with_outputs([qb]).unwrap().into();
    (circ, "(f0 + f1)".to_string())
}

#[fixture]
fn circ_add_angles_constants() -> (Circuit, String) {
    let qb_row = vec![qb_t()];
    let mut h = DFGBuilder::new(Signature::new(qb_row.clone(), qb_row)).unwrap();

    let qb = h.input_wires().next().unwrap();

    let point2 = h.add_load_value(ConstRotation::new(0.2).unwrap());
    let point3 = h.add_load_value(ConstRotation::new(0.3).unwrap());
    let point5 = h
        .add_dataflow_op(RotationOp::radd, [point2, point3])
        .unwrap()
        .out_wire(0);

    let qbs = h
        .add_dataflow_op(Tk2Op::Rx, [qb, point5])
        .unwrap()
        .outputs();
    let circ = h.finish_hugr_with_outputs(qbs).unwrap().into();
    (circ, "(0.2 + 0.3)".to_string())
}

#[fixture]
/// An Rx operation using some complex ops to compute its angle.
fn circ_complex_angle_computation() -> (Circuit, String) {
    let input_t = vec![qb_t(), rotation_type(), rotation_type()];
    let output_t = vec![qb_t()];
    let mut h = DFGBuilder::new(Signature::new(input_t, output_t)).unwrap();

    let [qb, r0, r1] = h.input_wires_arr();

    // Loading rotations and sympy expressions
    let point2 = h.add_load_value(ConstRotation::new(0.2).unwrap());
    let sympy = h
        .add_dataflow_op(SympyOpDef.with_expr("cos(pi)".to_string()), [])
        .unwrap()
        .out_wire(0);
    let added_rot = h
        .add_dataflow_op(RotationOp::radd, [sympy, point2])
        .unwrap()
        .out_wire(0);

    // Float operations and conversions
    let f0 = h
        .add_dataflow_op(RotationOp::to_halfturns, [r0])
        .unwrap()
        .out_wire(0);
    let f1 = h
        .add_dataflow_op(RotationOp::to_halfturns, [r1])
        .unwrap()
        .out_wire(0);
    let fpow = h
        .add_dataflow_op(FloatOps::fpow, [f0, f1])
        .unwrap()
        .out_wire(0);
    let rpow = h
        .add_dataflow_op(RotationOp::from_halfturns_unchecked, [fpow])
        .unwrap()
        .out_wire(0);

    let final_rot = h
        .add_dataflow_op(RotationOp::radd, [rpow, added_rot])
        .unwrap()
        .out_wire(0);

    let qbs = h
        .add_dataflow_op(Tk2Op::Rx, [qb, final_rot])
        .unwrap()
        .outputs();

    let circ = h.finish_hugr_with_outputs(qbs).unwrap().into();
    (circ, "((f0 ** f1) + (cos(pi) + 0.2))".to_string())
}

#[rstest]
#[case::simple(SIMPLE_JSON, 2, 2)]
#[case::simple(MULTI_REGISTER, 2, 3)]
#[case::unknown_op(UNKNOWN_OP, 2, 3)]
#[case::parametrized(PARAMETERIZED, 4, 2)]
fn json_roundtrip(#[case] circ_s: &str, #[case] num_commands: usize, #[case] num_qubits: usize) {
    let ser: circuit_json::SerialCircuit = serde_json::from_str(circ_s).unwrap();
    assert_eq!(ser.commands.len(), num_commands);

    let circ: Circuit = ser.clone().decode().unwrap();

    assert_eq!(circ.qubit_count(), num_qubits);

    let reser: SerialCircuit = SerialCircuit::encode(&circ).unwrap();
    validate_serial_circ(&reser);
    compare_serial_circs(&ser, &reser);
}

#[rstest]
#[cfg_attr(miri, ignore)] // Opening files is not supported in (isolated) miri
#[case::barenco_tof_10("../test_files/barenco_tof_10.json")]
fn json_file_roundtrip(#[case] circ: impl AsRef<std::path::Path>) {
    let reader = BufReader::new(std::fs::File::open(circ).unwrap());
    let ser: circuit_json::SerialCircuit = serde_json::from_reader(reader).unwrap();
    let circ: Circuit = ser.clone().decode().unwrap();
    let reser: SerialCircuit = SerialCircuit::encode(&circ).unwrap();
    validate_serial_circ(&reser);
    compare_serial_circs(&ser, &reser);
}

/// Test the serialisation roundtrip from a tket2 circuit.
///
/// Note: this is not a pure roundtrip as the encoder may add internal qubits/bits to the circuit.
#[rstest]
#[case::meas_ancilla(circ_measure_ancilla(), Signature::new_endo(vec![qb_t(), qb_t(), bool_t(), bool_t()]))]
#[case::preset_qubits(circ_preset_qubits(), Signature::new_endo(vec![qb_t(), qb_t(), qb_t()]))]
#[case::preset_parameterized(circ_parameterized(), Signature::new(vec![qb_t(), rotation_type(), rotation_type(), rotation_type()], vec![qb_t()]))]
fn circuit_roundtrip(#[case] circ: Circuit, #[case] decoded_sig: Signature) {
    let ser: SerialCircuit = SerialCircuit::encode(&circ).unwrap();
    let deser: Circuit = ser.clone().decode().unwrap();

    let deser_sig = deser.circuit_signature();
    assert_eq!(
        &deser_sig.input, &decoded_sig.input,
        "Input signature mismatch\n  Expected: {}\n  Actual:   {}",
        &deser_sig, &decoded_sig
    );
    assert_eq!(
        &deser_sig.output, &decoded_sig.output,
        "Output signature mismatch\n  Expected: {}\n  Actual:   {}",
        &deser_sig, &decoded_sig
    );

    let reser = SerialCircuit::encode(&deser).unwrap();
    validate_serial_circ(&reser);
    compare_serial_circs(&ser, &reser);
}

/// Test serialisation of circuits with a symbolic expression.
///
/// Note: this is not a proper roundtrip as the symbols f0 and f1 are not
/// converted back to circuit inputs. This would require parsing symbolic
/// expressions.
#[rstest]
#[case::symbolic(circ_add_angles_symbolic())]
#[case::constants(circ_add_angles_constants())]
#[case::complex(circ_complex_angle_computation())]
fn test_add_angle_serialise(#[case] circ_add_angles: (Circuit, String)) {
    let (circ, expected) = circ_add_angles;

    let ser: SerialCircuit = SerialCircuit::encode(&circ).unwrap();
    assert_eq!(ser.commands.len(), 1);
    assert_eq!(ser.commands[0].op.op_type, optype::OpType::Rx);
    assert_eq!(ser.commands[0].op.params, Some(vec![expected]));

    let deser: Circuit = ser.clone().decode().unwrap();
    let reser = SerialCircuit::encode(&deser).unwrap();
    validate_serial_circ(&reser);
    compare_serial_circs(&ser, &reser);
}
