use hugr::{
    ops::{LeafOp, OpType},
    Hugr, HugrView,
};

use crate::{extension::try_unwrap_json_op, Circuit, T2Op};

use super::PHIRModel;

pub fn phir_to_hugr(circ: &impl Circuit) -> Result<PHIRModel, &'static str> {
    let ph = PHIRModel::new();

    // Define quantum and classical variables for inputs

    // Add commands
    for com in circ.commands() {
        let optype = com.optype();
        // filter to only operations that involve qubits or output.
        // classical operations with multiple outputs?
        let name = t2op_name(optype)?;
    }
    // Add DFG as SeqBlock

    // Add conditional as IfBlock

    // Add wasm calls

    // export measured variables

    Ok(ph)
}

// TODO: function to generate PHIR expression tree from classical input wire.
// TODO: constant folding angles

/// Get the PHIR name for a quantum operation
fn t2op_name(t2op: &OpType) -> Result<&'static str, &'static str> {
    let err = Err("Unknown op");
    let OpType::LeafOp(leaf) = t2op else {
        return err;
    };

    if let Ok(t2op) = leaf.clone().try_into() {
        Ok(match t2op {
            T2Op::H => "H",
            T2Op::CX => "CX",
            T2Op::T => "T",
            T2Op::S => "SZ",
            T2Op::X => "X",
            T2Op::Y => "Y",
            T2Op::Z => "Z",
            T2Op::Tdg => "Tdg",
            T2Op::Sdg => "SZdg",
            T2Op::ZZMax => "SZZ",
            T2Op::Measure => "Measure",
            T2Op::RzF64 => "RZ",
            T2Op::RxF64 => "RX",
            T2Op::PhasedX => "R1XY",
            T2Op::ZZPhase => "RZZ",
            T2Op::CZ => "CZ",
            T2Op::AngleAdd | T2Op::TK1 => return err,
        })
    } else {
        err
    }
}
