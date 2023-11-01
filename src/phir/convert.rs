use std::{collections::HashMap, str::FromStr};

use super::model::{Bit, PHIRModel};
use crate::{
    circuit::Command,
    phir::model::{COpArg, CVarDefine, Data, ExportVar, Metadata, QOp, QOpArg, QVarDefine},
    Circuit, T2Op,
};
use derive_more::From;
use hugr::{
    extension::prelude::QB_T,
    hugr::CircuitUnit,
    ops::{custom::ExternalOp, LeafOp, OpTag, OpTrait, OpType},
    std_extensions::arithmetic::int_types::INT_TYPES,
    Wire,
};
use itertools::{Either, Itertools};
use strum_macros::{EnumIter, EnumString, IntoStaticStr};
use thiserror::Error;

/// Convert Circuit-like HUGR to PHIR.
pub fn circuit_to_phir(circ: &impl Circuit) -> Result<PHIRModel, &'static str> {
    let mut ph = PHIRModel::new();

    const QUBIT_ID: &str = "q";
    let q_arg = |index| (QUBIT_ID.to_string(), index as u64);

    let mut qubit_count = 0;
    let mut input_int_count = 0;
    let mut measure_exports = vec![];
    let _input_map: HashMap<usize, COpArg> = circ
        .units()
        .enumerate()
        .map(|(index, (wire, _, t))| match (wire, t) {
            (CircuitUnit::Wire(_), t) if t == INT_TYPES[6] => {
                let variable = format!("i{input_int_count}");
                let cvar_def: Data = Data {
                    data: CVarDefine {
                        data_type: "i64".to_string(),
                        variable: variable.clone(),
                        size: None,
                    }
                    .into(),
                    metadata: Metadata::default(),
                };
                input_int_count += 1;
                ph.append_op(cvar_def);
                (index, COpArg::Sym(variable))
            }
            (CircuitUnit::Linear(id), t) if t == QB_T => {
                qubit_count += 1;
                (index, COpArg::Bit(q_arg(id)))
            }
            _ => unimplemented!("Non-int64 input wires not supported"),
        })
        .collect();

    let qvar_def: Data = Data {
        data: QVarDefine {
            data_type: Some("qubits".to_string()),
            variable: "q".to_string(),
            size: qubit_count,
        }
        .into(),
        metadata: Metadata::default(),
    };
    ph.append_op(qvar_def);

    // Define quantum and classical variables for inputs

    // Add commands
    for com in circ.commands() {
        let optype = com.optype();
        // at the end take each output wire and export it
        // skip non-quantum ops
        // for integer in-wire, recursively convert to Cop
        // if the wire is one of many, assign a variable for it and log to wire
        // map
        // if a constant or arg from wire map encountered, use that directly.
        let qop = match t2op_name(optype) {
            Ok(qop) => qop.to_string(),
            Err(OpConvertError::Skip) => continue,
            Err(OpConvertError::Other(s)) => return Err(s),
        };

        let args: Vec<Bit> = com
            .inputs()
            .map(|(u, _, _)| match u {
                CircuitUnit::Wire(_) => todo!(),
                CircuitUnit::Linear(i) => q_arg(i),
            })
            .collect();

        let args: Vec<QOpArg> = if args.len() == 1 {
            let [arg]: [Bit; 1] = args.try_into().unwrap();
            vec![QOpArg::Bit(arg)]
        } else {
            vec![QOpArg::ListBit(args)]
        };
        // TODO measure, define output reg and record in "returns"
        let returns = if qop == "Measure" {
            let (bit, def, export) = measure_out_arg(com, circ);
            ph.insert_op(0, def);
            if let Some(export) = export {
                measure_exports.push(export);
            }
            // measures.insert(measure_wire, arg.clone());

            Some(vec![bit])
        } else {
            None
        };
        let phir_op = crate::phir::model::Op {
            op_enum: QOp { qop, args, returns }.into(),
            metadata: Metadata::default(),
        };

        ph.append_op(phir_op);
    }

    for export in measure_exports {
        ph.append_op(export);
    }
    // Add DFG as SeqBlock

    // Add conditional as IfBlock

    // Add wasm calls

    // export measured variables

    Ok(ph)
}

fn measure_out_arg(
    com: Command<'_, impl Circuit>,
    circ: &impl Circuit,
) -> (Bit, Data, Option<Data>) {
    let output = circ.get_io(circ.root()).expect("missing io")[1];

    let (wires, qb_indices): (Vec<_>, Vec<_>) = com.outputs().partition_map(|(c, _, _)| match c {
        CircuitUnit::Wire(w) => Either::Left(w),
        CircuitUnit::Linear(i) => Either::Right(i),
    });

    let [measure_wire]: [Wire; 1] = wires
        .try_into()
        .expect("Should only be one classical wire from measure.");
    let [qb_index]: [usize; 1] = qb_indices
        .try_into()
        .expect("Should only be one quantum wire from measure.");

    // variable name marked with qubit index being measured
    let variable = format!("c{}", qb_index);

    // declare a width-1 register per measurement
    // TODO what if qubit measured multiple times?
    let c_var_def: Data = Data {
        data: CVarDefine {
            data_type: "i64".to_string(),
            variable: variable.clone(),
            size: Some(1),
        }
        .into(),
        metadata: Metadata::default(),
    };

    let arg = (variable.clone(), 0);

    let export: Option<Data> = if circ
        .linked_ports(measure_wire.node(), measure_wire.source())
        .any(|(n, _)| n == output)
    {
        // if the measured value is output, export it
        // TODO generalise to export any intermediates connected to output.
        Some(Data {
            data: ExportVar {
                variables: vec![variable],
                to: None,
            }
            .into(),
            metadata: Metadata::default(),
        })
    } else {
        None
    };
    (arg, c_var_def, export)
}

// TODO: function to generate PHIR expression tree from classical input wire.
// TODO: constant folding angles

#[derive(From)]
enum OpConvertError {
    Skip,
    Other(&'static str),
}

/// Get the PHIR name for a quantum operation
fn t2op_name(op: &OpType) -> Result<&'static str, OpConvertError> {
    dbg!(op);
    let err = Err(OpConvertError::Other("Unknown op"));
    if let OpTag::Const | OpTag::LoadConst = op.tag() {
        return Err(OpConvertError::Skip);
    }
    let OpType::LeafOp(leaf) = op else {
        return err;
    };

    if let Ok(t2op) = leaf.try_into() {
        // https://github.com/CQCL/phir/blob/main/phir_spec_qasm.md
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
    } else if let Ok(phir_cop) = leaf.try_into() {
        let phir_cop: PhirCop = phir_cop;
        Ok(match phir_cop {
            PhirCop::Add => "+",
            PhirCop::Sub => "-",
            PhirCop::Mul => "*",
            PhirCop::Div => "/",
            PhirCop::Mod => "%",
            PhirCop::Eq => "==",
            PhirCop::Neq => "!=",
            PhirCop::Gt => ">",
            PhirCop::Lt => "<",
            PhirCop::Ge => ">=",
            PhirCop::Le => "<=",
            PhirCop::And => "&",
            PhirCop::Or => "|",
            PhirCop::Xor => "^",
            PhirCop::Not => "~",
            PhirCop::Lsh => "<<",
            PhirCop::Rsh => ">>",
        })
    } else {
        // TODO arithmetic
        Err(OpConvertError::Skip)
    }
}

#[derive(
    Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, EnumIter, IntoStaticStr, EnumString,
)]
enum PhirCop {
    #[strum(serialize = "iadd")]
    Add,
    #[strum(serialize = "isub")]
    Sub,
    #[strum(serialize = "imul")]
    Mul,
    #[strum(serialize = "idiv")]
    Div,
    #[strum(serialize = "imod_s")]
    Mod,
    #[strum(serialize = "ieq")]
    Eq,
    #[strum(serialize = "ine")]
    Neq,
    #[strum(serialize = "ilt_s")]
    Gt,
    #[strum(serialize = "igt_s")]
    Lt,
    #[strum(serialize = "ige_s")]
    Ge,
    #[strum(serialize = "ile_s")]
    Le,
    #[strum(serialize = "iand")]
    And,
    #[strum(serialize = "ior")]
    Or,
    #[strum(serialize = "ixor")]
    Xor,
    #[strum(serialize = "inot")]
    Not,
    #[strum(serialize = "ishl")]
    Lsh,
    #[strum(serialize = "ishr")]
    Rsh,
}

#[derive(Error, Debug, Clone)]
#[error("Not a Phir classical op.")]
struct NotPhirCop;

impl TryFrom<OpType> for PhirCop {
    type Error = NotPhirCop;

    fn try_from(op: OpType) -> Result<Self, Self::Error> {
        Self::try_from(&op)
    }
}

impl TryFrom<&OpType> for PhirCop {
    type Error = NotPhirCop;

    fn try_from(op: &OpType) -> Result<Self, Self::Error> {
        let OpType::LeafOp(leaf) = op else {
            return Err(NotPhirCop);
        };
        leaf.try_into()
    }
}

impl TryFrom<&LeafOp> for PhirCop {
    type Error = NotPhirCop;

    fn try_from(op: &LeafOp) -> Result<Self, Self::Error> {
        match op {
            LeafOp::CustomOp(b) => {
                let name = match b.as_ref() {
                    ExternalOp::Extension(e) => e.def().name(),
                    ExternalOp::Opaque(o) => o.name(),
                };

                PhirCop::from_str(name).map_err(|_| NotPhirCop)
            }
            _ => Err(NotPhirCop),
        }
    }
}

impl TryFrom<LeafOp> for PhirCop {
    type Error = NotPhirCop;

    fn try_from(op: LeafOp) -> Result<Self, Self::Error> {
        Self::try_from(&op)
    }
}
#[cfg(test)]
mod test {

    use hugr::Hugr;
    use rstest::{fixture, rstest};

    use crate::utils::build_simple_measure_circuit;

    use super::*;

    #[fixture]
    // A commutation forward exists but depth doesn't change
    fn sample() -> Hugr {
        build_simple_measure_circuit(2, 2, |circ| {
            circ.append(T2Op::H, [1])?;
            circ.append(T2Op::CX, [0, 1])?;
            circ.append(T2Op::Z, [0])?;
            circ.append(T2Op::X, [1])?;
            let mut c0 = circ.append_with_outputs(T2Op::Measure, [0])?;
            let c1 = circ.append_with_outputs(T2Op::Measure, [1])?;
            c0.extend(c1);
            Ok(c0)
        })
        .unwrap()
    }
    #[rstest]
    fn test_sample(sample: Hugr) {
        // rmp_serde::encode::write(&mut File::create("sample.hugr").unwrap(), &sample).unwrap();
        let ph = circuit_to_phir(&sample).unwrap();
        assert_eq!(ph.num_ops(), 11);
    }
}
