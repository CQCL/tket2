use std::{collections::HashMap, str::FromStr};

use super::model::{Bit, PHIRModel};
use crate::{
    circuit::Command,
    phir::model::{
        COp, COpArg, CVarDefine, CopReturn, Data, ExportVar, Metadata, QOp, QOpArg, QVarDefine,
    },
    Circuit, T2Op,
};
use derive_more::From;
use hugr::{
    extension::prelude::{BOOL_T, QB_T},
    ops::{custom::ExternalOp, Const, LeafOp, OpTag, OpTrait, OpType},
    std_extensions::arithmetic::{
        float_types::ConstF64,
        int_types::{ConstIntS, INT_TYPES},
    },
    types::{EdgeKind, TypeEnum},
    values::{CustomConst, PrimValue, Value},
    CircuitUnit, Node, Wire,
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
    let mut arg_map: HashMap<Wire, COpArg> = circ
        .units()
        .filter_map(|(cu, _, t)| match (cu, t) {
            (CircuitUnit::Wire(wire), t) if t == INT_TYPES[6] => {
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
                Some((wire, COpArg::Sym(variable)))
            }
            (CircuitUnit::Linear(_), t) if t == QB_T => {
                qubit_count += 1;
                None
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
            Ok(PhirOp::QOp(s)) => s.to_string(),
            Ok(_) => continue,
            Err(s) => return Err(s),
        };
        let mut angles = vec![];
        let args: Vec<Bit> = com
            .inputs()
            .filter_map(|(u, _, _)| match u {
                CircuitUnit::Wire(w) => {
                    // let cop_arg: COpArg = gen_c_op(circ, w, &mut ph);

                    // arg_map.insert(w, cop_arg);
                    let angle: ConstF64 = get_value(get_const(w, circ).unwrap()).unwrap();
                    angles.push(angle.value());
                    None
                }
                CircuitUnit::Linear(i) => Some(q_arg(i)),
            })
            .collect();

        let args: Vec<QOpArg> = if args.len() == 1 {
            let [arg]: [Bit; 1] = args.try_into().unwrap();
            vec![QOpArg::Bit(arg)]
        } else {
            vec![QOpArg::ListBit(args)]
        };

        let returns = if qop == "Measure" {
            let (bit, wire) = measure_out_arg(com);
            let def = def_int_var(bit.0.clone(), 1);

            ph.insert_op(0, def);
            arg_map.insert(wire, COpArg::Sym(bit.0.clone()));

            Some(vec![bit])
        } else {
            None
        };
        let phir_op = crate::phir::model::Op {
            op_enum: QOp {
                qop,
                args,
                returns,
                angles: (!angles.is_empty()).then_some((angles.clone(), "rad".to_string())),
            }
            .into(),
            // TODO once PECOS no longer requires angles in the metadata
            metadata: match angles.len() {
                0 => Metadata::default(),
                1 => Metadata::from_iter([("angle".to_string(), angles[0].into())]),
                _ => Metadata::from_iter([("angles".to_string(), angles.into())]),
            },
        };

        ph.append_op(phir_op);
    }
    // get all classical output wires
    let c_out_wires = in_neighbour_wires(circ, circ.output()).filter(|wire| {
        match circ.get_optype(wire.node()).port_kind(wire.source()) {
            Some(EdgeKind::Value(t)) => {
                t == BOOL_T
                    // Ignore sums and tuples
                        || !matches!(t.as_type_enum(), TypeEnum::Sum(_) | TypeEnum::Tuple(_))
            }
            _ => false,
        }
    });
    let mut temp_var_count = 0..;

    for wire in c_out_wires {
        let Some(variable) = get_c_op(&arg_map, circ, wire) else {
            continue;
        };

        //  = arg_map.remove(&wire).ok_or("Missing output variable")?;
        let variable = match variable {
            COpArg::Sym(s) => s,
            COpArg::Bit((s, _)) => s,
            _ => {
                let out_var_name = format!("__temp{}", temp_var_count.next().unwrap());

                // TODO expand to 64?
                let def = def_int_var(out_var_name.clone(), 32);
                let assign = crate::phir::model::Op {
                    op_enum: COp {
                        cop: "=".to_string(),
                        args: vec![variable],
                        returns: Some(vec![CopReturn::Sym(out_var_name.clone())]),
                    }
                    .into(),
                    metadata: Metadata::default(),
                };
                ph.append_op(def);
                ph.append_op(assign);

                out_var_name
            } // _ => return Err("Invalid export."),
        };
        let export = Data {
            data: ExportVar {
                variables: vec![variable],
                to: None,
            }
            .into(),
            metadata: Metadata::default(),
        };
        ph.append_op(export);
    }

    // Add DFG as SeqBlock

    // Add conditional as IfBlock

    // Add wasm calls

    // export measured variables

    Ok(ph)
}

fn in_neighbour_wires(circ: &impl Circuit, node: Node) -> impl Iterator<Item = Wire> + '_ {
    let node_type = circ.get_optype(node);
    circ.node_inputs(node)
        .filter(|port| {
            node_type.port_kind(*port).is_some_and(|k| match k {
                EdgeKind::Value(t) => t.copyable(),
                _ => false,
            })
        })
        .flat_map(move |port| circ.linked_ports(node, port))
        .map(|(n, p)| Wire::new(n, p))
}

fn get_c_op(arg_map: &HashMap<Wire, COpArg>, circ: &impl Circuit, wire: Wire) -> Option<COpArg> {
    if let Some(cop) = arg_map.get(&wire) {
        return Some(cop.clone());
    }

    if let Some(c) = get_const(wire, circ) {
        return Some(if c == Const::true_val() {
            COpArg::IntValue(1)
        } else if c == Const::false_val() {
            COpArg::IntValue(0)
        } else if let Some(int) = get_value::<ConstIntS>(c) {
            COpArg::IntValue(int.value())
        } else {
            panic!("Unknown constant.");
        });
    }
    if let Ok(PhirOp::Cop(cop)) = t2op_name(circ.get_optype(wire.node())) {
        Some(COpArg::COp(COp {
            cop: cop.phir_name().to_string(),
            args: in_neighbour_wires(circ, wire.node())
                .flat_map(|prev_wire| get_c_op(arg_map, circ, prev_wire))
                .collect(),
            returns: None,
        }))
    } else {
        None
    }
}

fn measure_out_arg(com: Command<'_, impl Circuit>) -> (Bit, Wire) {
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

    let arg = (variable.clone(), 0);

    (arg, measure_wire)
}

fn def_int_var(variable: String, size: u64) -> Data {
    Data {
        data: CVarDefine {
            data_type: "i64".to_string(),
            variable,
            size: Some(size),
        }
        .into(),
        metadata: Metadata::default(),
    }
}

fn get_const(wire: Wire, circ: &impl Circuit) -> Option<Const> {
    if circ.get_optype(wire.node()).tag() != OpTag::LoadConst {
        return None;
    }

    circ.input_neighbours(wire.node()).find_map(|n| {
        let const_op = circ.get_optype(n);

        const_op.clone().try_into().ok()
    })
}

fn get_value<T: CustomConst>(op: Const) -> Option<T> {
    // impl<T: CustomConst> TryFrom<Value> for T in Hugr crate
    if let Value::Prim {
        val: PrimValue::Extension { c: (custom,) },
    } = op.value()
    {
        let c: T = *(custom.clone()).downcast().ok()?;
        Some(c)
    } else {
        None
    }
}

// TODO: function to generate PHIR expression tree from classical input wire.
// TODO: constant folding angles

#[derive(From)]
enum OpConvertError {
    Skip,
    Other(&'static str),
}

enum PhirOp {
    QOp(&'static str),
    Cop(PhirCop),
    Skip,
}

/// Get the PHIR name for a quantum operation
fn t2op_name(op: &OpType) -> Result<PhirOp, &'static str> {
    let err = Err("Unknown op");
    if let OpTag::Const | OpTag::LoadConst = op.tag() {
        return Ok(PhirOp::Skip);
    }
    let OpType::LeafOp(leaf) = op else {
        return err;
    };

    if let Ok(t2op) = leaf.try_into() {
        // https://github.com/CQCL/phir/blob/main/phir_spec_qasm.md
        Ok(PhirOp::QOp(match t2op {
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
        }))
    } else if let Ok(phir_cop) = leaf.try_into() {
        Ok(PhirOp::Cop(phir_cop))
    } else {
        match leaf {
            LeafOp::Tag { .. } | LeafOp::MakeTuple { .. } | LeafOp::UnpackTuple { .. } => {
                Ok(PhirOp::Skip)
            }
            _ => err,
        }
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

impl PhirCop {
    fn phir_name(&self) -> &'static str {
        match self {
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
        }
    }
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

    use hugr::{
        builder::{BuildError, DFGBuilder, Dataflow, DataflowHugr},
        extension::{prelude::BOOL_T, ExtensionSet},
        std_extensions::arithmetic::float_types::{ConstF64, EXTENSION_ID},
        types::FunctionType,
        Hugr,
    };
    use rstest::{fixture, rstest};

    use crate::extension::REGISTRY;

    use super::*;

    #[fixture]
    // A commutation forward exists but depth doesn't change
    fn sample() -> Hugr {
        {
            let num_qubits = 2;
            let num_measured_bools = 2;
            let inputs = vec![QB_T; num_qubits];
            let outputs = [inputs.clone(), vec![BOOL_T; num_measured_bools]].concat();

            let mut h = DFGBuilder::new(FunctionType::new(inputs, outputs)).unwrap();
            let angle_const = ConstF64::new(1.2);

            let angle = h
                .add_load_const(angle_const.into(), ExtensionSet::from_iter([EXTENSION_ID]))
                .unwrap();
            let qbs = h.input_wires();

            let mut circ = h.as_circuit(qbs.into_iter().collect());

            let o: Result<Vec<Wire>, BuildError> = (|| {
                circ.append(T2Op::H, [1])?;
                circ.append(T2Op::CX, [0, 1])?;
                circ.append(T2Op::Z, [0])?;
                circ.append(T2Op::X, [1])?;
                circ.append_and_consume(
                    T2Op::RzF64,
                    [CircuitUnit::Linear(0), CircuitUnit::Wire(angle)],
                )?;
                let mut c0 = circ.append_with_outputs(T2Op::Measure, [0])?;
                let c1 = circ.append_with_outputs(T2Op::Measure, [1])?;
                c0.extend(c1);
                Ok(c0)
            })();
            let o = o.unwrap();

            let qbs = circ.finish();
            h.finish_hugr_with_outputs([qbs, o].concat(), &REGISTRY)
        }
        .unwrap()
    }
    #[rstest]
    fn test_sample(sample: Hugr) {
        let ph = circuit_to_phir(&sample).unwrap();
        assert_eq!(ph.num_ops(), 12);
    }
}