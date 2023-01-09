use crate::circuit::circuit::{Circuit, UnitID};
use crate::circuit::operation::{ConstValue, Op, WireType};
use portgraph::graph::Direction;
use std::collections::HashMap;
use std::str::FromStr;
use tket_json_rs::circuit_json::{Command, Operation, Permutation, Register, SerialCircuit};
use tket_json_rs::optype::OpType;

fn to_qubit(reg: Register) -> UnitID {
    UnitID::Qubit {
        reg_name: reg.0,
        index: reg.1.into_iter().map(|i| i as u32).collect(),
    }
}

fn to_bit(reg: Register) -> UnitID {
    UnitID::Bit {
        name: reg.0,
        index: reg.1.into_iter().map(|i| i as u32).collect(),
    }
}

impl TryFrom<OpType> for Op {
    type Error = OpConvertError;
    fn try_from(serial_op: OpType) -> Result<Self, Self::Error> {
        Ok(match serial_op {
            OpType::Input => Op::Input,
            OpType::Output => Op::Output,
            // OpType::Create => todo!(),
            // OpType::Discard => todo!(),
            // OpType::ClInput => todo!(),
            // OpType::ClOutput => todo!(),
            // OpType::Barrier => todo!(),
            // OpType::Label => todo!(),
            // OpType::Branch => todo!(),
            // OpType::Goto => todo!(),
            // OpType::Stop => todo!(),
            // OpType::ClassicalTransform => todo!(),
            // OpType::SetBits => todo!(),
            // OpType::CopyBits => todo!(),
            // OpType::RangePredicate => todo!(),
            // OpType::ExplicitPredicate => todo!(),
            // OpType::ExplicitModifier => todo!(),
            // OpType::MultiBit => todo!(),
            // OpType::Z => todo!(),
            // OpType::X => todo!(),
            // OpType::Y => todo!(),
            // OpType::S => todo!(),
            // OpType::Sdg => todo!(),
            // OpType::T => todo!(),
            // OpType::Tdg => todo!(),
            // OpType::V => todo!(),
            // OpType::Vdg => todo!(),
            // OpType::SX => todo!(),
            // OpType::SXdg => todo!(),
            OpType::H => Op::H,
            OpType::CX => Op::CX,
            // OpType::CY => todo!(),
            // OpType::CZ => todo!(),
            // OpType::CH => todo!(),
            // OpType::CV => todo!(),
            // OpType::CVdg => todo!(),
            // OpType::CSX => todo!(),
            // OpType::CSXdg => todo!(),
            // OpType::CRz => todo!(),
            // OpType::CRx => todo!(),
            // OpType::CRy => todo!(),
            // OpType::CU1 => todo!(),
            // OpType::CU3 => todo!(),
            // OpType::PhaseGadget => todo!(),
            // OpType::CCX => todo!(),
            // OpType::SWAP => todo!(),
            // OpType::CSWAP => todo!(),
            // OpType::BRIDGE => todo!(),
            OpType::noop => Op::Noop(WireType::Qubit),
            OpType::Measure => Op::Measure,
            // OpType::Collapse => todo!(),
            OpType::Reset => Op::Reset,
            // OpType::ECR => todo!(),
            // OpType::ISWAP => todo!(),
            // OpType::PhasedX => todo!(),
            // OpType::NPhasedX => todo!(),
            OpType::ZZMax => Op::ZZMax,
            // OpType::XXPhase => todo!(),
            // OpType::YYPhase => todo!(),
            // OpType::ZZPhase => todo!(),
            // OpType::XXPhase3 => todo!(),
            // OpType::ESWAP => todo!(),
            // OpType::FSim => todo!(),
            // OpType::Sycamore => todo!(),
            // OpType::ISWAPMax => todo!(),
            // OpType::PhasedISWAP => todo!(),
            // OpType::CnRy => todo!(),R
            // OpType::CnX => todo!(),
            // OpType::CircBox => todo!(),
            // OpType::Unitary1qBox => todo!(),
            // OpType::Unitary2qBox => todo!(),
            // OpType::Unitary3qBox => todo!(),
            // OpType::ExpBox => todo!(),
            // OpType::PauliExpBox => todo!(),
            // OpType::CliffBox => todo!(),
            // OpType::CustomGate => todo!(),
            // OpType::PhasePolyBox => todo!(),
            // OpType::QControlBox => todo!(),
            // OpType::ClassicalExpBox => todo!(),
            // OpType::Conditional => todo!(),
            // OpType::ProjectorAssertionBox => todo!(),
            // OpType::StabiliserAssertionBox => todo!(),
            // OpType::UnitaryTableauBox => todo!(),
            OpType::Rx => Op::RxF64,
            // OpType::Ry => todo!(),
            OpType::Rz => Op::RzF64,
            OpType::TK1 => Op::TK1,
            OpType::AngleAdd => Op::AngleAdd,
            OpType::AngleMul => Op::AngleMul,
            OpType::AngleNeg => Op::AngleNeg,
            OpType::QuatMul => Op::QuatMul,
            OpType::RxF64 => Op::RxF64,
            OpType::RzF64 => Op::RzF64,
            OpType::Rotation => Op::Rotation,
            OpType::ToRotation => Op::ToRotation,
            _ => return Err(OpConvertError),
        })
    }

    // }
}

#[derive(Debug)]
pub struct OpConvertError;

impl TryFrom<&Op> for OpType {
    fn try_from(op: &Op) -> Result<Self, Self::Error> {
        // let (op_type, params) = match op {
        //     Op::H => (OpType::H, vec![]),
        //     Op::CX => (OpType::CX, vec![]),
        //     Op::ZZMax => (OpType::ZZMax, vec![]),
        //     Op::Reset => (OpType::Reset, vec![]),
        //     Op::Input => (OpType::Input, vec![]),
        //     Op::Output => (OpType::Output, vec![]),
        //     Op::Rx(p) => (OpType::Rx, vec![p]),
        //     Op::Ry(p) => (OpType::Ry, vec![p]),
        //     Op::Rz(p) => (OpType::Rz, vec![p]),
        //     Op::TK1(a, b, c) => (OpType::TK1, vec![a, b, c]),
        //     Op::ZZPhase(p) => (OpType::ZZPhase, vec![p]),
        //     Op::PhasedX(p1, p2) => (OpType::PhasedX, vec![p1, p2]),
        //     Op::Measure => (OpType::Measure, vec![]),
        //     Op::Barrier => (OpType::Barrier, vec![]),
        //     Op::Noop(WireType::Qubit) => (OpType::noop, vec![]),
        //     _ => panic!("Not supported by Serialized TKET-1: {:?}", op),
        // };

        // let signature = match self.signature() {
        //     Signature::Linear(sig) => sig.iter().map(|wt| match wt {
        //         WireType::Quantum => todo!(),
        //         WireType::Classical => todo!(),
        //         WireType::Bool => todo!(),
        //     }),
        //     Signature::NonLinear(_, _) => panic!(),
        // }

        // let params = (!params.is_empty())
        //     .then(|| params.into_iter().map(|p| p.to_string().into()).collect());
        // Operation {
        //     op_type,
        //     // params: params.map(|ps| ps.iter().map(|e| e.as_str().to_string()).collect()),
        //     params,
        //     signature: None,
        //     op_box: None,
        //     n_qb: None,
        //     conditional: None,
        // }

        Ok(match op {
            Op::H => OpType::H,
            Op::CX => OpType::CX,
            Op::ZZMax => OpType::ZZMax,
            Op::Reset => OpType::Reset,
            Op::Input => OpType::Input,
            Op::Output => OpType::Output,
            Op::Measure => OpType::Measure,
            Op::Barrier => OpType::Barrier,
            Op::RxF64 => OpType::Rx,
            Op::RzF64 => OpType::Rz,
            Op::TK1 => OpType::TK1,
            Op::Noop(WireType::Qubit) => OpType::noop,
            Op::AngleAdd => OpType::AngleAdd,
            Op::AngleMul => OpType::AngleMul,
            Op::AngleNeg => OpType::AngleNeg,
            Op::QuatMul => OpType::QuatMul,
            Op::Rotation => OpType::Rotation,
            Op::ToRotation => OpType::ToRotation,
            Op::Copy { .. } => OpType::Copy,
            Op::Const(_) => OpType::Const,
            Op::Custom(cbox) => {
                if let Some(tk1op) = cbox.downcast_ref::<Operation>() {
                    tk1op.op_type.clone()
                } else {
                    return Err(OpConvertError);
                }
            }
            _ => return Err(OpConvertError),
        })
    }

    type Error = OpConvertError;
}

impl<P: Into<String> + Clone + std::fmt::Display> From<SerialCircuit<P>> for Circuit {
    fn from(serialcirc: SerialCircuit<P>) -> Self {
        let uids: Vec<_> = serialcirc
            .qubits
            .into_iter()
            .map(to_qubit)
            .chain(serialcirc.bits.into_iter().map(to_bit))
            .collect();

        let mut circ = Circuit::with_uids(uids);

        circ.name = serialcirc.name;
        // circ.phase = Param::new(serialcirc.phase);
        circ.phase = f64::from_str(&serialcirc.phase.clone().into()[..]).unwrap();

        let frontier: HashMap<UnitID, usize> = circ
            .unitids()
            .enumerate()
            .map(|(i, uid)| (uid.clone(), i))
            .collect();
        for com in serialcirc.commands {
            let ps = com.op.params.clone();
            let op: Op = com
                .op
                .op_type
                .clone()
                .try_into()
                .unwrap_or(Op::Custom(Box::new(map_op(com.op))));
            let args: Vec<_> = com
                .args
                .into_iter()
                .map(|reg| {
                    // relies on TKET1 constraint that all registers have
                    // unique names
                    *frontier
                        .get(&to_qubit(reg.clone()))
                        .or_else(|| frontier.get(&to_bit(reg)))
                        .expect("Reg not found.")
                })
                .collect();

            // assumes the linear wires are always the first ones
            let v = circ.append_op(op, &args[..]).unwrap();
            if let Some(params) = ps {
                let mut prev = circ.dag.node_edges(v, Direction::Incoming).last();
                for p in params.into_iter() {
                    let p_str = p.to_string();
                    let param_source = if let Ok(f) = f64::from_str(&p_str[..]) {
                        let e = circ.add_edge(WireType::Angle);
                        circ.add_vertex_with_edges(
                            Op::Const(ConstValue::f64_angle(f)),
                            vec![],
                            vec![e],
                        );
                        e
                    } else {
                        circ.add_unitid(UnitID::Angle(p_str))
                    };
                    circ.dag
                        .connect(v, param_source, Direction::Incoming, prev)
                        .unwrap();
                    prev = Some(param_source);
                    // circ.tup_add_edge(
                    //     param_source,
                    //     (v, (args.len() + i) as u8).into(),
                    //     WireType::Angle,
                    // );
                }
            };
        }
        // TODO implicit perm

        circ
    }
}

impl From<UnitID> for Register {
    fn from(uid: UnitID) -> Self {
        match uid {
            UnitID::Qubit {
                reg_name: name,
                index,
            }
            | UnitID::Bit { name, index } => {
                Register(name, index.into_iter().map(i64::from).collect())
            }
            _ => panic!("Not supported: {:?}", uid),
        }
    }
}

impl<P: From<String> + std::fmt::Debug + Clone + Send + Sync> From<Circuit> for SerialCircuit<P> {
    fn from(circ: Circuit) -> Self {
        let commands = circ
            .to_commands()
            .filter_map(|com| {
                let params = match com.op {
                    Op::Input | Op::Output | Op::Const(_) => return None,
                    Op::RzF64 | Op::RxF64 => param_strings(&circ, &com, 1),
                    Op::TK1 => param_strings(&circ, &com, 3),
                    _ => None,
                };
                let tk1op = if let Op::Custom(cbox) = com.op {
                    cbox.downcast_ref::<Operation>().cloned()
                } else {
                    None
                };

                Some(Command {
                    op: map_op(tk1op.unwrap_or_else(|| Operation {
                        op_type: com.op.try_into().unwrap(),
                        n_qb: None,
                        params,
                        op_box: None,
                        signature: None,
                        conditional: None,
                    })),
                    args: com.args.into_iter().map(Into::into).collect(),
                    opgroup: None,
                })
            })
            .collect();

        let qubits: Vec<Register> = circ.qubits().map(Into::into).collect();
        let implicit_permutation = qubits
            .iter()
            .map(|q| Permutation(q.clone(), q.clone()))
            .collect();
        let bits = circ.bits().map(Into::into).collect();

        SerialCircuit {
            commands,
            name: circ.name,
            phase: circ.phase.to_string().into(),
            qubits,
            bits,
            implicit_permutation,
        }
    }
}

fn map_op<P, T: Into<P>>(op: Operation<T>) -> Operation<P> {
    Operation {
        op_type: op.op_type,
        n_qb: op.n_qb,
        params: op
            .params
            .map(|params| params.into_iter().map(|p| p.into()).collect()),
        op_box: op.op_box,
        signature: op.signature,
        conditional: op.conditional,
    }
}

fn param_strings(
    circ: &Circuit,
    com: &crate::circuit::circuit::Command,
    num: u32,
) -> Option<Vec<String>> {
    Some(
        (0..num)
            .map(|i| {
                let angle_edge = circ
                    .edge_at_port(com.vertex, 1 + i as usize, Direction::Incoming)
                    .expect("Expected an angle wire.");
                let pred_n = circ
                    .dag
                    .edge_endpoint(angle_edge, Direction::Outgoing)
                    .unwrap();
                let pred = &circ
                    .dag
                    .node_weight(pred_n)
                    .expect("Expected predecessor node.")
                    .op;
                match pred {
                    Op::Const(ConstValue::Angle(p)) => p.to_f64().to_string(),
                    Op::Input => match &circ.uids[circ
                        .port_of_edge(pred_n, angle_edge, Direction::Outgoing)
                        .unwrap()]
                    {
                        UnitID::Angle(s) => s.clone(),
                        _ => panic!("Must be an Angle input"),
                    },
                    _ => panic!("Only constant or simple string param inputs supported."),
                }
            })
            .collect(),
    )
}
