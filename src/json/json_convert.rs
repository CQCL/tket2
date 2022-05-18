use std::collections::HashMap;

use crate::circuit::circuit::{Circuit, UnitID};
use crate::circuit::operation::{Op, Param, WireType};
use crate::graph::graph::PortIndex;
use tket_json_rs::circuit_json::{Command, Operation, Permutation, Register, SerialCircuit};
use tket_json_rs::optype::OpType;

fn to_qubit(reg: Register) -> UnitID {
    UnitID::Qubit {
        name: reg.0,
        index: reg.1.into_iter().map(|i| i as u32).collect(),
    }
}

fn to_bit(reg: Register) -> UnitID {
    UnitID::Bit {
        name: reg.0,
        index: reg.1.into_iter().map(|i| i as u32).collect(),
    }
}

impl From<Operation<Param>> for Op {
    fn from(serial_op: Operation<Param>) -> Self {
        let params = serial_op.params;
        if let Some(mut params) = params {
            match serial_op.op_type {
                OpType::Rx => Op::Rx(params.remove(0)),
                OpType::Ry => Op::Ry(params.remove(0)),
                OpType::Rz => Op::Rz(params.remove(0)),
                OpType::ZZPhase => Op::ZZPhase(params.remove(0)),
                OpType::PhasedX => Op::PhasedX(params.remove(0), params.remove(0)),
                OpType::TK1 => Op::TK1(params.remove(0), params.remove(0), params.remove(0)),

                _ => panic!("Parameters for unparametrised op."),
            }
        } else {
            match serial_op.op_type {
                OpType::Input => Op::Input,
                OpType::Output => Op::Output,
                OpType::Create => todo!(),
                OpType::Discard => todo!(),
                OpType::ClInput => todo!(),
                OpType::ClOutput => todo!(),
                OpType::Barrier => todo!(),
                OpType::Label => todo!(),
                OpType::Branch => todo!(),
                OpType::Goto => todo!(),
                OpType::Stop => todo!(),
                OpType::ClassicalTransform => todo!(),
                OpType::SetBits => todo!(),
                OpType::CopyBits => todo!(),
                OpType::RangePredicate => todo!(),
                OpType::ExplicitPredicate => todo!(),
                OpType::ExplicitModifier => todo!(),
                OpType::MultiBit => todo!(),
                OpType::Z => todo!(),
                OpType::X => todo!(),
                OpType::Y => todo!(),
                OpType::S => todo!(),
                OpType::Sdg => todo!(),
                OpType::T => todo!(),
                OpType::Tdg => todo!(),
                OpType::V => todo!(),
                OpType::Vdg => todo!(),
                OpType::SX => todo!(),
                OpType::SXdg => todo!(),
                OpType::H => Op::H,
                OpType::CX => Op::CX,
                OpType::CY => todo!(),
                OpType::CZ => todo!(),
                OpType::CH => todo!(),
                OpType::CV => todo!(),
                OpType::CVdg => todo!(),
                OpType::CSX => todo!(),
                OpType::CSXdg => todo!(),
                OpType::CRz => todo!(),
                OpType::CRx => todo!(),
                OpType::CRy => todo!(),
                OpType::CU1 => todo!(),
                OpType::CU3 => todo!(),
                OpType::PhaseGadget => todo!(),
                OpType::CCX => todo!(),
                OpType::SWAP => todo!(),
                OpType::CSWAP => todo!(),
                OpType::BRIDGE => todo!(),
                OpType::noop => Op::Noop,
                OpType::Measure => Op::Measure,
                OpType::Collapse => todo!(),
                OpType::Reset => todo!(),
                OpType::ECR => todo!(),
                OpType::ISWAP => todo!(),
                OpType::PhasedX => todo!(),
                OpType::NPhasedX => todo!(),
                OpType::ZZMax => Op::ZZMax,
                OpType::XXPhase => todo!(),
                OpType::YYPhase => todo!(),
                OpType::ZZPhase => todo!(),
                OpType::XXPhase3 => todo!(),
                OpType::ESWAP => todo!(),
                OpType::FSim => todo!(),
                OpType::Sycamore => todo!(),
                OpType::ISWAPMax => todo!(),
                OpType::PhasedISWAP => todo!(),
                OpType::CnRy => todo!(),
                OpType::CnX => todo!(),
                OpType::CircBox => todo!(),
                OpType::Unitary1qBox => todo!(),
                OpType::Unitary2qBox => todo!(),
                OpType::Unitary3qBox => todo!(),
                OpType::ExpBox => todo!(),
                OpType::PauliExpBox => todo!(),
                OpType::CliffBox => todo!(),
                OpType::CustomGate => todo!(),
                OpType::PhasePolyBox => todo!(),
                OpType::QControlBox => todo!(),
                OpType::ClassicalExpBox => todo!(),
                OpType::Conditional => todo!(),
                OpType::ProjectorAssertionBox => todo!(),
                OpType::StabiliserAssertionBox => todo!(),
                OpType::UnitaryTableauBox => todo!(),
                _ => panic!("No parameterised for parametrised op."),
            }
        }
    }
}

impl From<Op> for Operation<Param> {
    fn from(op: Op) -> Self {
        let (op_type, params) = match op {
            Op::H => (OpType::H, vec![]),
            Op::CX => (OpType::CX, vec![]),
            Op::ZZMax => (OpType::ZZMax, vec![]),
            Op::Reset => (OpType::Reset, vec![]),
            Op::Input => (OpType::Input, vec![]),
            Op::Output => (OpType::Output, vec![]),
            Op::Rx(p) => (OpType::Rx, vec![p]),
            Op::Ry(p) => (OpType::Ry, vec![p]),
            Op::Rz(p) => (OpType::Rz, vec![p]),
            Op::TK1(a, b, c) => (OpType::TK1, vec![a, b, c]),
            Op::ZZPhase(p) => (OpType::ZZPhase, vec![p]),
            Op::PhasedX(p1, p2) => (OpType::PhasedX, vec![p1, p2]),
            Op::Measure => (OpType::Measure, vec![]),
            Op::Barrier => (OpType::Barrier, vec![]),
            Op::Noop => (OpType::noop, vec![]),
            _ => panic!("Not supported by Serialized TKET-1: {:?}", op),
        };
        // let signature = match self.signature() {
        //     Signature::Linear(sig) => sig.iter().map(|wt| match wt {
        //         WireType::Quantum => todo!(),
        //         WireType::Classical => todo!(),
        //         WireType::Bool => todo!(),
        //     }),
        //     Signature::NonLinear(_, _) => panic!(),
        // }
        let params = (!params.is_empty()).then(|| params);
        Operation {
            op_type,
            // params: params.map(|ps| ps.iter().map(|e| e.as_str().to_string()).collect()),
            params,
            signature: None,
            op_box: None,
            n_qb: None,
            conditional: None,
        }
    }
}

impl From<SerialCircuit<Param>> for Circuit {
    fn from(serialcirc: SerialCircuit<Param>) -> Self {
        let uids: Vec<_> = serialcirc
            .qubits
            .into_iter()
            .map(to_qubit)
            .chain(serialcirc.bits.into_iter().map(to_bit))
            .collect();

        let mut circ = Circuit::with_uids(uids);

        circ.name = serialcirc.name;
        // circ.phase = Param::new(serialcirc.phase);
        circ.phase = serialcirc.phase;

        let frontier: HashMap<UnitID, PortIndex> = circ
            .unitids()
            .enumerate()
            .map(|(i, uid)| (uid.clone(), PortIndex::new(i)))
            .collect();
        for com in serialcirc.commands {
            let op: Op = com.op.into();
            let args = com
                .args
                .into_iter()
                .zip(op.signature().expect("No signature for op").linear)
                .map(|(reg, wiretype)| match wiretype {
                    WireType::Qubit => to_qubit(reg),
                    WireType::LinearBit | WireType::Bool => to_bit(reg),
                    _ => panic!("Unsupported wiretype {:?}", wiretype),
                })
                .map(|uid| frontier[&uid])
                .collect();
            circ.append_op(op, &args).unwrap();
        }
        // TODO implicit perm
        circ
    }
}

impl From<UnitID> for Register {
    fn from(uid: UnitID) -> Self {
        match uid {
            UnitID::Qubit { name, index } | UnitID::Bit { name, index } => {
                Register(name, index.into_iter().map(|i| i as i64).collect())
            }
            _ => panic!("Not supported: {:?}", uid),
        }
    }
}

impl From<Circuit> for SerialCircuit<Param> {
    fn from(circ: Circuit) -> Self {
        let commands = circ
            .to_commands()
            .filter_map(|com| {
                let op: Operation<Param> = com.op.into();
                match op.op_type {
                    OpType::Input | OpType::Output => None,
                    _ => Some(Command {
                        op,
                        args: com.args.into_iter().map(Into::into).collect(),
                        opgroup: None,
                    }),
                }
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
            phase: circ.phase,
            qubits,
            bits,
            implicit_permutation,
        }
    }
}
