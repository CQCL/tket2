use crate::circuit::circuit::{Circuit, UnitID};
use crate::circuit::operation::{Op, Signature, WireType};
use crate::circuit_json::{Operation, Permutation, Register, SerialCircuit};
use crate::optype::OpType;

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

impl From<Operation> for Op {
    fn from(serial_op: Operation) -> Self {
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
            OpType::Rx => todo!(),
            OpType::Ry => todo!(),
            OpType::Rz => todo!(),
            OpType::U3 => todo!(),
            OpType::U2 => todo!(),
            OpType::U1 => todo!(),
            OpType::TK1 => todo!(),
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
            OpType::noop => todo!(),
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
        }
    }
}

impl From<SerialCircuit> for Circuit {
    fn from(serialcirc: SerialCircuit) -> Self {
        let mut circ = Self::new();

        circ.name = serialcirc.name;
        circ.phase = serialcirc.phase;

        for qb in serialcirc.qubits {
            circ.add_unitid(to_qubit(qb));
        }

        for bit in serialcirc.bits {
            circ.add_unitid(to_bit(bit));
        }

        for com in serialcirc.commands {
            let op: Op = com.op.into();
            let args = com
                .args
                .into_iter()
                .zip(if let Signature::Linear(sig) = op.signature() {
                    sig
                } else {
                    panic!()
                })
                .map(|(reg, wiretype)| match wiretype {
                    WireType::Quantum => to_qubit(reg),
                    WireType::Classical | WireType::Bool => to_bit(reg),
                })
                .collect();
            circ.add_op(op, &args, com.opgroup).unwrap();
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
        }
    }
}

impl From<Circuit> for SerialCircuit {
    fn from(circ: Circuit) -> Self {
        let commands = circ
            .to_commands()
            .filter_map(|com| {
                let op = com.op.to_serialized();
                match op.op_type {
                    OpType::Input | OpType::Output => None,
                    _ => Some(crate::circuit_json::Command {
                        op,
                        args: com.args.into_iter().map(Into::into).collect(),
                        opgroup: com.opgroup,
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
