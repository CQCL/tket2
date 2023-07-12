//! Intermediate structure for converting encoding [`Circuit`]s into [`SerialCircuit`]s.

use hugr::ops::{OpTag, OpTrait, OpType};
use hugr::Node;
use itertools::{Either, Itertools};
use tket_json_rs::circuit_json::{self, SerialCircuit};

use crate::circuit::command::{Command, Unit};
use crate::circuit::Circuit;
use crate::utils::{BIT, QB};

use super::op::JsonOp;
use super::OpConvertError;

/// The state of an in-progress [`SerialCircuit`] being built from a [`Circuit`].
#[derive(Debug, PartialEq)]
pub(super) struct JsonEncoder {
    /// The name of the circuit being encoded.
    name: Option<String>,
    /// The current commands
    commands: Vec<circuit_json::Command>,
    /// The qubit registers
    qubits: Vec<circuit_json::Register>,
    /// The bit registers
    bits: Vec<circuit_json::Register>,
}

impl JsonEncoder {
    /// Create a new [`JsonEncoder`] from a [`Circuit`].
    pub fn new<'circ>(circ: &impl Circuit<'circ>) -> Self {
        let name = circ.name().map(str::to_string);

        // Compute the linear qubit and bit registers
        // TODO We are checking for Hugr's bit. We should change this when the
        // decoding starts using linear bits instead.
        // TODO Through an error on non-recognized unit types, or just ignore?
        let (qubits, bits) = circ
            .units()
            .iter()
            .filter(|(_, ty)| ty == &QB || ty == &BIT)
            .partition_map(|(unit, ty)| {
                if ty == &QB {
                    Either::Left(unit_to_qubit_register(*unit).unwrap())
                } else {
                    Either::Right(unit_to_bit_register(*unit).unwrap())
                }
            });

        Self {
            name,
            commands: vec![],
            qubits,
            bits,
        }
    }

    /// Add a circuit command to the serialization.
    pub fn add_command(&mut self, command: Command) -> Result<(), OpConvertError> {
        // If the command is a constant, store it so we can recover it as
        // an argument later.
        if OpTag::Const.is_superset(command.op.tag()) {
            self.register_const(command.node, command.op);
        }

        let args = vec![]; // TODO command.inputs.iter().map(|unit| unit.index()).collect();

        // TODO Restore the opgroup (once the decoding supports it)
        let opgroup = None;

        let op: JsonOp = command.op.try_into()?;
        let op: circuit_json::Operation = op.into_operation();

        let command = circuit_json::Command { op, args, opgroup };
        self.commands.push(command);
        Ok(())
    }

    pub fn finish(self) -> SerialCircuit {
        SerialCircuit {
            name: self.name,
            phase: "".to_string(),
            commands: self.commands,
            qubits: self.qubits,
            bits: self.bits,
            implicit_permutation: vec![],
        }
    }

    fn register_const(&mut self, _node: Node, _op_type: &OpType) {
        todo!()
    }
}

/// Cast a linear [`Unit`] to a qubit [`circuit_json::Register`].
fn unit_to_qubit_register(unit: Unit) -> Option<circuit_json::Register> {
    let Unit::Linear(index) = unit else { return None; };
    Some(circuit_json::Register("q".to_string(), vec![index as i64]))
}

/// Cast a linear [`Unit`] to a bit [`circuit_json::Register`].
fn unit_to_bit_register(unit: Unit) -> Option<circuit_json::Register> {
    let Unit::Linear(index) = unit else { return None; };
    Some(circuit_json::Register("b".to_string(), vec![index as i64]))
}
