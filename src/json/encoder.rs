//! Intermediate structure for converting encoding [`Circuit`]s into [`SerialCircuit`]s.

use std::collections::HashMap;

use hugr::ops::{Const, ConstValue, OpType};
use hugr::Wire;
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
    /// A register of wires with constant values, used to recover TKET1
    /// parameters.
    parameters: HashMap<Wire, String>,
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
            parameters: HashMap::new(),
        }
    }

    /// Add a circuit command to the serialization.
    pub fn add_command(&mut self, command: Command) -> Result<(), OpConvertError> {
        // Register any output of the command that can be used as a TKET1 parameter.
        self.record_parameters(&command);

        let args = vec![]; // TODO command.inputs.iter().map(|unit| unit.index()).collect();

        // TODO Restore the opgroup (once the decoding supports it)
        let opgroup = None;

        let op: JsonOp = command.op.try_into()?;
        let op: circuit_json::Operation = op.into_operation();

        // TODO: Update op.params. Leave untouched the ones that contain free variables.
        // (update decoder to ignore them too, but store them in the wrapped op)

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

    /// Record any output of the command that can be used as a TKET1 parameter.
    ///
    /// Associates the output wires with the parameter expression.
    fn record_parameters(&mut self, command: &Command) {
        // Only consider commands where all inputs are parameters.
        let inputs = command
            .inputs
            .iter()
            .filter_map(|unit| match unit {
                Unit::W(wire) => self.parameters.get(wire),
                Unit::Linear(_) => None,
            })
            .collect_vec();
        if inputs.len() != command.inputs.len() {
            return;
        }

        let param = match command.op {
            OpType::Const(Const(value)) => {
                // New constant, register it if it can be interpreted as a parameter.
                match value {
                    ConstValue::Int { value, .. } => value.to_string(),
                    ConstValue::F64(value) => value.to_string(),
                    _ => return,
                }
            }
            OpType::LoadConstant(_op_type) => {
                // Re-use the parameter from the input.
                inputs[0].clone()
            }
            _ => {
                // In the future we may want to support arithmetic operations.
                // (Just concatenating the inputs, no need for evaluation).
                return;
            }
        };

        for unit in &command.outputs {
            if let Unit::W(wire) = unit {
                self.parameters.insert(*wire, param.clone());
            }
        }
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
