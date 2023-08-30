//! Intermediate structure for converting encoding [`Circuit`]s into [`SerialCircuit`]s.

use std::collections::HashMap;

use downcast_rs::Downcast;
use hugr::extension::prelude::QB_T;
use hugr::ops::OpType;
use hugr::std_extensions::arithmetic::float_types::ConstF64;
use hugr::values::{PrimValue, Value};
use hugr::Wire;
use itertools::Itertools;
use tket_json_rs::circuit_json::{self, Permutation, SerialCircuit};

use crate::circuit::command::{CircuitUnit, Command};
use crate::circuit::Circuit;
use crate::extension::LINEAR_BIT;

use super::op::JsonOp;
use super::{OpConvertError, METADATA_IMPLICIT_PERM, METADATA_PHASE};

/// The state of an in-progress [`SerialCircuit`] being built from a [`Circuit`].
#[derive(Debug, PartialEq)]
pub(super) struct JsonEncoder {
    /// The name of the circuit being encoded.
    name: Option<String>,
    /// Global phase value. Defaults to "0"
    phase: String,
    /// Implicit permutation of output qubits
    implicit_permutation: Vec<Permutation>,
    /// The current commands
    commands: Vec<circuit_json::Command>,
    /// The TKET1 qubit registers associated to each qubit unit of the circuit.
    qubit_regs: HashMap<CircuitUnit, circuit_json::Register>,
    /// The TKET1 bit registers associated to each linear bit unit of the circuit.
    bit_regs: HashMap<CircuitUnit, circuit_json::Register>,
    /// A register of wires with constant values, used to recover TKET1
    /// parameters.
    parameters: HashMap<Wire, String>,
}

impl JsonEncoder {
    /// Create a new [`JsonEncoder`] from a [`Circuit`].
    pub fn new<'circ>(circ: &impl Circuit<'circ>) -> Self {
        let name = circ.name().map(str::to_string);

        // Compute the linear qubit and bit registers. Each one have independent
        // indices starting from zero.
        //
        // TODO Throw an error on non-recognized unit types, or just ignore?
        let mut bit_units = HashMap::new();
        let mut qubit_units = HashMap::new();
        for (unit, ty) in circ.units() {
            if ty == QB_T {
                let index = vec![qubit_units.len() as i64];
                let reg = circuit_json::Register("q".to_string(), index);
                qubit_units.insert(unit, reg);
            } else if ty == *LINEAR_BIT {
                let index = vec![bit_units.len() as i64];
                let reg = circuit_json::Register("c".to_string(), index);
                bit_units.insert(unit, reg);
            }
        }

        let mut encoder = Self {
            name,
            phase: "0".to_string(),
            implicit_permutation: vec![],
            commands: vec![],
            qubit_regs: qubit_units,
            bit_regs: bit_units,
            parameters: HashMap::new(),
        };

        // Encode other parameters stored in the metadata
        if let Some(meta) = circ.get_metadata(circ.root()).as_object() {
            if let Some(phase) = meta.get(METADATA_PHASE) {
                // TODO: Check for invalid encoded metadata
                encoder.phase = phase.as_str().unwrap().to_string();
            }
            if let Some(implicit_perm) = meta.get(METADATA_IMPLICIT_PERM) {
                // TODO: Check for invalid encoded metadata
                encoder.implicit_permutation =
                    serde_json::from_value(implicit_perm.clone()).unwrap();
            }
        }

        encoder
    }

    /// Add a circuit command to the serialization.
    pub fn add_command(&mut self, command: Command, optype: &OpType) -> Result<(), OpConvertError> {
        // Register any output of the command that can be used as a TKET1 parameter.
        self.record_parameters(&command, optype);

        let args = command
            .inputs()
            .iter()
            .filter_map(|&u| self.unit_to_register(u))
            .collect();

        // TODO Restore the opgroup (once the decoding supports it)
        let opgroup = None;

        let op: JsonOp = optype.try_into()?;
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
            phase: self.phase,
            commands: self.commands,
            qubits: self.qubit_regs.into_values().collect_vec(),
            bits: self.bit_regs.into_values().collect_vec(),
            implicit_permutation: self.implicit_permutation,
        }
    }

    /// Record any output of the command that can be used as a TKET1 parameter.
    ///
    /// Associates the output wires with the parameter expression.
    fn record_parameters(&mut self, command: &Command, optype: &OpType) {
        // Only consider commands where all inputs are parameters.
        let inputs = command
            .inputs()
            .iter()
            .filter_map(|unit| match unit {
                CircuitUnit::Wire(wire) => self.parameters.get(wire),
                CircuitUnit::Linear(_) => None,
            })
            .collect_vec();
        if inputs.len() != command.inputs().len() {
            return;
        }

        let param = match optype {
            OpType::Const(const_op) => {
                // New constant, register it if it can be interpreted as a parameter.
                match const_op.value() {
                    Value::Prim(PrimValue::Extension(v)) => {
                        if let Some(f) = v.as_any().downcast_ref::<ConstF64>() {
                            f.to_string()
                        } else {
                            return;
                        }
                    }
                    _ => return,
                }
            }
            OpType::LoadConstant(_op_type) => {
                // Re-use the parameter from the input.
                inputs[0].clone()
            }
            _ => {
                // In the future we may want to support arithmetic operations.
                // (Just concatenating the inputs and the operation symbol, no
                // need for evaluation).
                return;
            }
        };

        for unit in command.outputs() {
            if let CircuitUnit::Wire(wire) = unit {
                self.parameters.insert(*wire, param.clone());
            }
        }
    }

    fn unit_to_register(&self, unit: CircuitUnit) -> Option<circuit_json::Register> {
        self.qubit_regs
            .get(&unit)
            .or_else(|| self.bit_regs.get(&unit))
            .cloned()
    }
}
