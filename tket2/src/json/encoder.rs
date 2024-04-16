//! Intermediate structure for converting encoding [`Circuit`]s into [`SerialCircuit`]s.

use core::panic;
use std::collections::HashMap;

use hugr::extension::prelude::QB_T;
use hugr::ops::{OpName, OpType};
use hugr::std_extensions::arithmetic::float_types::ConstF64;
use hugr::Wire;
use itertools::{Either, Itertools};
use tket_json_rs::circuit_json::{self, Permutation, Register, SerialCircuit};

use crate::circuit::command::{CircuitUnit, Command};
use crate::circuit::Circuit;
use crate::extension::LINEAR_BIT;
use crate::ops::{match_symb_const_op, op_matches};
use crate::Tk2Op;

use super::op::JsonOp;
use super::{
    OpConvertError, METADATA_B_REGISTERS, METADATA_IMPLICIT_PERM, METADATA_PHASE,
    METADATA_Q_REGISTERS,
};

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
    qubit_to_reg: HashMap<CircuitUnit, Register>,
    /// The TKET1 bit registers associated to each linear bit unit of the circuit.
    bit_to_reg: HashMap<CircuitUnit, Register>,
    /// The ordered TKET1 names for the input qubit registers.
    qubit_registers: Vec<Register>,
    /// The ordered TKET1 names for the input bit registers.
    bit_registers: Vec<Register>,
    /// A register of wires with constant values, used to recover TKET1
    /// parameters.
    parameters: HashMap<Wire, String>,
}

impl JsonEncoder {
    /// Create a new [`JsonEncoder`] from a [`Circuit`].
    pub fn new(circ: &impl Circuit) -> Self {
        let name = circ.name().map(str::to_string);

        let mut qubit_registers = vec![];
        let mut bit_registers = vec![];
        let mut phase = "0".to_string();
        let mut implicit_permutation = vec![];

        // Recover other parameters stored in the metadata
        // TODO: Check for invalid encoded metadata
        let root = circ.root();
        if let Some(p) = circ.get_metadata(root, METADATA_PHASE) {
            phase = p.as_str().unwrap().to_string();
        }
        if let Some(perm) = circ.get_metadata(root, METADATA_IMPLICIT_PERM) {
            implicit_permutation = serde_json::from_value(perm.clone()).unwrap();
        }
        if let Some(q_regs) = circ.get_metadata(root, METADATA_Q_REGISTERS) {
            qubit_registers = serde_json::from_value(q_regs.clone()).unwrap();
        }
        if let Some(b_regs) = circ.get_metadata(root, METADATA_B_REGISTERS) {
            bit_registers = serde_json::from_value(b_regs.clone()).unwrap();
        }

        // Map the Hugr units to tket1 register names.
        // Uses the names from the metadata if available, or initializes new sequentially-numbered registers.
        let mut bit_to_reg = HashMap::new();
        let mut qubit_to_reg = HashMap::new();
        let get_register = |registers: &mut Vec<Register>, prefix: &str, index| {
            registers.get(index).cloned().unwrap_or_else(|| {
                let r = Register(prefix.to_string(), vec![index as i64]);
                registers.push(r.clone());
                r
            })
        };
        for (unit, _, ty) in circ.units() {
            if ty == QB_T {
                let index = qubit_to_reg.len();
                let reg = get_register(&mut qubit_registers, "q", index);
                qubit_to_reg.insert(unit, reg);
            } else if ty == *LINEAR_BIT {
                let index = bit_to_reg.len();
                let reg = get_register(&mut bit_registers, "b", index);
                bit_to_reg.insert(unit, reg.clone());
            }
        }

        Self {
            name,
            phase,
            implicit_permutation,
            commands: vec![],
            qubit_to_reg,
            bit_to_reg,
            qubit_registers,
            bit_registers,
            parameters: HashMap::new(),
        }
    }

    /// Add a circuit command to the serialization.
    pub fn add_command<C: Circuit>(
        &mut self,
        command: Command<'_, C>,
        optype: &OpType,
    ) -> Result<(), OpConvertError> {
        // Register any output of the command that can be used as a TKET1 parameter.
        if self.record_parameters(&command, optype) {
            // for now all ops that record parameters should be ignored (are
            // just constants)
            return Ok(());
        }

        let (args, params): (Vec<Register>, Vec<Wire>) =
            command
                .inputs()
                .partition_map(|(u, _, _)| match self.unit_to_register(u) {
                    Some(r) => Either::Left(r),
                    None => match u {
                        CircuitUnit::Wire(w) => Either::Right(w),
                        CircuitUnit::Linear(_) => {
                            panic!("No register found for the linear input {u:?}.")
                        }
                    },
                });

        // TODO Restore the opgroup (once the decoding supports it)
        let opgroup = None;
        let op: JsonOp = optype.try_into()?;
        let mut op: circuit_json::Operation = op.into_operation();
        if !params.is_empty() {
            op.params = Some(
                params
                    .into_iter()
                    .filter_map(|w| self.parameters.get(&w))
                    .cloned()
                    .collect(),
            )
        }
        // TODO: ops that contain free variables.
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
            qubits: self.qubit_registers,
            bits: self.bit_registers,
            implicit_permutation: self.implicit_permutation,
        }
    }

    /// Record any output of the command that can be used as a TKET1 parameter.
    /// Returns whether parameters were recorded.
    /// Associates the output wires with the parameter expression.
    fn record_parameters<C: Circuit>(&mut self, command: &Command<'_, C>, optype: &OpType) -> bool {
        // Only consider commands where all inputs are parameters.
        let inputs = command
            .inputs()
            .filter_map(|(unit, _, _)| match unit {
                CircuitUnit::Wire(wire) => self.parameters.get(&wire),
                CircuitUnit::Linear(_) => None,
            })
            .collect_vec();
        if inputs.len() != command.input_count() {
            debug_assert!(
                !matches!(optype, OpType::Const(_) | OpType::LoadConstant(_)),
                "Found a {} with {} inputs, of which {} are non-linear. In node {:?}",
                optype.name(),
                command.input_count(),
                inputs.len(),
                command.node()
            );
            return false;
        }

        let param = match optype {
            OpType::Const(const_op) => {
                // New constant, register it if it can be interpreted as a parameter.
                let Some(const_float) = const_op.get_custom_value::<ConstF64>() else {
                    return false;
                };
                const_float.to_string()
            }
            OpType::LoadConstant(_op_type) => {
                // Re-use the parameter from the input.
                inputs[0].clone()
            }
            op if op_matches(op, Tk2Op::AngleAdd) => {
                format!("{} + {}", inputs[0], inputs[1])
            }
            _ => {
                let Some(s) = match_symb_const_op(optype) else {
                    return false;
                };
                s.to_string()
            }
        };

        for (unit, _, _) in command.outputs() {
            match unit {
                CircuitUnit::Wire(wire) => self.add_parameter(wire, param.clone()),
                CircuitUnit::Linear(_) => panic!(
                    "Found a non-wire output {unit:?} for a {} command.",
                    optype.name()
                ),
            }
        }
        true
    }

    /// Translate a linear [`CircuitUnit`] into a [`Register`], if possible.
    fn unit_to_register(&self, unit: CircuitUnit) -> Option<Register> {
        self.qubit_to_reg
            .get(&unit)
            .or_else(|| self.bit_to_reg.get(&unit))
            .cloned()
    }

    pub(super) fn add_parameter(&mut self, wire: Wire, param: String) {
        self.parameters.insert(wire, param);
    }
}
