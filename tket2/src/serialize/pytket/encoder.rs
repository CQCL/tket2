//! Intermediate structure for converting encoding [`Circuit`]s into [`SerialCircuit`]s.

use core::panic;
use std::collections::HashMap;

use hugr::extension::prelude::QB_T;
use hugr::ops::{NamedOp, OpType};
use hugr::std_extensions::arithmetic::float_types::FLOAT64_TYPE;
use hugr::{HugrView, Wire};
use itertools::{Either, Itertools};
use tket_json_rs::circuit_json::{self, Permutation, Register, SerialCircuit};

use crate::circuit::command::{CircuitUnit, Command};
use crate::circuit::Circuit;
use crate::extension::LINEAR_BIT;
use crate::ops::{match_symb_const_op, op_matches};
use crate::Tk2Op;

use super::op::Tk1Op;
use super::{
    try_constant_to_param, OpConvertError, TK1ConvertError, METADATA_B_REGISTERS,
    METADATA_IMPLICIT_PERM, METADATA_PHASE, METADATA_Q_REGISTERS,
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
    ///
    /// Nb: Although `tket-json-rs` calls these "registers", they're actually
    /// identifiers for single qubits in the `Register::0` register.
    qubit_registers: Vec<Register>,
    /// The ordered TKET1 names for the input bit registers.
    ///
    /// Nb: Although `tket-json-rs` calls these "registers", they're actually
    /// identifiers for single bits in the `Register::0` register.
    bit_registers: Vec<Register>,
    /// A register of wires with constant values, used to recover TKET1
    /// parameters.
    parameters: HashMap<Wire, String>,
}

impl JsonEncoder {
    /// Create a new [`JsonEncoder`] from a [`Circuit`].
    pub fn new(circ: &Circuit<impl HugrView>) -> Result<Self, TK1ConvertError> {
        let name = circ.name().map(str::to_string);
        let hugr = circ.hugr();

        let mut qubit_registers = vec![];
        let mut bit_registers = vec![];
        let mut phase = "0".to_string();
        let mut implicit_permutation = vec![];

        // Recover other parameters stored in the metadata
        // TODO: Check for invalid encoded metadata
        let root = circ.parent();
        if let Some(p) = hugr.get_metadata(root, METADATA_PHASE) {
            phase = p.as_str().unwrap().to_string();
        }
        if let Some(perm) = hugr.get_metadata(root, METADATA_IMPLICIT_PERM) {
            implicit_permutation = serde_json::from_value(perm.clone()).unwrap();
        }
        if let Some(q_regs) = hugr.get_metadata(root, METADATA_Q_REGISTERS) {
            qubit_registers = serde_json::from_value(q_regs.clone()).unwrap();
        }
        if let Some(b_regs) = hugr.get_metadata(root, METADATA_B_REGISTERS) {
            bit_registers = serde_json::from_value(b_regs.clone()).unwrap();
        }

        // Map the Hugr units to tket1 register names.
        // Uses the names from the metadata if available, or initializes new sequentially-numbered registers.
        let mut qubit_to_reg = HashMap::new();
        let mut bit_to_reg = HashMap::new();
        let get_register = |registers: &mut Vec<Register>, name: &str, index| {
            registers.get(index).cloned().unwrap_or_else(|| {
                let r = Register(name.to_string(), vec![index as i64]);
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

        let mut encoder = Self {
            name,
            phase,
            implicit_permutation,
            commands: vec![],
            qubit_to_reg,
            bit_to_reg,
            qubit_registers,
            bit_registers,
            parameters: HashMap::new(),
        };

        encoder.add_input_parameters(circ)?;

        Ok(encoder)
    }

    /// Add a circuit command to the serialization.
    pub fn add_command<T: HugrView>(
        &mut self,
        command: Command<'_, T>,
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

        // Convert the command's operator to a pytket serialized one. This will
        // return an error for operations that should have been caught by the
        // `record_parameters` branch above (in addition to other unsupported
        // ops).
        let op: Tk1Op = Tk1Op::try_from_optype(optype.clone())?;
        let mut op: circuit_json::Operation = op
            .serialised_op()
            .ok_or_else(|| OpConvertError::UnsupportedOpSerialization(optype.clone()))?;

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

    /// Finish building and return the final [`SerialCircuit`].
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
    fn record_parameters<T: HugrView>(
        &mut self,
        command: &Command<'_, T>,
        optype: &OpType,
    ) -> bool {
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
                match try_constant_to_param(const_op.value()) {
                    Some(param) => param,
                    None => return false,
                }
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

    /// Associate a parameter expression with a wire.
    fn add_parameter(&mut self, wire: Wire, param: String) {
        self.parameters.insert(wire, param);
    }

    /// Adds a parameter for each floating-point input to the circuit.
    fn add_input_parameters(
        &mut self,
        circ: &Circuit<impl HugrView>,
    ) -> Result<(), TK1ConvertError> {
        let mut num_f64_inputs = 0;
        for (wire, _, typ) in circ.units() {
            match wire {
                CircuitUnit::Linear(_) => {}
                CircuitUnit::Wire(wire) if typ == FLOAT64_TYPE => {
                    let param = format!("f{num_f64_inputs}");
                    num_f64_inputs += 1;
                    self.add_parameter(wire, param);
                }
                CircuitUnit::Wire(_) => return Err(TK1ConvertError::NonSerializableInputs { typ }),
            }
        }
        Ok(())
    }
}
