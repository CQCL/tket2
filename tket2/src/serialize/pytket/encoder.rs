//! Intermediate structure for converting encoding [`Circuit`]s into [`SerialCircuit`]s.

use core::panic;
use std::collections::{HashMap, HashSet, VecDeque};

use hugr::extension::prelude::{BOOL_T, QB_T};
use hugr::ops::{NamedOp, OpType};
use hugr::std_extensions::arithmetic::float_types::FLOAT64_TYPE;
use hugr::{HugrView, Wire};
use itertools::Itertools;
use tket_json_rs::circuit_json::Register as RegisterUnit;
use tket_json_rs::circuit_json::{self, Permutation, SerialCircuit};

use crate::circuit::command::{CircuitUnit, Command};
use crate::circuit::Circuit;
use crate::ops::{match_symb_const_op, op_matches};
use crate::Tk2Op;

use super::op::Tk1Op;
use super::{
    try_constant_to_param, OpConvertError, TK1ConvertError, METADATA_B_REGISTERS,
    METADATA_IMPLICIT_PERM, METADATA_OPGROUP, METADATA_PHASE, METADATA_Q_REGISTERS,
};

/// The state of an in-progress [`SerialCircuit`] being built from a [`Circuit`].
#[derive(Debug, Clone)]
pub(super) struct Tk1Encoder {
    /// The name of the circuit being encoded.
    name: Option<String>,
    /// Global phase value. Defaults to "0"
    phase: String,
    /// Implicit permutation of output qubits
    implicit_permutation: Vec<Permutation>,
    /// The current serialised commands
    commands: Vec<circuit_json::Command>,
    /// A tracker for the qubits used in the circuit.
    qubits: QubitTracker,
    /// A tracker for the bits used in the circuit.
    bits: BitTracker,
    /// A tracker for the operation parameters used in the circuit.
    parameters: ParameterTracker,
}

impl Tk1Encoder {
    /// Create a new [`JsonEncoder`] from a [`Circuit`].
    pub fn new(circ: &Circuit<impl HugrView>) -> Result<Self, TK1ConvertError> {
        let name = circ.name().map(str::to_string);
        let hugr = circ.hugr();

        // Check for unsupported input types.
        for (_, _, typ) in circ.units() {
            if ![FLOAT64_TYPE, QB_T, BOOL_T].contains(&typ) {
                return Err(TK1ConvertError::NonSerializableInputs { typ });
            }
        }

        // Recover other parameters stored in the metadata
        // TODO: Check for invalid encoded metadata
        let phase = match hugr.get_metadata(circ.parent(), METADATA_PHASE) {
            Some(p) => p.as_str().unwrap().to_string(),
            None => "0".to_string(),
        };
        let implicit_permutation = match hugr.get_metadata(circ.parent(), METADATA_IMPLICIT_PERM) {
            Some(perm) => serde_json::from_value(perm.clone()).unwrap(),
            None => vec![],
        };
        let qubit_tracker = QubitTracker::new(circ);
        let bit_tracker = BitTracker::new(circ);
        let parameter_tracker = ParameterTracker::new(circ);

        Ok(Self {
            name,
            phase,
            implicit_permutation,
            commands: vec![],
            qubits: qubit_tracker,
            bits: bit_tracker,
            parameters: parameter_tracker,
        })
    }

    /// Add a circuit command to the serialization.
    pub fn add_command<T: HugrView>(
        &mut self,
        command: Command<'_, T>,
        optype: &OpType,
    ) -> Result<(), OpConvertError> {
        // Register any output of the command that can be used as a TKET1 parameter.
        if self.parameters.record_parameters(&command, optype) {
            // for now all ops that record parameters should be ignored (are
            // just constants)
            return Ok(());
        }

        let tk1op: Tk1Op = Tk1Op::try_from_optype(optype.clone())?;

        // Get the registers and wires associated with the operation's inputs.
        let mut qubit_args = Vec::with_capacity(tk1op.qubit_inputs());
        let mut bit_args = Vec::with_capacity(tk1op.bit_inputs());
        let mut params = Vec::with_capacity(tk1op.num_params());
        for (unit, _, ty) in command.inputs() {
            if ty == QB_T {
                let reg = self.unit_to_register(unit).unwrap_or_else(|| {
                    panic!(
                        "No register found for qubit input {unit} in node {}.",
                        command.node(),
                    )
                });
                qubit_args.push(reg);
            } else if ty == BOOL_T {
                let reg = self.unit_to_register(unit).unwrap_or_else(|| {
                    panic!(
                        "No register found for bit input {unit} in node {}.",
                        command.node(),
                    )
                });
                bit_args.push(reg);
            } else if ty == FLOAT64_TYPE {
                let CircuitUnit::Wire(param_wire) = unit else {
                    unreachable!("Float types are not linear.")
                };
                params.push(param_wire);
            } else {
                return Err(OpConvertError::UnsupportedInputType {
                    typ: ty.clone(),
                    optype: optype.clone(),
                    node: command.node(),
                });
            }
        }

        let mut bit_output_count = 0;
        let mut qubit_output_count = 0;
        for (unit, _, ty) in command.outputs() {
            if ty == QB_T {
                // Currently we do not support encoding operations that allocate
                // new qubits.
                qubit_output_count += 1;
                if qubit_args.len() < qubit_output_count {
                    return Err(OpConvertError::TooManyOutputQubits {
                        typ: ty.clone(),
                        optype: optype.clone(),
                        node: command.node(),
                    });
                }
            } else if ty == BOOL_T {
                // If the operation has any bit outputs, associate their wires with either an
                // input bit register, or create a new one.
                let CircuitUnit::Wire(wire) = unit else {
                    panic!("Bool types are not linear.")
                };
                if let Some(reg) = bit_args.get(bit_output_count) {
                    self.bits.set_bit_register(wire, reg.clone());
                } else {
                    let reg = self.bits.add_bit_register(wire);
                    bit_args.push(reg.clone());
                }
                bit_output_count += 1;
            } else {
                return Err(OpConvertError::UnsupportedOutputType {
                    typ: ty.clone(),
                    optype: optype.clone(),
                    node: command.node(),
                });
            }
        }

        let opgroup: Option<String> = command
            .metadata(METADATA_OPGROUP)
            .and_then(serde_json::Value::as_str)
            .map(ToString::to_string);

        // Convert the command's operator to a pytket serialized one. This will
        // return an error for operations that should have been caught by the
        // `record_parameters` branch above (in addition to other unsupported
        // ops).
        let mut serial_op: circuit_json::Operation = tk1op
            .serialised_op()
            .ok_or_else(|| OpConvertError::UnsupportedOpSerialization(optype.clone()))?;

        if !params.is_empty() {
            serial_op.params = Some(
                params
                    .into_iter()
                    .filter_map(|w| self.parameters.get(&w))
                    .cloned()
                    .collect(),
            )
        }
        // TODO: ops that contain free variables.
        // (update decoder to ignore them too, but store them in the wrapped op)

        let mut args = qubit_args;
        args.append(&mut bit_args);
        let command = circuit_json::Command {
            op: serial_op,
            args,
            opgroup,
        };
        self.commands.push(command);

        Ok(())
    }

    /// Finish building and return the final [`SerialCircuit`].
    pub fn finish(self) -> SerialCircuit {
        SerialCircuit {
            name: self.name,
            phase: self.phase,
            commands: self.commands,
            qubits: self.qubits.finish(),
            bits: self.bits.finish(),
            implicit_permutation: self.implicit_permutation,
        }
    }

    /// Translate a linear [`CircuitUnit`] into a [`RegisterUnit`], if possible.
    fn unit_to_register(&self, unit: CircuitUnit) -> Option<RegisterUnit> {
        match unit {
            CircuitUnit::Linear(i) => self.qubits.get(i).cloned(),
            CircuitUnit::Wire(wire) => self.bits.get(&wire).cloned(),
        }
    }
}

/// A structure for tracking qubits used in the circuit being encoded.
///
/// Nb: Although `tket-json-rs` has a "Register" struct, it's actually
/// an identifier for single qubits in the `Register::0` register.
/// We rename it to `RegisterUnit` here to avoid confusion.
#[derive(Debug, Clone, Default)]
struct QubitTracker {
    /// The ordered TKET1 names for the input qubit registers.
    inputs: Vec<RegisterUnit>,
    /// The TKET1 qubit registers associated to each qubit unit of the circuit.
    qubit_to_reg: HashMap<usize, RegisterUnit>,
}

impl QubitTracker {
    /// Create a new [`QubitTracker`] from the bit inputs of a [`Circuit`].
    /// Reads the [`METADATA_Q_REGISTERS`] metadata entry with preset pytket bit register names.
    ///
    /// If the circuit contains more bit inputs than the provided list,
    /// new registers are created for the remaining bits.
    pub fn new(circ: &Circuit<impl HugrView>) -> Self {
        let mut tracker = QubitTracker::default();

        if let Some(input_regs) = circ
            .hugr()
            .get_metadata(circ.parent(), METADATA_Q_REGISTERS)
        {
            tracker.inputs = serde_json::from_value(input_regs.clone()).unwrap();
        }

        let qubit_count = circ.units().filter(|(_, _, ty)| ty == &QB_T).count();

        for i in 0..qubit_count {
            // Use the given input register names if available, or create new ones.
            if let Some(reg) = tracker.inputs.get(i) {
                tracker.qubit_to_reg.insert(i, reg.clone());
            } else {
                let reg = tracker.add_qubit_register(i).clone();
                tracker.inputs.push(reg);
            }
        }

        tracker
    }

    /// Add a new register unit for a bit wire.
    pub fn add_qubit_register(&mut self, unit_id: usize) -> &RegisterUnit {
        let reg = RegisterUnit("q".to_string(), vec![self.qubit_to_reg.len() as i64]);
        self.qubit_to_reg.insert(unit_id, reg);
        self.qubit_to_reg.get(&unit_id).unwrap()
    }

    /// Returns the register unit for a bit wire, if it exists.
    pub fn get(&self, unit_id: usize) -> Option<&RegisterUnit> {
        self.qubit_to_reg.get(&unit_id)
    }

    /// Consumes the tracker and returns the final list of bit registers.
    pub fn finish(mut self) -> Vec<RegisterUnit> {
        let mut missing_regs: HashSet<_> = self.qubit_to_reg.into_values().collect();
        for reg in &self.inputs {
            missing_regs.remove(reg);
        }
        self.inputs.extend(missing_regs);
        self.inputs
    }
}

/// A structure for tracking bits used in the circuit being encoded.
///
/// Nb: Although `tket-json-rs` has a "Register" struct, it's actually
/// an identifier for single bits in the `Register::0` register.
/// We rename it to `RegisterUnit` here to avoid confusion.
#[derive(Debug, Clone, Default)]
struct BitTracker {
    /// The ordered TKET1 names for the bit inputs, and their identifying wire
    /// from the circuit input node.
    inputs: Vec<RegisterUnit>,
    /// Map each bit wire to a TKET1 register element.
    bit_to_reg: HashMap<Wire, RegisterUnit>,
    /// Unused input bits, that can be written to.
    unused_inputs: VecDeque<RegisterUnit>,
}

impl BitTracker {
    /// Create a new [`BitTracker`] from the bit inputs of a [`Circuit`].
    /// Reads the [`METADATA_B_REGISTERS`] metadata entry with preset pytket bit register names.
    ///
    /// If the circuit contains more bit inputs than the provided list,
    /// new registers are created for the remaining bits.
    ///
    /// TODO: Compute output bit permutations when finishing the circuit.
    pub fn new(circ: &Circuit<impl HugrView>) -> Self {
        let mut tracker = BitTracker::default();

        if let Some(input_regs) = circ
            .hugr()
            .get_metadata(circ.parent(), METADATA_B_REGISTERS)
        {
            tracker.inputs = serde_json::from_value(input_regs.clone()).unwrap();
        }

        let bit_input_wires = circ.units().filter_map(|u| match u {
            (CircuitUnit::Wire(w), _, ty) if ty == BOOL_T => Some(w),
            _ => None,
        });

        for (i, wire) in bit_input_wires.enumerate() {
            // Use the given input register names if available, or create new ones.
            let reg = if let Some(reg) = tracker.inputs.get(i) {
                tracker.bit_to_reg.insert(wire, reg.clone());
                reg
            } else {
                let reg = tracker.add_bit_register(wire).clone();
                tracker.inputs.push(reg);
                tracker.inputs.last().unwrap()
            };

            // If the input is not used in the circuit, store it as a bit that can be written to.
            if circ
                .hugr()
                .linked_inputs(wire.node(), wire.source())
                .next()
                .is_none()
            {
                tracker.unused_inputs.push_back(reg.clone());
            }
        }

        tracker
    }

    /// Add a new register unit for a bit wire.
    pub fn add_bit_register(&mut self, wire: Wire) -> &RegisterUnit {
        let reg = match self.unused_inputs.pop_front() {
            Some(reg) => reg,
            None => RegisterUnit("c".to_string(), vec![self.bit_to_reg.len() as i64]),
        };
        self.bit_to_reg.insert(wire, reg);
        self.bit_to_reg.get(&wire).unwrap()
    }

    /// Set the register for a bit wire
    pub fn set_bit_register(&mut self, wire: Wire, reg: RegisterUnit) {
        self.bit_to_reg.insert(wire, reg);
    }

    /// Returns the register unit for a bit wire, if it exists.
    pub fn get(&self, wire: &Wire) -> Option<&RegisterUnit> {
        self.bit_to_reg.get(wire)
    }

    /// Consumes the tracker and returns the final list of bit registers.
    pub fn finish(mut self) -> Vec<RegisterUnit> {
        let mut missing_regs: HashSet<_> = self.bit_to_reg.into_values().collect();
        for reg in &self.inputs {
            missing_regs.remove(reg);
        }
        self.inputs.extend(missing_regs);
        self.inputs
    }
}

/// A structure for tracking the parameters of a circuit being encoded.
#[derive(Debug, Clone, Default)]
struct ParameterTracker {
    /// The parameters associated with each wire.
    parameters: HashMap<Wire, String>,
}

impl ParameterTracker {
    /// Create a new [`ParameterTracker`] from the input parameters of a [`Circuit`].
    fn new(circ: &Circuit<impl HugrView>) -> Self {
        let mut tracker = ParameterTracker::default();

        let float_input_wires = circ.units().filter_map(|u| match u {
            (CircuitUnit::Wire(w), _, ty) if ty == FLOAT64_TYPE => Some(w),
            _ => None,
        });

        for (i, wire) in float_input_wires.enumerate() {
            tracker.add_parameter(wire, format!("f{i}"));
        }

        tracker
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

    /// Associate a parameter expression with a wire.
    fn add_parameter(&mut self, wire: Wire, param: String) {
        self.parameters.insert(wire, param);
    }

    /// Returns the parameter expression for a wire, if it exists.
    fn get(&self, wire: &Wire) -> Option<&String> {
        self.parameters.get(wire)
    }
}
