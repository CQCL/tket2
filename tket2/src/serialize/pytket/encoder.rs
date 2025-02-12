//! Intermediate structure for encoding [`Circuit`]s into [`SerialCircuit`]s.

mod bit_tracker;
mod param_tracker;
mod qubit_tracker;
mod unsupported_tracker;

use core::panic;

use bit_tracker::BitTracker;
use param_tracker::ParameterTracker;
use qubit_tracker::QubitTracker;

use hugr::extension::prelude::{bool_t, qb_t};
use hugr::ops::OpType;
use hugr::std_extensions::arithmetic::float_types::float64_type;
use hugr::HugrView;
use itertools::Itertools;
use tket_json_rs::circuit_json::{self, SerialCircuit};
use tket_json_rs::register::ElementId as RegisterUnit;

use crate::circuit::command::{CircuitUnit, Command};
use crate::circuit::Circuit;
use crate::extension::rotation::rotation_type;
use crate::Tk2Op;

use super::op::Tk1Op;
use super::{
    OpConvertError, TK1ConvertError, METADATA_B_OUTPUT_REGISTERS, METADATA_B_REGISTERS,
    METADATA_OPGROUP, METADATA_PHASE, METADATA_Q_OUTPUT_REGISTERS, METADATA_Q_REGISTERS,
};

/// The state of an in-progress [`SerialCircuit`] being built from a [`Circuit`].
#[derive(Debug, Clone)]
pub(super) struct Tk1Encoder {
    /// The name of the circuit being encoded.
    name: Option<String>,
    /// Global phase value.
    ///
    /// Defaults to "0" unless the circuit has a [METADATA_PHASE] metadata
    /// entry.
    phase: String,
    /// The already-encoded serialised pytket commands.
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
            if ![rotation_type(), float64_type(), qb_t(), bool_t()].contains(&typ) {
                return Err(TK1ConvertError::NonSerializableInputs { typ });
            }
        }

        // Recover other parameters stored in the metadata
        // TODO: Check for invalid encoded metadata
        let phase = match hugr.get_metadata(circ.parent(), METADATA_PHASE) {
            Some(p) => p.as_str().unwrap().to_string(),
            None => "0".to_string(),
        };

        let qubit_tracker = QubitTracker::new(circ);
        let bit_tracker = BitTracker::new(circ);
        let parameter_tracker = ParameterTracker::new(circ);

        Ok(Self {
            name,
            phase,
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
        if self.parameters.record_parameters(&command, optype)? {
            // for now all ops that record parameters should be ignored (are
            // just constants)
            return Ok(());
        }

        // Special case for the QAlloc operation.
        // This does not translate to a TKET1 operation, we just start tracking a new qubit register.
        if optype == &Tk2Op::QAlloc.into() {
            let Some((CircuitUnit::Linear(unit_id), _, _)) = command.outputs().next() else {
                panic!("QAlloc should have a single qubit output.")
            };
            debug_assert!(self.qubits.get(unit_id).is_none());
            self.qubits.add_qubit_register(unit_id);
            return Ok(());
        }

        let Some(tk1op) = Tk1Op::try_from_optype(optype.clone())? else {
            // This command should be ignored.
            return Ok(());
        };

        // Get the registers and wires associated with the operation's inputs.
        let mut qubit_args = Vec::with_capacity(tk1op.qubit_inputs());
        let mut bit_args = Vec::with_capacity(tk1op.bit_inputs());
        let mut params = Vec::with_capacity(tk1op.num_params());
        for (unit, _, ty) in command.inputs() {
            if ty == qb_t() {
                let reg = self.unit_to_register(unit).unwrap_or_else(|| {
                    panic!(
                        "No register found for qubit input {unit} in node {}.",
                        command.node(),
                    )
                });
                qubit_args.push(reg);
            } else if ty == bool_t() {
                let reg = self.unit_to_register(unit).unwrap_or_else(|| {
                    panic!(
                        "No register found for bit input {unit} in node {}.",
                        command.node(),
                    )
                });
                bit_args.push(reg);
            } else if [rotation_type(), float64_type()].contains(&ty) {
                let CircuitUnit::Wire(param_wire) = unit else {
                    unreachable!("Angle types are not linear.")
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

        for (unit, _, ty) in command.outputs() {
            if ty == qb_t() {
                // If the qubit is not already in the qubit tracker, add it as a
                // new register.
                let CircuitUnit::Linear(unit_id) = unit else {
                    panic!("Qubit types are linear.")
                };
                if self.qubits.get(unit_id).is_none() {
                    let reg = self.qubits.add_qubit_register(unit_id);
                    qubit_args.push(reg.clone());
                }
            } else if ty == bool_t() {
                // If the operation has any bit outputs, create a new one bit
                // register.
                //
                // Note that we do not reassign input registers to the new
                // output wires as we do not know if the bit value was modified
                // by the operation, and the old value may be needed later.
                //
                // This may cause register duplication for opaque operations
                // with input bits.
                let CircuitUnit::Wire(wire) = unit else {
                    panic!("Bool types are not linear.")
                };
                let reg = self.bits.add_bit_register(wire);
                bit_args.push(reg.clone());
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
    pub fn finish(self, circ: &Circuit<impl HugrView>) -> SerialCircuit {
        let (qubits, qubits_permutation) = self.qubits.finish(circ);
        let (bits, mut bits_permutation) = self.bits.finish(circ);

        let mut implicit_permutation = qubits_permutation;
        implicit_permutation.append(&mut bits_permutation);

        let mut ser = SerialCircuit::new(self.name, self.phase);

        ser.commands = self.commands;
        ser.qubits = qubits.into_iter().map_into().collect();
        ser.bits = bits.into_iter().map_into().collect();
        ser.implicit_permutation = implicit_permutation;
        ser.number_of_ws = None;
        ser
    }

    /// Translate a linear [`CircuitUnit`] into a [`RegisterUnit`], if possible.
    fn unit_to_register(&self, unit: CircuitUnit) -> Option<RegisterUnit> {
        match unit {
            CircuitUnit::Linear(i) => self.qubits.get(i).cloned(),
            CircuitUnit::Wire(wire) => self.bits.get(&wire).cloned(),
        }
    }
}

/// A utility class for finding new unused qubit/bit names.
#[derive(Debug, Clone, Default)]
struct RegisterUnitGenerator {
    /// The next index to use for a new register.
    next_unit: u16,
    /// The register name to use.
    register: String,
}

impl RegisterUnitGenerator {
    /// Create a new [`RegisterUnitGenerator`]
    ///
    /// Scans the set of existing registers to find the last used index, and
    /// starts generating new unit names from there.
    pub fn new<'a>(
        register: impl ToString,
        existing: impl IntoIterator<Item = &'a RegisterUnit>,
    ) -> Self {
        let register = register.to_string();
        let mut last_unit: Option<u16> = None;
        for reg in existing {
            if reg.0 != register {
                continue;
            }
            last_unit = Some(last_unit.unwrap_or_default().max(reg.1[0] as u16));
        }
        RegisterUnitGenerator {
            register,
            next_unit: last_unit.map_or(0, |i| i + 1),
        }
    }

    /// Returns a fresh register unit.
    pub fn next(&mut self) -> RegisterUnit {
        let unit = self.next_unit;
        self.next_unit += 1;
        RegisterUnit(self.register.clone(), vec![unit as i64])
    }
}
