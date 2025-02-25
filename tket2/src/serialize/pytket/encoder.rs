//! Intermediate structure for encoding [`Circuit`]s into [`SerialCircuit`]s.

use core::panic;
use std::collections::{HashMap, HashSet, VecDeque};

use hugr::extension::prelude::{bool_t, qb_t};
use hugr::ops::{OpTrait, OpType};
use hugr::std_extensions::arithmetic::float_types::float64_type;
use hugr::types::Type;
use hugr::{HugrView, Node, Wire};
use itertools::Itertools;
use tket_json_rs::circuit_json::{self, SerialCircuit};
use tket_json_rs::register::ElementId as RegisterUnit;

use crate::circuit::command::{CircuitUnit, Command};
use crate::circuit::Circuit;
use crate::extension::rotation::rotation_type;
use crate::serialize::pytket::RegisterHash;
use crate::Tk2Op;

use super::op::Tk1Op;
use super::param::encode::fold_param_op;
use super::{
    OpConvertError, TK1ConvertError, METADATA_B_OUTPUT_REGISTERS, METADATA_B_REGISTERS,
    METADATA_INPUT_PARAMETERS, METADATA_OPGROUP, METADATA_PHASE, METADATA_Q_OUTPUT_REGISTERS,
    METADATA_Q_REGISTERS,
};

/// The state of an in-progress [`SerialCircuit`] being built from a [`Circuit`].
#[derive(Debug, Clone)]
pub(super) struct Tk1Encoder {
    /// The name of the circuit being encoded.
    name: Option<String>,
    /// Global phase value. Defaults to "0"
    phase: String,
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
    pub fn new(circ: &Circuit<impl HugrView<Node = Node>>) -> Result<Self, TK1ConvertError> {
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
    pub fn add_command<T: HugrView<Node = Node>>(
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
    pub fn finish(self, circ: &Circuit<impl HugrView<Node = Node>>) -> SerialCircuit {
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

/// A structure for tracking qubits used in the circuit being encoded.
///
/// Nb: Although `tket-json-rs` has a "Register" struct, it's actually
/// an identifier for single qubits in the `Register::0` register.
/// We rename it to `RegisterUnit` here to avoid confusion.
#[derive(Debug, Clone, Default)]
struct QubitTracker {
    /// The ordered TKET1 names for the input qubit registers.
    inputs: Vec<RegisterUnit>,
    /// The ordered TKET1 names for the output qubit registers.
    outputs: Option<Vec<RegisterUnit>>,
    /// The TKET1 qubit registers associated to each qubit unit of the circuit.
    qubit_to_reg: HashMap<usize, RegisterUnit>,
    /// A generator of new registers units to use for bit wires.
    unit_generator: RegisterUnitGenerator,
}

impl QubitTracker {
    /// Create a new [`QubitTracker`] from the qubit inputs of a [`Circuit`].
    /// Reads the [`METADATA_Q_REGISTERS`] metadata entry with preset pytket qubit register names.
    ///
    /// If the circuit contains more qubit inputs than the provided list,
    /// new registers are created for the remaining qubits.
    pub fn new(circ: &Circuit<impl HugrView<Node = Node>>) -> Self {
        let mut tracker = QubitTracker::default();

        if let Some(input_regs) = circ
            .hugr()
            .get_metadata(circ.parent(), METADATA_Q_REGISTERS)
        {
            tracker.inputs = serde_json::from_value(input_regs.clone()).unwrap();
        }
        let output_regs = circ
            .hugr()
            .get_metadata(circ.parent(), METADATA_Q_OUTPUT_REGISTERS)
            .map(|regs| serde_json::from_value(regs.clone()).unwrap());
        if let Some(output_regs) = output_regs {
            tracker.outputs = Some(output_regs);
        }

        tracker.unit_generator = RegisterUnitGenerator::new(
            "q",
            tracker
                .inputs
                .iter()
                .chain(tracker.outputs.iter().flatten()),
        );

        let qubit_count = circ.units().filter(|(_, _, ty)| ty == &qb_t()).count();

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

    /// Add a new register unit for a qubit wire.
    pub fn add_qubit_register(&mut self, unit_id: usize) -> &RegisterUnit {
        let reg = self.unit_generator.next();
        self.qubit_to_reg.insert(unit_id, reg);
        self.qubit_to_reg.get(&unit_id).unwrap()
    }

    /// Returns the register unit for a qubit wire, if it exists.
    pub fn get(&self, unit_id: usize) -> Option<&RegisterUnit> {
        self.qubit_to_reg.get(&unit_id)
    }

    /// Consumes the tracker and returns the final list of qubit registers, along
    /// with the final permutation of the outputs.
    pub fn finish(
        mut self,
        _circ: &Circuit<impl HugrView<Node = Node>>,
    ) -> (Vec<RegisterUnit>, Vec<circuit_json::ImplicitPermutation>) {
        // Ensure the input and output lists have the same registers.
        let mut outputs = self.outputs.unwrap_or_default();
        let mut input_regs: HashSet<RegisterHash> =
            self.inputs.iter().map(RegisterHash::from).collect();
        let output_regs: HashSet<RegisterHash> = outputs.iter().map(RegisterHash::from).collect();

        for inp in &self.inputs {
            if !output_regs.contains(&inp.into()) {
                outputs.push(inp.clone());
            }
        }
        for out in &outputs {
            if !input_regs.contains(&out.into()) {
                self.inputs.push(out.clone());
            }
        }
        input_regs.extend(output_regs);

        // Add registers defined mid-circuit to both ends.
        for reg in self.qubit_to_reg.into_values() {
            if !input_regs.contains(&(&reg).into()) {
                self.inputs.push(reg.clone());
                outputs.push(reg);
            }
        }

        // TODO: Look at the circuit outputs to determine the final permutation.
        //
        // We don't have the `CircuitUnit::Linear` assignments for the outputs
        // here, so that requires some extra piping.
        let permutation = outputs
            .into_iter()
            .zip(&self.inputs)
            .map(|(out, inp)| circuit_json::ImplicitPermutation(inp.clone().into(), out.into()))
            .collect_vec();

        (self.inputs, permutation)
    }
}

/// A structure for tracking bits used in the circuit being encoded.
///
/// Nb: Although `tket-json-rs` has a "Register" struct, it's actually
/// an identifier for single bits in the `Register::0` register.
/// We rename it to `RegisterUnit` here to avoid confusion.
#[derive(Debug, Clone, Default)]
struct BitTracker {
    /// The ordered TKET1 names for the bit inputs.
    inputs: Vec<RegisterUnit>,
    /// The expected order of TKET1 names for the bit outputs,
    /// if that was stored in the metadata.
    outputs: Option<Vec<RegisterUnit>>,
    /// Map each bit wire to a TKET1 register element.
    bit_to_reg: HashMap<Wire, RegisterUnit>,
    /// Registers defined in the metadata, but not present in the circuit
    /// inputs.
    unused_registers: VecDeque<RegisterUnit>,
    /// A generator of new registers units to use for bit wires.
    unit_generator: RegisterUnitGenerator,
}

impl BitTracker {
    /// Create a new [`BitTracker`] from the bit inputs of a [`Circuit`].
    /// Reads the [`METADATA_B_REGISTERS`] metadata entry with preset pytket bit register names.
    ///
    /// If the circuit contains more bit inputs than the provided list,
    /// new registers are created for the remaining bits.
    ///
    /// TODO: Compute output bit permutations when finishing the circuit.
    pub fn new(circ: &Circuit<impl HugrView<Node = Node>>) -> Self {
        let mut tracker = BitTracker::default();

        if let Some(input_regs) = circ
            .hugr()
            .get_metadata(circ.parent(), METADATA_B_REGISTERS)
        {
            tracker.inputs = serde_json::from_value(input_regs.clone()).unwrap();
        }
        let output_regs = circ
            .hugr()
            .get_metadata(circ.parent(), METADATA_B_OUTPUT_REGISTERS)
            .map(|regs| serde_json::from_value(regs.clone()).unwrap());
        if let Some(output_regs) = output_regs {
            tracker.outputs = Some(output_regs);
        }

        tracker.unit_generator = RegisterUnitGenerator::new(
            "c",
            tracker
                .inputs
                .iter()
                .chain(tracker.outputs.iter().flatten()),
        );

        let bit_input_wires = circ.units().filter_map(|u| match u {
            (CircuitUnit::Wire(w), _, ty) if ty == bool_t() => Some(w),
            _ => None,
        });

        let mut unused_registers: HashSet<RegisterUnit> = tracker.inputs.iter().cloned().collect();
        for (i, wire) in bit_input_wires.enumerate() {
            // If the input is not used in the circuit, ignore it.
            if circ
                .hugr()
                .linked_inputs(wire.node(), wire.source())
                .next()
                .is_none()
            {
                continue;
            }

            // Use the given input register names if available, or create new ones.
            if let Some(reg) = tracker.inputs.get(i) {
                unused_registers.remove(reg);
                tracker.bit_to_reg.insert(wire, reg.clone());
            } else {
                let reg = tracker.add_bit_register(wire).clone();
                tracker.inputs.push(reg);
            };
        }

        // If a register was defined in the metadata but not used in the circuit,
        // we keep it so it can be assigned to an operation output.
        tracker.unused_registers = unused_registers.into_iter().collect();

        tracker
    }

    /// Add a new register unit for a bit wire.
    pub fn add_bit_register(&mut self, wire: Wire) -> &RegisterUnit {
        let reg = self
            .unused_registers
            .pop_front()
            .unwrap_or_else(|| self.unit_generator.next());

        self.bit_to_reg.insert(wire, reg);
        self.bit_to_reg.get(&wire).unwrap()
    }

    /// Returns the register unit for a bit wire, if it exists.
    pub fn get(&self, wire: &Wire) -> Option<&RegisterUnit> {
        self.bit_to_reg.get(wire)
    }

    /// Consumes the tracker and returns the final list of bit registers, along
    /// with the final permutation of the outputs.
    pub fn finish(
        mut self,
        circ: &Circuit<impl HugrView<Node = Node>>,
    ) -> (Vec<RegisterUnit>, Vec<circuit_json::ImplicitPermutation>) {
        let mut circuit_output_order: Vec<RegisterUnit> = Vec::with_capacity(self.inputs.len());
        for (node, port) in circ.hugr().all_linked_outputs(circ.output_node()) {
            let wire = Wire::new(node, port);
            if let Some(reg) = self.bit_to_reg.get(&wire) {
                circuit_output_order.push(reg.clone());
            }
        }

        // Ensure the input and output lists have the same registers.
        let mut outputs = self.outputs.unwrap_or_default();
        let mut input_regs: HashSet<RegisterHash> =
            self.inputs.iter().map(RegisterHash::from).collect();
        let output_regs: HashSet<RegisterHash> = outputs.iter().map(RegisterHash::from).collect();

        for inp in &self.inputs {
            if !output_regs.contains(&inp.into()) {
                outputs.push(inp.clone());
            }
        }
        for out in &outputs {
            if !input_regs.contains(&out.into()) {
                self.inputs.push(out.clone());
            }
        }
        input_regs.extend(output_regs);

        // Add registers defined mid-circuit to both ends.
        for reg in self.bit_to_reg.into_values() {
            if !input_regs.contains(&(&reg).into()) {
                self.inputs.push(reg.clone());
                outputs.push(reg);
            }
        }

        // And ensure `circuit_output_order` has all virtual registers added too.
        let circuit_outputs: HashSet<RegisterHash> = circuit_output_order
            .iter()
            .map(RegisterHash::from)
            .collect();
        for out in &outputs {
            if !circuit_outputs.contains(&out.into()) {
                circuit_output_order.push(out.clone());
            }
        }

        // Compute the final permutation. This is a combination of two mappings:
        // - First, the original implicit permutation for the circuit, if this was decoded from pytket.
        let original_permutation: HashMap<RegisterUnit, RegisterHash> = self
            .inputs
            .iter()
            .zip(&outputs)
            .map(|(inp, out)| (inp.clone(), RegisterHash::from(out)))
            .collect();
        // - Second, the actual reordering of outputs seen at the circuit's output node.
        let mut circuit_permutation: HashMap<RegisterHash, RegisterUnit> = outputs
            .iter()
            .zip(circuit_output_order)
            .map(|(out, circ_out)| (RegisterHash::from(out), circ_out))
            .collect();
        // The final permutation is the composition of these two mappings.
        let permutation = original_permutation
            .into_iter()
            .map(|(inp, out)| {
                circuit_json::ImplicitPermutation(
                    inp.into(),
                    circuit_permutation.remove(&out).unwrap().into(),
                )
            })
            .collect_vec();

        (self.inputs, permutation)
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
    fn new(circ: &Circuit<impl HugrView<Node = Node>>) -> Self {
        let mut tracker = ParameterTracker::default();

        let angle_input_wires = circ.units().filter_map(|u| match u {
            (CircuitUnit::Wire(w), _, ty) if [rotation_type(), float64_type()].contains(&ty) => {
                Some(w)
            }
            _ => None,
        });

        // The input parameter names may be specified in the metadata.
        let fixed_input_names: Vec<String> = circ
            .hugr()
            .get_metadata(circ.parent(), METADATA_INPUT_PARAMETERS)
            .and_then(|params| serde_json::from_value(params.clone()).ok())
            .unwrap_or_default();
        let extra_names = (fixed_input_names.len()..).map(|i| format!("f{i}"));
        let mut param_name = fixed_input_names.into_iter().chain(extra_names);

        for wire in angle_input_wires {
            tracker.add_parameter(wire, param_name.next().unwrap());
        }

        tracker
    }

    /// Record any output of the command that can be used as a TKET1 parameter.
    /// Returns whether parameters were recorded.
    /// Associates the output wires with the parameter expression.
    fn record_parameters<T: HugrView<Node = Node>>(
        &mut self,
        command: &Command<'_, T>,
        optype: &OpType,
    ) -> Result<bool, OpConvertError> {
        let input_count = if let Some(signature) = optype.dataflow_signature() {
            // Only consider commands where all inputs and some outputs are
            // parameters that we can track.
            let tracked_params: [Type; 2] = [rotation_type(), float64_type()];
            let all_inputs = signature
                .input()
                .iter()
                .all(|ty| tracked_params.contains(ty));
            let some_output = signature
                .output()
                .iter()
                .any(|ty| tracked_params.contains(ty));
            if !all_inputs || !some_output {
                return Ok(false);
            }
            signature.input_count()
        } else if let OpType::Const(_) = optype {
            // `Const` is a special non-dataflow command we can handle.
            // It has zero inputs.
            0
        } else {
            // Not a parameter-generating command.
            return Ok(false);
        };

        // Collect the input parameters.
        let mut inputs = Vec::with_capacity(input_count);
        for (unit, _, _) in command.inputs() {
            let CircuitUnit::Wire(wire) = unit else {
                panic!("Angle types are not linear")
            };
            let Some(param) = self.parameters.get(&wire) else {
                let typ = rotation_type();
                return Err(OpConvertError::UnresolvedParamInput {
                    typ,
                    optype: optype.clone(),
                    node: command.node(),
                });
            };
            inputs.push(param.as_str());
        }

        let Some(param) = fold_param_op(optype, &inputs) else {
            return Ok(false);
        };

        for (unit, _, _) in command.outputs() {
            if let CircuitUnit::Wire(wire) = unit {
                self.add_parameter(wire, param.clone())
            }
        }
        Ok(true)
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
