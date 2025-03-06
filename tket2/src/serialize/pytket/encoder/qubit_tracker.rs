//! Tracking of encoded qubit wires during pytket circuit encoding.

use std::collections::{HashMap, HashSet};

use hugr::extension::prelude::qb_t;
use hugr::{HugrView, Wire};
use itertools::Itertools;
use tket_json_rs::circuit_json;
use tket_json_rs::register::ElementId as RegisterUnit;

use crate::circuit::Circuit;
use crate::serialize::pytket::RegisterHash;

use super::{RegisterUnitGenerator, METADATA_Q_OUTPUT_REGISTERS, METADATA_Q_REGISTERS};

/// A structure for tracking qubits used in the circuit being encoded.
///
/// Nb: Although `tket-json-rs` has a "Register" struct, it's actually
/// an identifier for single qubits in the `Register::0` register.
/// We rename it to `RegisterUnit` here to avoid confusion.
#[derive(Debug, Clone, Default)]
pub struct QubitTracker {
    /// The ordered TKET1 names for the input qubit registers.
    inputs: Vec<RegisterUnit>,
    /// The ordered TKET1 names for the output qubit registers.
    outputs: Option<Vec<RegisterUnit>>,
    /// The TKET1 qubit registers associated to each unused qubit wire.
    qubit_to_reg: HashMap<Wire, RegisterUnit>,
    /// The set of all qubit ids used in the circuit.
    ///
    /// This includes ids that have been allocated and dropped mid-circuit,
    /// and are not the `qubit_to_reg` list any more.
    seen_qubits: HashSet<RegisterUnit>,
    /// A generator of new registers units to use for bit wires.
    unit_generator: RegisterUnitGenerator,
}

impl QubitTracker {
    /// Create a new [`QubitTracker`] from the qubit inputs of a [`Circuit`].
    /// Reads the [`METADATA_Q_REGISTERS`] metadata entry with preset pytket qubit register names.
    ///
    /// If the circuit contains more qubit inputs than the provided list,
    /// new registers are created for the remaining qubits.
    pub fn new(circ: &Circuit<impl HugrView>) -> Self {
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

        // Collect all the qubit ids seen in the input/outputs, and initialize the unit generator.
        tracker.seen_qubits.extend(
            tracker
                .inputs
                .iter()
                .chain(tracker.outputs.iter().flatten())
                .cloned(),
        );
        tracker.unit_generator = RegisterUnitGenerator::new("q", &tracker.seen_qubits);

        let input_node = circ.input_node();
        let qubit_t = qb_t();
        for (i, ty) in circ.circuit_signature().output().iter().enumerate() {
            if ty != &qubit_t {
                continue;
            }
            let wire = Wire::new(input_node, i);
            // Use the given input register names if available, or create new ones.
            if let Some(reg) = tracker.inputs.get(i) {
                tracker.qubit_to_reg.insert(wire, reg.clone());
            } else {
                let reg = tracker.init_qubit(wire).clone();
                tracker.inputs.push(reg);
            }
        }

        tracker
    }

    /// Add a new register unit for a qubit wire.
    pub fn init_qubit(&mut self, wire: Wire) -> &RegisterUnit {
        let reg = self.unit_generator.next();
        self.seen_qubits.insert(reg.clone());
        self.qubit_to_reg.insert(wire, reg);
        self.qubit_to_reg.get(&wire).unwrap()
    }

    /// Returns the register unit for a qubit wire, if it exists.
    pub fn get(&self, wire: Wire) -> Option<&RegisterUnit> {
        self.qubit_to_reg.get(&wire)
    }

    /// Returns the register unit associated with a wire and updates it to a new
    /// wire.
    ///
    /// Returns the register unit if it exists, or `None` if it doesn't.
    pub fn update(&mut self, wire: Wire, updated_wire: Wire) -> Option<&RegisterUnit> {
        let reg = self.qubit_to_reg.remove(&wire)?;
        self.qubit_to_reg.insert(updated_wire, reg);
        Some(self.qubit_to_reg.get(&updated_wire).unwrap())
    }

    /// Consumes the tracker and returns the final list of qubit registers, along
    /// with the final permutation of the outputs.
    pub fn finish(
        self,
        circ: &Circuit<impl HugrView>,
    ) -> (Vec<RegisterUnit>, Vec<circuit_json::ImplicitPermutation>) {
        // Collect the qubit registers at the output of the circuit, in the
        // order they appear.
        let mut circuit_output_order: Vec<RegisterUnit> = Vec::with_capacity(self.inputs.len());
        for (node, port) in circ.hugr().all_linked_outputs(circ.output_node()) {
            let wire = Wire::new(node, port);
            if let Some(reg) = self.qubit_to_reg.get(&wire) {
                circuit_output_order.push(reg.clone());
            }
        }

        compute_permutation(
            circuit_output_order,
            self.inputs,
            self.outputs.unwrap_or_default(),
            self.seen_qubits,
        )
    }
}

/// Compute the final unit permutation for a circuit.
///
/// Arguments:
/// - `output_order`: The final order of output registers, computed from the
///   wires at the output node.
/// - `declared_inputs`: The list of input registers declared at the start of
///   the circuit.
/// - `declared_outputs`: The list of output registers declared at the start of
///   the circuit, potentially in a different order than `declared_inputs`.
/// - `seen_registers`: The set of all registers seen in the circuit, including
///     those that were allocated and dropped mid-circuit.
///
/// Returns:
/// - The complete list of registers used throughout the circuit.
/// - The final permutation of the output registers.
pub(super) fn compute_permutation(
    mut circuit_output_order: Vec<RegisterUnit>,
    mut declared_inputs: Vec<RegisterUnit>,
    mut declared_outputs: Vec<RegisterUnit>,
    seen_registers: HashSet<RegisterUnit>,
) -> (Vec<RegisterUnit>, Vec<circuit_json::ImplicitPermutation>) {
    // Ensure the input and output lists have the same registers.
    let mut input_regs: HashSet<RegisterHash> =
        declared_inputs.iter().map(RegisterHash::from).collect();
    let output_regs: HashSet<RegisterHash> =
        declared_outputs.iter().map(RegisterHash::from).collect();
    for inp in &declared_inputs {
        if !output_regs.contains(&inp.into()) {
            declared_outputs.push(inp.clone());
        }
    }
    for out in &declared_outputs {
        if !input_regs.contains(&out.into()) {
            declared_inputs.push(out.clone());
        }
    }
    input_regs.extend(output_regs);

    // Add registers defined mid-circuit to both ends.
    for reg in seen_registers {
        if !input_regs.contains(&(&reg).into()) {
            declared_inputs.push(reg.clone());
            declared_outputs.push(reg);
        }
    }

    // And ensure that `circuit_output_order` has all the new registers added too.
    let circuit_outputs: HashSet<RegisterHash> = circuit_output_order
        .iter()
        .map(RegisterHash::from)
        .collect();
    for out in &declared_outputs {
        if !circuit_outputs.contains(&out.into()) {
            circuit_output_order.push(out.clone());
        }
    }

    // Compute the final permutation. This is a combination of two mappings:
    // - First, the original implicit permutation for the circuit, if this was decoded from pytket.
    let original_permutation: HashMap<RegisterUnit, RegisterHash> = declared_inputs
        .iter()
        .zip(&declared_outputs)
        .map(|(inp, out)| (inp.clone(), RegisterHash::from(out)))
        .collect();
    // - Second, the actual reordering of outputs seen at the circuit's output node.
    let mut circuit_permutation: HashMap<RegisterHash, RegisterUnit> = declared_outputs
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

    (declared_inputs, permutation)
}
