//! Tracking of encoded qubit wires during pytket circuit encoding.

use std::collections::{HashMap, HashSet};

use hugr::extension::prelude::qb_t;
use hugr::HugrView;
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
        _circ: &Circuit<impl HugrView>,
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
