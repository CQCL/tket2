//! Tracking of encoded bit wires during pytket circuit encoding.

use std::collections::{HashMap, HashSet, VecDeque};

use hugr::extension::prelude::bool_t;
use hugr::{CircuitUnit, HugrView, Wire};
use tket_json_rs::circuit_json;
use tket_json_rs::register::ElementId as RegisterUnit;

use crate::circuit::Circuit;

use super::{RegisterUnitGenerator, METADATA_B_OUTPUT_REGISTERS, METADATA_B_REGISTERS};

/// A structure for tracking bits used in the circuit being encoded.
///
/// Nb: Although `tket-json-rs` has a "Register" struct, it's actually
/// an identifier for single bits in the `Register::0` register.
/// We rename it to `RegisterUnit` here to avoid confusion.
#[derive(Debug, Clone, Default)]
pub struct BitTracker {
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
    pub fn new(circ: &Circuit<impl HugrView>) -> Self {
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
    pub fn get(&self, wire: Wire) -> Option<&RegisterUnit> {
        self.bit_to_reg.get(&wire)
    }

    /// Consumes the tracker and returns the final list of bit registers, along
    /// with the final permutation of the outputs.
    pub fn finish(
        self,
        circ: &Circuit<impl HugrView>,
    ) -> (Vec<RegisterUnit>, Vec<circuit_json::ImplicitPermutation>) {
        let mut circuit_output_order: Vec<RegisterUnit> = Vec::with_capacity(self.inputs.len());
        for (node, port) in circ.hugr().all_linked_outputs(circ.output_node()) {
            let wire = Wire::new(node, port);
            if let Some(reg) = self.bit_to_reg.get(&wire) {
                circuit_output_order.push(reg.clone());
            }
        }

        super::qubit_tracker::compute_permutation(
            circuit_output_order,
            self.inputs,
            self.outputs.unwrap_or_default(),
            self.bit_to_reg.into_values().collect(),
        )
    }
}
