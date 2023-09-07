//! Iterators over the units of a circuit.

use std::iter::FusedIterator;

use hugr::extension::prelude;
use hugr::hugr::CircuitUnit;
use hugr::types::{Type, TypeBound, TypeRow};
use hugr::{Node, Port, Wire};

use super::Circuit;

/// An iterator over the units of a circuit.
pub struct Units<'a> {
    /// Whether to only
    output_mode: UnitType,
    /// The inputs to the circuit
    inputs: Option<&'a TypeRow>,
    /// Input node of the circuit
    input_node: Node,
    /// The current index in the inputs
    current: usize,
    /// The amount of linear units yielded.
    linear_count: usize,
}

impl<'a> Units<'a> {
    /// Create a new iterator over the units of a circuit.
    pub(super) fn new(circuit: &'a impl Circuit, output_mode: UnitType) -> Self {
        Self {
            output_mode,
            inputs: circuit.get_function_type().map(|ft| &ft.input),
            input_node: circuit.input(),
            current: 0,
            linear_count: 0,
        }
    }

    /// Construct an output value to yield.
    fn make_value(&self, typ: &Type, input_port: Port) -> (CircuitUnit, Type) {
        match type_is_linear(typ) {
            true => (CircuitUnit::Linear(self.linear_count - 1), typ.clone()),
            false => (
                CircuitUnit::Wire(Wire::new(self.input_node, input_port)),
                typ.clone(),
            ),
        }
    }
}

impl<'a> Iterator for Units<'a> {
    type Item = (CircuitUnit, Type);

    fn next(&mut self) -> Option<Self::Item> {
        let inputs = self.inputs?;
        loop {
            let typ = inputs.get(self.current)?;
            let input_port = Port::new_outgoing(self.current);
            self.current += 1;
            if type_is_linear(typ) {
                self.linear_count += 1;
            }
            if self.output_mode.accept(typ) {
                return Some(self.make_value(typ, input_port));
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self
            .inputs
            .map(|inputs| inputs.len() - self.current)
            .unwrap_or(0);
        match self.output_mode {
            UnitType::All => (len, Some(len)),
            _ => (0, Some(len)),
        }
    }
}

impl<'a> FusedIterator for Units<'a> {}

/// What kind of units to iterate over.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum UnitType {
    /// All units.
    All,
    /// Only the linear units.
    Linear,
    /// Only the qubit units.
    Qubits,
    /// Only the non-linear units.
    NonLinear,
}

impl UnitType {
    /// Check if a [`Type`] should be yielded.
    pub fn accept(self, typ: &Type) -> bool {
        match self {
            UnitType::All => true,
            UnitType::Linear => type_is_linear(typ),
            UnitType::Qubits => *typ == prelude::QB_T,
            UnitType::NonLinear => !type_is_linear(typ),
        }
    }
}

fn type_is_linear(typ: &Type) -> bool {
    !TypeBound::Copyable.contains(typ.least_upper_bound())
}
