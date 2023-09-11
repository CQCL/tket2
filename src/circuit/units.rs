//! Iterators over the units of a circuit.

use std::iter::FusedIterator;

use hugr::extension::prelude;
use hugr::hugr::CircuitUnit;
use hugr::ops::OpTrait;
use hugr::types::{Type, TypeBound, TypeRow};
use hugr::{Node, Port, Wire};

use super::Circuit;

/// An iterator over the units at the output of a [Node].
#[derive(Clone, Debug)]
pub struct Units {
    /// Whether to only.
    output_mode: UnitType,
    /// The types of the node outputs.
    //
    // TODO: We could avoid cloning the TypeRow if `OpType::signature` returned
    // a reference.
    node_output_types: TypeRow,
    /// The node of the circuit.
    node: Node,
    /// The current index in the inputs.
    pub(self) current: usize,
    /// The amount of linear units yielded.
    linear_count: usize,
}

impl Units {
    /// Create a new iterator over the units of a node.
    //
    // FIXME: Currently this ignores any incoming linear unit labels, and just
    // assigns new ids sequentially.
    #[inline]
    #[allow(unused)]
    pub(super) fn new(circuit: &impl Circuit, node: Node, output_mode: UnitType) -> Self {
        Self {
            output_mode,
            node_output_types: circuit.get_optype(node).signature().output,
            node,
            current: 0,
            linear_count: 0,
        }
    }

    /// Create a new iterator over the input units of a circuit.
    ///
    /// This iterator will yield all units originating from the circuit's input
    /// node.
    #[inline]
    pub(super) fn new_input(circuit: &impl Circuit, output_mode: UnitType) -> Self {
        Self {
            output_mode,
            node_output_types: circuit
                .get_function_type()
                .map_or_else(Default::default, |ft| ft.input.clone()),
            node: circuit.input(),
            current: 0,
            linear_count: 0,
        }
    }

    /// Add the corresponding ports to the iterator output.
    #[inline]
    pub fn with_ports(self) -> UnitPorts {
        UnitPorts { units: self }
    }

    /// Construct an output value to yield.
    #[inline]
    fn make_value(&self, typ: &Type, input_port: Port) -> (CircuitUnit, Type) {
        match type_is_linear(typ) {
            true => (CircuitUnit::Linear(self.linear_count - 1), typ.clone()),
            false => (
                CircuitUnit::Wire(Wire::new(self.node, input_port)),
                typ.clone(),
            ),
        }
    }
}

impl Iterator for Units {
    type Item = (CircuitUnit, Type);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let typ = self.node_output_types.get(self.current)?;
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
        let len = self.node_output_types.len() - self.current;
        match self.output_mode {
            UnitType::All => (len, Some(len)),
            _ => (0, Some(len)),
        }
    }
}

impl FusedIterator for Units {}

/// An iterator over the units of a circuit, including their [`Port`]s.
///
/// A simple wrapper around [`Units`].
#[repr(transparent)]
pub struct UnitPorts {
    /// The internal Units iterator.
    units: Units,
}

impl Iterator for UnitPorts {
    type Item = (CircuitUnit, Port, Type);

    fn next(&mut self) -> Option<Self::Item> {
        let port = Port::new_outgoing(self.units.current);
        let (unit, typ) = self.units.next()?;
        Some((unit, port, typ))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.units.size_hint()
    }
}

impl FusedIterator for UnitPorts {}

/// What kind of units to iterate over.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(super) enum UnitType {
    /// All units.
    #[default]
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
