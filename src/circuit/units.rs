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
pub struct Units<LA = ()> {
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
    /// A pre-set assignment of units that maps linear output ports to
    /// [`CircuitUnit`] ids.
    ///
    /// The default type is `()`, which assigns new linear ids sequentially.
    unit_assigner: LA,
}

impl Units<()> {
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
            unit_assigner: (),
        }
    }
}

impl<LA> Units<LA>
where
    LA: LinearUnitAssigner,
{
    /// Create a new iterator over the units of a node.
    //
    // Note that this ignores any incoming linear unit labels, and just assigns
    // new unit ids sequentially.
    #[inline]
    #[allow(unused)]
    pub(super) fn new(
        circuit: &impl Circuit,
        node: Node,
        output_mode: UnitType,
        unit_assigner: LA,
    ) -> Self {
        Self {
            output_mode,
            node_output_types: circuit.get_optype(node).signature().output,
            node,
            current: 0,
            linear_count: 0,
            unit_assigner,
        }
    }

    /// Construct an output value to yield.
    #[inline]
    fn make_value(&self, typ: &Type, port: Port) -> (CircuitUnit, Port, Type) {
        match type_is_linear(typ) {
            true => (
                self.unit_assigner.assign(port, self.linear_count - 1),
                port,
                typ.clone(),
            ),
            false => (
                CircuitUnit::Wire(Wire::new(self.node, port)),
                port,
                typ.clone(),
            ),
        }
    }
}

impl Iterator for Units {
    type Item = (CircuitUnit, Port, Type);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let typ = self.node_output_types.get(self.current)?;
            let port = Port::new_outgoing(self.current);
            self.current += 1;
            if type_is_linear(typ) {
                self.linear_count += 1;
            }
            if self.output_mode.accept(typ) {
                return Some(self.make_value(typ, port));
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

/// A map for assigning linear unit ids to ports.
pub trait LinearUnitAssigner {
    /// Assign a linear unit id to an output port.
    fn assign(&self, port: Port, unit: usize) -> CircuitUnit;
}

impl LinearUnitAssigner for () {
    fn assign(&self, _port: Port, unit: usize) -> CircuitUnit {
        CircuitUnit::Linear(unit)
    }
}

impl<'a> LinearUnitAssigner for &'a Vec<usize> {
    fn assign(&self, _port: Port, unit: usize) -> CircuitUnit {
        CircuitUnit::Linear(self[unit])
    }
}

fn type_is_linear(typ: &Type) -> bool {
    !TypeBound::Copyable.contains(typ.least_upper_bound())
}
