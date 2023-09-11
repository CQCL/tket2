//! Iterators over the units of a circuit.

use std::iter::FusedIterator;

use hugr::extension::prelude;
use hugr::hugr::CircuitUnit;
use hugr::ops::OpTrait;
use hugr::types::{Type, TypeBound, TypeRow};
use hugr::{Direction, Node, Port, Wire};

use super::Circuit;

/// A linear unit id, used in [`CircuitUnit::Linear`].
// TODO: Add this to hugr?
pub type LinearUnit = usize;

/// An iterator over the units in the input or output boundary of a [Node].
#[derive(Clone, Debug)]
pub struct Units<LA = ()> {
    /// Filter over the yielded units.
    ///
    /// It can be set to ignore non-linear units, only yield qubits, between
    /// other options. See [`UnitType`] for more information.
    mode: UnitType,
    /// The node of the circuit.
    node: Node,
    /// The direction of the boundary.
    direction: Direction,
    /// The types of the boundary.
    //
    // TODO: We could avoid cloning the TypeRow if `OpType::signature` returned
    // a reference.
    type_row: TypeRow,
    /// The current index in the type row.
    current: usize,
    /// The amount of linear units yielded.
    linear_count: usize,
    /// A pre-set assignment of units that maps linear ports to
    /// [`CircuitUnit::Linear`] ids.
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
    pub(super) fn new_circ_input(circuit: &impl Circuit, output_mode: UnitType) -> Self {
        Self {
            mode: output_mode,
            node: circuit.input(),
            direction: Direction::Outgoing,
            type_row: circuit
                .get_function_type()
                .map_or_else(Default::default, |ft| ft.input.clone()),
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
    pub(super) fn new(
        circuit: &impl Circuit,
        node: Node,
        direction: Direction,
        output_mode: UnitType,
        unit_assigner: LA,
    ) -> Self {
        let sig = circuit.get_optype(node).signature();
        let type_row = match direction {
            Direction::Outgoing => sig.output,
            Direction::Incoming => sig.input,
        };
        Self {
            mode: output_mode,
            node,
            direction,
            type_row,
            current: 0,
            linear_count: 0,
            unit_assigner,
        }
    }

    /// Construct an output value to yield.
    ///
    /// Calls [`LinearUnitAssigner::assign`] to assign a linear unit id to the linear ports.
    /// Non-linear ports are assigned [`CircuitUnit::Wire`]s.
    #[inline]
    fn make_value(&self, typ: &Type, port: Port) -> (CircuitUnit, Port, Type) {
        let unit = if type_is_linear(typ) {
            let linear_unit = self.unit_assigner.assign(port, self.linear_count - 1);
            CircuitUnit::Linear(linear_unit)
        } else {
            match self.direction {
                Direction::Outgoing => CircuitUnit::Wire(Wire::new(self.node, port)),
                Direction::Incoming => CircuitUnit::Wire(Wire::new(self.node, port)),
            }
        };
        (unit, port, typ.clone())
    }
}

impl<LA> Iterator for Units<LA>
where
    LA: LinearUnitAssigner,
{
    type Item = (CircuitUnit, Port, Type);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let typ = self.type_row.get(self.current)?;
            let port = Port::new(self.direction, self.current);
            self.current += 1;
            if type_is_linear(typ) {
                self.linear_count += 1;
            }
            if self.mode.accept(typ) {
                return Some(self.make_value(typ, port));
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.type_row.len() - self.current;
        match self.mode {
            UnitType::All => (len, Some(len)),
            _ => (0, Some(len)),
        }
    }
}

impl<LA> FusedIterator for Units<LA> where LA: LinearUnitAssigner {}

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
    ///
    /// # Parameters
    /// - port: The node's port in the node.
    /// - linear_count: The number of linear units yielded so far.
    fn assign(&self, port: Port, linear_count: usize) -> LinearUnit;
}

impl LinearUnitAssigner for () {
    fn assign(&self, _port: Port, linear_count: usize) -> LinearUnit {
        linear_count
    }
}

fn type_is_linear(typ: &Type) -> bool {
    !TypeBound::Copyable.contains(typ.least_upper_bound())
}
