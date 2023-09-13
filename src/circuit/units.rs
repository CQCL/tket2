//! Iterators over the units of a circuit.

use std::iter::FusedIterator;

use hugr::extension::prelude;
use hugr::hugr::CircuitUnit;
use hugr::ops::OpTrait;
use hugr::types::{EdgeKind, Type, TypeBound, TypeRow};
use hugr::{Direction, Node, Port, Wire};

use super::Circuit;

/// A linear unit id, used in [`CircuitUnit::Linear`].
// TODO: Add this to hugr?
pub type LinearUnit = usize;

/// An iterator over the units in the input or output boundary of a [Node].
#[derive(Clone, Debug)]
pub struct Units<UL = ()> {
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
    types: TypeRow,
    /// The current index in the type row.
    current: usize,
    /// The amount of linear units yielded.
    linear_count: usize,
    /// A pre-set assignment of units that maps linear ports to
    /// [`CircuitUnit::Linear`] ids.
    ///
    /// The default type is `()`, which assigns new linear ids sequentially.
    unit_assigner: UL,
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
            types: circuit
                .get_function_type()
                .map_or_else(Default::default, |ft| ft.input.clone()),
            current: 0,
            linear_count: 0,
            unit_assigner: (),
        }
    }
}

impl<UL> Units<UL>
where
    UL: UnitLabeller,
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
        direction: Direction,
        output_mode: UnitType,
        unit_assigner: UL,
    ) -> Self {
        Self {
            mode: output_mode,
            node,
            direction,
            types: Self::init_types(circuit, node, direction),
            current: 0,
            linear_count: 0,
            unit_assigner,
        }
    }

    /// Initialize the boundary types.
    ///
    /// We use a [`TypeRow`] to avoid allocating for simple boundaries, but if
    /// any static port is present we create a new owned [`TypeRow`] with them included.
    //
    // TODO: This is quite hacky, but we need it to accept Const static inputs.
    // We should revisit it once this is reworked on the HUGR side.
    fn init_types(circuit: &impl Circuit, node: Node, direction: Direction) -> TypeRow {
        let optype = circuit.get_optype(node);
        let sig = optype.signature();
        let mut types = match direction {
            Direction::Outgoing => sig.output,
            Direction::Incoming => sig.input,
        };
        if let Some(other) = optype.static_input() {
            if direction == Direction::Incoming {
                types.to_mut().push(other);
            }
        }
        if let Some(EdgeKind::Static(other)) = optype.other_port(direction) {
            types.to_mut().push(other);
        }
        types
    }

    /// Construct an output value to yield.
    ///
    /// Calls [`UnitLabeller::assign_linear`] to assign a linear unit id to the linear ports.
    /// Non-linear ports are assigned [`CircuitUnit::Wire`]s via [`UnitLabeller::assign_wire`].
    #[inline]
    fn make_value(&self, typ: &Type, port: Port) -> Option<(CircuitUnit, Port, Type)> {
        let unit = if type_is_linear(typ) {
            let linear_unit =
                self.unit_assigner
                    .assign_linear(self.node, port, self.linear_count - 1);
            CircuitUnit::Linear(linear_unit)
        } else {
            let wire = self.unit_assigner.assign_wire(self.node, port)?;
            CircuitUnit::Wire(wire)
        };
        Some((unit, port, typ.clone()))
    }
}

impl<UL> Iterator for Units<UL>
where
    UL: UnitLabeller,
{
    type Item = (CircuitUnit, Port, Type);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let typ = self.types.get(self.current)?;
            let port = Port::new(self.direction, self.current);
            self.current += 1;
            if type_is_linear(typ) {
                self.linear_count += 1;
            }
            if self.mode.accept(typ) {
                let val = self.make_value(typ, port);
                if val.is_some() {
                    return val;
                }
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.types.len() - self.current;
        if self.mode == UnitType::All && self.direction == Direction::Outgoing {
            (len, Some(len))
        } else {
            // Even when yielding every unit, a disconnected input non-linear
            // port cannot be assigned a `CircuitUnit::Wire` and so it will be
            // skipped.
            (0, Some(len))
        }
    }
}

impl<UL> FusedIterator for Units<UL> where UL: UnitLabeller {}

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

/// An trait for assigning linear unit ids and wires to ports of a node.
pub trait UnitLabeller {
    /// Assign a linear unit id to an port.
    ///
    /// # Parameters
    /// - node: The node in the circuit.
    /// - port: The node's port in the node.
    /// - linear_count: The number of linear units yielded so far.
    fn assign_linear(&self, node: Node, port: Port, linear_count: usize) -> LinearUnit;

    /// Assign a wire to a port, if possible.
    ///
    /// # Parameters
    /// - node: The node in the circuit.
    /// - port: The node's port in the node.
    fn assign_wire(&self, node: Node, port: Port) -> Option<Wire>;
}

/// The default [`UnitLabeller`] that assigns new linear unit ids
/// sequentially, and only assigns wires to an outgoing ports (as input ports
/// require querying the HUGR for their neighbours).
impl UnitLabeller for () {
    #[inline]
    fn assign_linear(&self, _: Node, _: Port, linear_count: usize) -> LinearUnit {
        linear_count
    }

    #[inline]
    fn assign_wire(&self, node: Node, port: Port) -> Option<Wire> {
        match port.direction() {
            Direction::Incoming => None,
            Direction::Outgoing => Some(Wire::new(node, port)),
        }
    }
}

fn type_is_linear(typ: &Type) -> bool {
    !TypeBound::Copyable.contains(typ.least_upper_bound())
}
