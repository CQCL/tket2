//! Circuit commands.
//!
//! A [`Command`] is an operation applied to an specific wires, possibly identified by their index in the circuit's input vector.

use std::collections::HashMap;
use std::iter::FusedIterator;

use hugr::ops::OpTrait;
pub use hugr::ops::OpType;
pub use hugr::types::{ClassicType, EdgeKind, LinearType, Signature, SimpleType, TypeRow};
use hugr::HugrView;
pub use hugr::{Node, Port, Wire};

use super::Circuit;

/// Descriptor of a wire in a [`Circuit`]. If it is a qubit or linear bit
/// originating from the circuit's input, it is described by an index.
/// Otherwise, it is described by an internal [`Wire`].
//
// TODO Better name?
// TODO Merge this with CircuitBuilder::AppendWire?
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Unit {
    /// Arbitrary wire.
    W(Wire),
    /// Index of a linear element in the [`Circuit`]'s input vector.
    Linear(usize),
}

impl From<usize> for Unit {
    fn from(value: usize) -> Self {
        Unit::Linear(value)
    }
}

impl From<Wire> for Unit {
    fn from(value: Wire) -> Self {
        Unit::W(value)
    }
}

/// An operation applied to specific wires.
pub struct Command<'circ> {
    /// The operation.
    pub op: &'circ OpType,
    /// The operation node.
    pub node: Node,
    /// The input units to the operation.
    pub inputs: Vec<Unit>,
    /// The output units to the operation.
    pub outputs: Vec<Unit>,
}

/// An iterator over the commands of a circuit.
#[derive(Clone, Debug)]
pub struct CommandIterator<'circ, Circ> {
    /// The circuit
    circ: &'circ Circ,
    /// Toposorted nodes
    nodes: Vec<Node>,
    /// Current element in `nodes`
    current: usize,
    /// Last wires for each linear `Unit`
    wire_unit: HashMap<Wire, usize>,
}

impl<'circ, Circ> CommandIterator<'circ, Circ>
where
    Circ: HugrView
        + petgraph::visit::IntoNeighborsDirected
        + petgraph::visit::IntoNodeIdentifiers
        + petgraph::visit::Visitable
        + petgraph::visit::GraphBase<NodeId = Node>,
{
    /// Create a new iterator over the commands of a circuit.
    pub(super) fn new(circ: &'circ Circ) -> Self {
        let nodes = petgraph::algo::toposort(circ, None).unwrap();
        Self {
            circ,
            nodes,
            current: 0,
            wire_unit: HashMap::new(),
        }
    }

    /// Process a new node, updating wires in `unit_wires` and returns the
    /// command for the node.
    fn process_node(&mut self, node: Node) -> Command<'circ> {
        let optype = self.circ.get_optype(node);
        let sig = optype.signature();

        // Get the wire corresponding to each input unit.
        // TODO: Add this to HugrView?
        let inputs = sig
            .input_ports()
            .filter_map(|port| {
                let (from, from_port) = self.circ.linked_ports(node, port).next()?;
                let wire = Wire::new(from, from_port);
                // Get the unit corresponding to a wire, or return a wire Unit.
                match self.wire_unit.remove(&wire) {
                    Some(unit) => {
                        if let Some(new_port) = self.circ.follow_linear_port(node, port) {
                            self.wire_unit.insert(Wire::new(node, new_port), unit);
                        }
                        Some(Unit::Linear(unit))
                    }
                    None => Some(Unit::W(wire)),
                }
            })
            .collect();

        let outputs = sig
            .output_ports()
            .map(|port| {
                let wire = Wire::new(node, port);
                match self.wire_unit.get(&wire) {
                    Some(&unit) => Unit::Linear(unit),
                    None => Unit::W(wire),
                }
            })
            .collect();

        Command {
            op: optype,
            node,
            inputs,
            outputs,
        }
    }
}

impl<'circ, Circ> Iterator for CommandIterator<'circ, Circ>
where
    Circ: HugrView
        + petgraph::visit::IntoNeighborsDirected
        + petgraph::visit::IntoNodeIdentifiers
        + petgraph::visit::Visitable
        + petgraph::visit::GraphBase<NodeId = Node>,
{
    type Item = Command<'circ>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current == self.nodes.len() {
            return None;
        }
        let node = self.nodes[self.current];
        self.current += 1;
        Some(self.process_node(node))
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), Some(self.len()))
    }
}

impl<Circ> ExactSizeIterator for CommandIterator<'_, Circ>
where
    Circ: HugrView
        + petgraph::visit::IntoNeighborsDirected
        + petgraph::visit::IntoNodeIdentifiers
        + petgraph::visit::Visitable
        + petgraph::visit::GraphBase<NodeId = Node>,
{
    #[inline]
    fn len(&self) -> usize {
        self.nodes.len() - self.current
    }
}

impl<Circ> FusedIterator for CommandIterator<'_, Circ> where
    Circ: HugrView
        + petgraph::visit::IntoNeighborsDirected
        + petgraph::visit::IntoNodeIdentifiers
        + petgraph::visit::Visitable
        + petgraph::visit::GraphBase<NodeId = Node>
{
}
