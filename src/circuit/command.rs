//! Circuit commands.
//!
//! A [`Command`] is an operation applied to an specific wires, possibly identified by their index in the circuit's input vector.

use std::collections::HashMap;
use std::iter::FusedIterator;

use hugr::hugr::views::HierarchyView;
use hugr::ops::{OpTag, OpTrait};
use petgraph::visit::{GraphBase, IntoNeighborsDirected, IntoNodeIdentifiers};

use super::Circuit;

pub use hugr::hugr::CircuitUnit;
pub use hugr::ops::OpType;
pub use hugr::types::{Type, EdgeKind, Signature, Type, TypeRow};
pub use hugr::{Node, Port, Wire};

/// An operation applied to specific wires.
pub struct Command<'circ> {
    /// The operation.
    pub op: &'circ OpType,
    /// The operation node.
    pub node: Node,
    /// The input units to the operation.
    pub inputs: Vec<CircuitUnit>,
    /// The output units to the operation.
    pub outputs: Vec<CircuitUnit>,
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
    /// Last wires for each linear `CircuitUnit`
    wire_unit: HashMap<Wire, usize>,
}

impl<'circ, Circ> CommandIterator<'circ, Circ>
where
    Circ: HierarchyView<'circ>,
    for<'a> &'a Circ: GraphBase<NodeId = Node> + IntoNeighborsDirected + IntoNodeIdentifiers,
{
    /// Create a new iterator over the commands of a circuit.
    pub(super) fn new(circ: &'circ Circ) -> Self {
        // Initialize the linear units from the input's linear ports.
        // TODO: Clean this up
        let input_node_wires = circ
            .node_outputs(circ.input())
            .map(|port| Wire::new(circ.input(), port));
        let wire_unit = input_node_wires
            .zip(circ.units().iter())
            .filter_map(|(wire, (unit, _))| match unit {
                CircuitUnit::Linear(i) => Some((wire, *i)),
                _ => None,
            })
            .collect();

        let nodes = petgraph::algo::toposort(circ, None).unwrap();
        Self {
            circ,
            nodes,
            current: 0,
            wire_unit,
        }
    }

    /// Process a new node, updating wires in `unit_wires` and returns the
    /// command for the node if it's not an input or output.
    fn process_node(&mut self, node: Node) -> Option<Command<'circ>> {
        let optype = self.circ.get_optype(node);
        let sig = optype.signature();

        // The root node is ignored.
        if node == self.circ.root() {
            return None;
        }

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
                        Some(CircuitUnit::Linear(unit))
                    }
                    None => Some(CircuitUnit::Wire(wire)),
                }
            })
            .collect();

        // The units in `self.wire_units` have been updated.
        // Now we can early return if the node should be ignored.
        let tag = optype.tag();
        if tag == OpTag::Input || tag == OpTag::Output {
            return None;
        }

        let outputs = sig
            .output_ports()
            .map(|port| {
                let wire = Wire::new(node, port);
                match self.wire_unit.get(&wire) {
                    Some(&unit) => CircuitUnit::Linear(unit),
                    None => CircuitUnit::Wire(wire),
                }
            })
            .collect();

        Some(Command {
            op: optype,
            node,
            inputs,
            outputs,
        })
    }
}

impl<'circ, Circ> Iterator for CommandIterator<'circ, Circ>
where
    Circ: HierarchyView<'circ>,
    for<'a> &'a Circ: GraphBase<NodeId = Node> + IntoNeighborsDirected + IntoNodeIdentifiers,
{
    type Item = Command<'circ>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.current == self.nodes.len() {
                return None;
            }
            let node = self.nodes[self.current];
            let com = self.process_node(node);
            self.current += 1;
            if com.is_some() {
                return com;
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.nodes.len() - self.current))
    }
}

impl<'circ, Circ> FusedIterator for CommandIterator<'circ, Circ>
where
    Circ: HierarchyView<'circ>,
    for<'a> &'a Circ: GraphBase<NodeId = Node> + IntoNeighborsDirected + IntoNodeIdentifiers,
{
}
