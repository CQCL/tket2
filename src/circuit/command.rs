//! Circuit commands.
//!
//! A [`Command`] is an operation applied to an specific wires, possibly identified by their index in the circuit's input vector.

use std::collections::{HashMap, HashSet};
use std::iter::FusedIterator;

use hugr::hugr::NodeType;
use hugr::ops::{OpTag, OpTrait};
use petgraph::visit as pv;

use super::units::{LinearUnit, LinearUnitAssigner, UnitType, Units};
use super::Circuit;

pub use hugr::hugr::CircuitUnit;
pub use hugr::ops::OpType;
pub use hugr::types::{EdgeKind, Signature, Type, TypeRow};
pub use hugr::{Direction, Node, Port, Wire};

/// An operation applied to specific wires.
#[derive(Eq, PartialOrd, Ord, Hash)]
pub struct Command<'circ, Circ> {
    /// The circuit.
    circ: &'circ Circ,
    /// The operation node.
    node: Node,
    /// An assignment of linear units to the node's ports.
    linear_units: Vec<Option<LinearUnit>>,
}

impl<'circ, Circ: Circuit> Command<'circ, Circ> {
    /// Returns the node corresponding to this command.
    pub fn node(&self) -> Node {
        self.node
    }

    /// Returns the [`NodeType`] of the command.
    pub fn nodetype(&self) -> &NodeType {
        self.circ.get_nodetype(self.node)
    }

    /// Returns the [`OpType`] of the command.
    pub fn optype(&self) -> &OpType {
        self.circ.get_optype(self.node)
    }

    /// Returns the output units of this command.
    pub fn outputs(&self) -> Units<&'_ Self> {
        Units::new(
            self.circ,
            self.node,
            Direction::Outgoing,
            UnitType::All,
            self,
        )
    }

    /// Returns the output wires of this command.
    pub fn output_wires(&self) -> impl FusedIterator<Item = (CircuitUnit, Wire)> + '_ {
        self.outputs()
            .map(|(unit, port, _)| (unit, Wire::new(self.node, port)))
    }

    /// Returns the output units of this command.
    pub fn inputs(&self) -> Units<&'_ Self> {
        Units::new(
            self.circ,
            self.node,
            Direction::Incoming,
            UnitType::All,
            self,
        )
    }

    /// Returns the number of inputs of this command.
    pub fn input_count(&self) -> usize {
        self.optype().signature().input_count()
    }

    /// Returns the number of outputs of this command.
    pub fn output_count(&self) -> usize {
        self.optype().signature().output_count()
    }
}

impl<'a, 'circ, Circ> LinearUnitAssigner for &'a Command<'circ, Circ> {
    fn assign(&self, port: Port, _linear_count: usize) -> LinearUnit {
        self.linear_units
            .get(port.index())
            .copied()
            .flatten()
            .unwrap_or_else(|| {
                panic!(
                    "Could not assign a linear unit to port {port:?} of node {:?}",
                    self.node
                )
            })
    }
}

impl<'circ, Circ: Circuit> std::fmt::Debug for Command<'circ, Circ> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Command")
            .field("circ", &self.circ.name())
            .field("node", &self.node)
            .field("linear_units", &self.linear_units)
            .finish()
    }
}

impl<'circ, Circ> PartialEq for Command<'circ, Circ> {
    fn eq(&self, other: &Self) -> bool {
        self.node == other.node && self.linear_units == other.linear_units
    }
}

impl<'circ, Circ> Clone for Command<'circ, Circ> {
    fn clone(&self) -> Self {
        Self {
            circ: self.circ,
            node: self.node,
            linear_units: self.linear_units.clone(),
        }
    }
}

/// A non-borrowing topological walker over the nodes of a circuit.
type NodeWalker = pv::Topo<Node, HashSet<Node>>;

/// An iterator over the commands of a circuit.
#[derive(Clone)]
pub struct CommandIterator<'circ, Circ> {
    /// The circuit.
    circ: &'circ Circ,
    /// Toposorted nodes.
    nodes: NodeWalker,
    /// Last wires for each linear `CircuitUnit`.
    wire_unit: HashMap<Wire, usize>,
    /// Remaining commands, not counting I/O nodes.
    remaining: usize,
}

impl<'circ, Circ: std::fmt::Debug> std::fmt::Debug for CommandIterator<'circ, Circ> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CommandIterator")
            .field("circ", &self.circ)
            .field("wire_unit", &self.wire_unit)
            .field("remaining", &self.remaining)
            .finish()
    }
}

impl<'circ, Circ> CommandIterator<'circ, Circ>
where
    Circ: Circuit,
{
    /// Create a new iterator over the commands of a circuit.
    pub(super) fn new(circ: &'circ Circ) -> Self {
        // Initialize the map assigning linear units to the input's linear
        // ports.
        let wire_unit = circ
            .units()
            .filter_map(|(unit, port, _)| match unit {
                CircuitUnit::Linear(i) => Some((Wire::new(circ.input(), port), i)),
                _ => None,
            })
            .collect();

        let nodes = pv::Topo::new(&circ.as_petgraph());
        Self {
            circ,
            nodes,
            wire_unit,
            // Ignore the input and output nodes, and the root.
            remaining: circ.node_count() - 3,
        }
    }

    /// Process a new node, updating wires in `unit_wires`.
    ///
    /// Returns the an option with the `linear_units` used to construct a
    /// [`Command`], if the node is not an input or output.
    ///
    /// We don't return the command directly to avoid lifetime issues due to the
    /// mutable borrow here.
    fn process_node(&mut self, node: Node) -> Option<Vec<Option<LinearUnit>>> {
        let optype = self.circ.get_optype(node);
        let sig = optype.signature();

        // The root node is ignored.
        if node == self.circ.root() {
            return None;
        }

        // Get the wire corresponding to each input unit.
        // TODO: Add this to HugrView?
        let _inputs: Vec<_> = sig
            .input_ports()
            .chain(
                // add the static input port
                optype
                    .static_input()
                    // TODO query optype for this port once it is available in hugr.
                    .map(|_| Port::new_incoming(sig.input.len())),
            )
            .filter_map(|port| {
                let (from, from_port) = self.circ.linked_ports(node, port).next()?;
                let wire = Wire::new(from, from_port);
                // Get the unit corresponding to a wire, or return a wire Unit.
                match self.wire_unit.remove(&wire) {
                    Some(unit) => {
                        if let Some(new_port) = self.follow_linear_port(node, port) {
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

        let mut outputs: Vec<_> = sig
            .output_ports()
            .map(|port| {
                let wire = Wire::new(node, port);
                match self.wire_unit.get(&wire) {
                    Some(&unit) => CircuitUnit::Linear(unit),
                    None => CircuitUnit::Wire(wire),
                }
            })
            .collect();
        if let OpType::Const(_) = optype {
            // add the static output port from a const.
            outputs.push(CircuitUnit::Wire(Wire::new(
                node,
                optype.other_port_index(Direction::Outgoing).unwrap(),
            )))
        }

        todo!()
    }

    /// Returns the linear port on the node that corresponds to the same linear unit.
    ///
    /// We assume the linear data uses the same port offsets on both sides of the node.
    /// In the future we may want to have a more general mechanism to handle this.
    fn follow_linear_port(&self, node: Node, port: Port) -> Option<Port> {
        let optype = self.circ.get_optype(node);
        if !optype.port_kind(port)?.is_linear() {
            return None;
        }
        let other_port = Port::new(port.direction().reverse(), port.index());
        if optype.port_kind(other_port) == optype.port_kind(port) {
            Some(other_port)
        } else {
            None
        }
    }
}

impl<'circ, Circ> Iterator for CommandIterator<'circ, Circ>
where
    Circ: Circuit,
{
    type Item = Command<'circ, Circ>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let node = self.nodes.next(&self.circ.as_petgraph())?;
            if let Some(linear_units) = self.process_node(node) {
                self.remaining -= 1;
                return Some(Command {
                    circ: self.circ,
                    node,
                    linear_units,
                });
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<'circ, Circ> FusedIterator for CommandIterator<'circ, Circ> where Circ: Circuit {}

#[cfg(test)]
mod test {
    use hugr::hugr::views::{HierarchyView, SiblingGraph};
    use hugr::ops::OpName;
    use hugr::HugrView;
    use itertools::Itertools;
    use std::fmt::Debug;

    use crate::utils::build_simple_circuit;
    use crate::T2Op;

    use super::*;

    fn assert_eq_iter<T>(x: impl Iterator<Item = T>, expected: impl IntoIterator<Item = T>)
    where
        T: PartialEq + Debug,
    {
        assert_eq!(x.collect_vec(), expected.into_iter().collect_vec());
    }

    #[test]
    fn iterate_commands() {
        let hugr = build_simple_circuit(2, |circ| {
            circ.append(T2Op::H, [0])?;
            circ.append(T2Op::CX, [0, 1])?;
            circ.append(T2Op::T, [1])?;
            Ok(())
        })
        .unwrap();
        let circ: SiblingGraph<'_> = SiblingGraph::new(&hugr, hugr.root());

        assert_eq!(CommandIterator::new(&circ).count(), 3);

        // TODO: Expose the operation names directly in T2Op to clean this up
        let t2op_name = |op: T2Op| <T2Op as Into<OpType>>::into(op).name();

        let mut commands = CommandIterator::new(&circ);

        let hadamard = commands.next().unwrap();
        assert_eq!(hadamard.optype().name().as_str(), t2op_name(T2Op::H));
        assert_eq_iter(
            hadamard.inputs().map(|(u, _, _)| u),
            [CircuitUnit::Linear(0)],
        );
        assert_eq_iter(
            hadamard.outputs().map(|(u, _, _)| u),
            [CircuitUnit::Linear(0)],
        );

        let cx = commands.next().unwrap();
        assert_eq!(cx.optype().name().as_str(), t2op_name(T2Op::CX));
        assert_eq_iter(
            cx.inputs().map(|(unit, _, _)| unit),
            [CircuitUnit::Linear(0), CircuitUnit::Linear(1)],
        );
        assert_eq_iter(
            cx.outputs().map(|(unit, _, _)| unit),
            [CircuitUnit::Linear(0), CircuitUnit::Linear(1)],
        );

        let t = commands.next().unwrap();
        assert_eq!(t.optype().name().as_str(), t2op_name(T2Op::T));
        assert_eq_iter(
            t.inputs().map(|(unit, _, _)| unit),
            [CircuitUnit::Linear(1)],
        );
        assert_eq_iter(
            t.outputs().map(|(unit, _, _)| unit),
            [CircuitUnit::Linear(1)],
        );

        assert_eq!(commands.next(), None);
    }
}
