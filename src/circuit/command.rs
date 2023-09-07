//! Circuit commands.
//!
//! A [`Command`] is an operation applied to an specific wires, possibly identified by their index in the circuit's input vector.

use std::collections::HashMap;
use std::iter::FusedIterator;

use hugr::ops::{OpTag, OpTrait};

use super::Circuit;

pub use hugr::hugr::CircuitUnit;
pub use hugr::ops::OpType;
pub use hugr::types::{EdgeKind, Signature, Type, TypeRow};
pub use hugr::{Direction, Node, Port, Wire};

/// An operation applied to specific wires.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Command {
    /// The operation node.
    node: Node,
    /// The input units to the operation.
    inputs: Vec<CircuitUnit>,
    /// The output units to the operation.
    outputs: Vec<CircuitUnit>,
}

impl Command {
    /// Returns the node corresponding to this command.
    pub fn node(&self) -> Node {
        self.node
    }

    /// Returns the output units of this command.
    pub fn outputs(&self) -> &Vec<CircuitUnit> {
        &self.outputs
    }

    /// Returns the output units of this command.
    pub fn inputs(&self) -> &Vec<CircuitUnit> {
        &self.inputs
    }
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
    Circ: Circuit,
{
    /// Create a new iterator over the commands of a circuit.
    pub(super) fn new(circ: &'circ Circ) -> Self {
        // Initialize the linear units from the input's linear ports.
        // TODO: Clean this up
        let input_node_wires = circ
            .node_outputs(circ.input())
            .map(|port| Wire::new(circ.input(), port));
        let wire_unit = input_node_wires
            .zip(circ.linear_units())
            .filter_map(|(wire, (unit, _))| match unit {
                CircuitUnit::Linear(i) => Some((wire, i)),
                _ => None,
            })
            .collect();

        let nodes = petgraph::algo::toposort(&circ.as_petgraph(), None).unwrap();
        Self {
            circ,
            nodes,
            current: 0,
            wire_unit,
        }
    }

    /// Process a new node, updating wires in `unit_wires` and returns the
    /// command for the node if it's not an input or output.
    fn process_node(&mut self, node: Node) -> Option<Command> {
        let optype = self.circ.get_optype(node);
        let sig = optype.signature();

        // The root node is ignored.
        if node == self.circ.root() {
            return None;
        }

        // Get the wire corresponding to each input unit.
        // TODO: Add this to HugrView?
        let inputs: Vec<_> = sig
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
        Some(Command {
            node,
            inputs,
            outputs,
        })
    }

    fn follow_linear_port(&self, node: Node, port: Port) -> Option<Port> {
        let optype = self.circ.get_optype(node);
        if !optype.port_kind(port)?.is_linear() {
            return None;
        }
        // TODO: We assume the linear data uses the same port offsets on both sides of the node.
        // In the future we may want to have a more general mechanism to handle this.
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
    type Item = Command;

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

impl<'circ, Circ> FusedIterator for CommandIterator<'circ, Circ> where Circ: Circuit {}

#[cfg(test)]
mod test {
    use hugr::hugr::views::{HierarchyView, SiblingGraph};
    use hugr::ops::OpName;
    use hugr::HugrView;

    use crate::utils::build_simple_circuit;
    use crate::T2Op;

    use super::*;

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
        assert_eq!(
            circ.command_optype(&hadamard).name().as_str(),
            t2op_name(T2Op::H)
        );
        assert_eq!(hadamard.inputs(), &[CircuitUnit::Linear(0)]);
        assert_eq!(hadamard.outputs(), &[CircuitUnit::Linear(0)]);

        let cx = commands.next().unwrap();
        assert_eq!(
            circ.command_optype(&cx).name().as_str(),
            t2op_name(T2Op::CX)
        );
        assert_eq!(
            cx.inputs(),
            &[CircuitUnit::Linear(0), CircuitUnit::Linear(1)]
        );
        assert_eq!(
            cx.outputs(),
            &[CircuitUnit::Linear(0), CircuitUnit::Linear(1)]
        );

        let t = commands.next().unwrap();
        assert_eq!(circ.command_optype(&t).name().as_str(), t2op_name(T2Op::T));
        assert_eq!(t.inputs(), &[CircuitUnit::Linear(1)]);
        assert_eq!(t.outputs(), &[CircuitUnit::Linear(1)]);

        assert_eq!(commands.next(), None);
    }
}
