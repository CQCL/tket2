//! Quantum circuit representation and operations.

pub mod command;
mod hash;

pub use hash::CircuitHash;

use self::command::{Command, CommandIterator};

use hugr::extension::prelude::QB_T;
use hugr::hugr::{CircuitUnit, NodeType};
use hugr::ops::OpTrait;
use hugr::HugrView;

pub use hugr::hugr::views::HierarchyView;
pub use hugr::ops::OpType;
use hugr::types::TypeBound;
pub use hugr::types::{EdgeKind, Signature, Type, TypeRow};
pub use hugr::{Node, Port, Wire};
use petgraph::visit::{GraphBase, IntoNeighborsDirected, IntoNodeIdentifiers};

/// An object behaving like a quantum circuit.
//
// TODO: More methods:
// - other_{in,out}puts (for non-linear i/o + const inputs)?
// - Vertical slice iterator
// - Gate count map
// - Depth
pub trait Circuit<'circ>: HugrView {
    /// An iterator over the commands in the circuit.
    type Commands: Iterator<Item = Command>;

    /// An iterator over the commands applied to an unit.
    type UnitCommands: Iterator<Item = Command>;

    /// Return the name of the circuit
    fn name(&self) -> Option<&str>;

    /// Get the linear inputs of the circuit and their types.
    fn units(&self) -> Vec<(CircuitUnit, Type)>;

    /// Returns the units corresponding to qubits inputs to the circuit.
    #[inline]
    fn qubits(&self) -> Vec<CircuitUnit> {
        self.units()
            .iter()
            .filter(|(_, typ)| typ == &QB_T)
            .map(|(unit, _)| *unit)
            .collect()
    }

    /// Returns the input node to the circuit.
    fn input(&self) -> Node;

    /// Returns the output node to the circuit.
    fn output(&self) -> Node;

    /// Given a linear port in a node, returns the corresponding port on the other side of the node (if any).
    fn follow_linear_port(&self, node: Node, port: Port) -> Option<Port>;

    /// Returns all the commands in the circuit, in some topological order.
    ///
    /// Ignores the Input and Output nodes.
    fn commands(&'circ self) -> Self::Commands;

    /// Returns all the commands applied to the given unit, in order.
    fn unit_commands(&'circ self) -> Self::UnitCommands;

    /// Returns the [`NodeType`] of a command.
    fn command_nodetype(&self, command: &Command) -> &NodeType {
        self.get_nodetype(command.node())
    }

    /// Returns the [`OpType`] of a command.
    fn command_optype(&self, command: &Command) -> &OpType {
        self.get_optype(command.node())
    }

    /// The number of gates in the circuit.
    fn num_gates(&self) -> usize;
}

impl<'circ, T> Circuit<'circ> for T
where
    T: 'circ + HierarchyView<'circ>,
    for<'a> &'a T: GraphBase<NodeId = Node> + IntoNeighborsDirected + IntoNodeIdentifiers,
{
    type Commands = CommandIterator<'circ, T>;
    type UnitCommands = std::iter::Empty<Command>;

    #[inline]
    fn name(&self) -> Option<&str> {
        let meta = self.get_metadata(self.root()).as_object()?;
        meta.get("name")?.as_str()
    }

    #[inline]
    fn units(&self) -> Vec<(CircuitUnit, Type)> {
        let root = self.root();
        let optype = self.get_optype(root);
        optype
            .signature()
            .input_types()
            .iter()
            .filter(|&typ| !TypeBound::Copyable.contains(typ.least_upper_bound()))
            .enumerate()
            .map(|(i, typ)| (i.into(), typ.clone()))
            .collect()
    }

    fn follow_linear_port(&self, node: Node, port: Port) -> Option<Port> {
        let optype = self.get_optype(node);
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

    fn commands(&'circ self) -> Self::Commands {
        // Traverse the circuit in topological order.
        CommandIterator::new(self)
    }

    fn unit_commands(&'circ self) -> Self::UnitCommands {
        // TODO Can we associate linear i/o with the corresponding unit without
        // doing the full toposort?
        unimplemented!()
    }

    #[inline]
    fn input(&self) -> Node {
        return self.children(self.root()).next().unwrap();
    }

    #[inline]
    fn output(&self) -> Node {
        return self.children(self.root()).nth(1).unwrap();
    }

    #[inline]
    fn num_gates(&self) -> usize {
        self.children(self.root()).count() - 2
    }
}

#[cfg(test)]
mod tests {
    use std::sync::OnceLock;

    use hugr::{
        hugr::views::{DescendantsGraph, HierarchyView},
        ops::handle::DfgID,
        Hugr, HugrView,
    };

    use crate::{circuit::Circuit, json::load_tk1_json_str};

    static CIRC: OnceLock<Hugr> = OnceLock::new();

    fn test_circuit() -> DescendantsGraph<'static, DfgID> {
        let hugr = CIRC.get_or_init(|| {
            load_tk1_json_str(
                r#"{
                "phase": "0",
                "bits": [],
                "qubits": [["q", [0]], ["q", [1]]],
                "commands": [
                    {"args": [["q", [0]]], "op": {"type": "H"}},
                    {"args": [["q", [0]], ["q", [1]]], "op": {"type": "CX"}}
                ],
                "implicit_permutation": [[["q", [0]], ["q", [0]]], [["q", [1]], ["q", [1]]]]
            }"#,
            )
            .unwrap()
        });
        DescendantsGraph::new(hugr, hugr.root())
    }

    #[test]
    fn test_num_gates() {
        let circ = test_circuit();
        assert_eq!(circ.num_gates(), 2);
    }
}
