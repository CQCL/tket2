//! Quantum circuit representation and operations.

pub mod command;

//#[cfg(feature = "pyo3")]
//pub mod py_circuit;

//#[cfg(feature = "tkcxx")]
//pub mod unitarybox;

// TODO: Move TKET1's custom op definition to tket-rs (or hugr?)
//mod tk1ops;

use crate::utils::QB;

use self::command::{Command, CommandIterator};

use hugr::hugr::CircuitUnit;
use hugr::ops::OpTrait;

pub use hugr::hugr::region::Region;
pub use hugr::ops::OpType;
pub use hugr::types::{ClassicType, EdgeKind, Signature, SimpleType, TypeRow};
pub use hugr::{Node, Port, Wire};
use petgraph::visit::{GraphBase, IntoNeighborsDirected, IntoNodeIdentifiers};

/// An object behaving like a quantum circuit.
//
// TODO: More methods:
// - other_{in,out}puts (for non-linear i/o + const inputs)?
// - Vertical slice iterator
// - Gate count map
// - Depth
pub trait Circuit<'circ> {
    /// An iterator over the commands in the circuit.
    type Commands: Iterator<Item = Command<'circ>>;

    /// An iterator over the commands applied to an unit.
    type UnitCommands: Iterator<Item = Command<'circ>>;

    /// Return the name of the circuit
    fn name(&self) -> Option<&str>;

    /// Get the linear inputs of the circuit and their types.
    fn units(&self) -> Vec<(CircuitUnit, SimpleType)>;

    /// Returns the ports corresponding to qubits inputs to the circuit.
    #[inline]
    fn qubits(&self) -> Vec<CircuitUnit> {
        self.units()
            .iter()
            .filter(|(_, typ)| typ == &QB)
            .map(|(unit, _)| *unit)
            .collect()
    }

    /// Given a linear port in a node, returns the corresponding port on the other side of the node (if any).
    fn follow_linear_port(&self, node: Node, port: Port) -> Option<Port>;

    /// Returns all the commands in the circuit, in some topological order.
    fn commands<'a: 'circ>(&'a self) -> Self::Commands;

    /// Returns all the commands applied to the given unit, in order.
    fn unit_commands<'a: 'circ>(&'a self) -> Self::UnitCommands;
}

impl<'circ, T> Circuit<'circ> for T
where
    T: 'circ + Region<'circ>,
    for<'a> &'a T: GraphBase<NodeId = Node> + IntoNeighborsDirected + IntoNodeIdentifiers,
{
    type Commands = CommandIterator<'circ, T>;
    type UnitCommands = std::iter::Empty<Command<'circ>>;

    #[inline]
    fn name(&self) -> Option<&str> {
        let meta = self.get_metadata(self.root()).as_object()?;
        meta.get("name")?.as_str()
    }

    #[inline]
    fn units(&self) -> Vec<(CircuitUnit, SimpleType)> {
        let root = self.root();
        let optype = self.get_optype(root);
        optype
            .signature()
            .input_df_types()
            .iter()
            .filter(|typ| !typ.is_classical())
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
        debug_assert_eq!(optype.port_kind(other_port), optype.port_kind(port));
        Some(other_port)
    }

    fn commands<'a: 'circ>(&'a self) -> Self::Commands {
        // Traverse the circuit in topological order.
        CommandIterator::new(self)
    }

    fn unit_commands<'a: 'circ>(&'a self) -> Self::UnitCommands {
        // TODO Can we associate linear i/o with the corresponding unit without
        // doing the full toposort?
        todo!()
    }
}
