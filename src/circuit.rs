//! Quantum circuit representation and operations.

use std::collections::HashMap;
use std::iter::FusedIterator;

use hugr::ops::OpTrait;
pub use hugr::ops::OpType;
pub use hugr::types::{ClassicType, EdgeKind, LinearType, Signature, SimpleType, TypeRow};
use hugr::HugrView;
pub use hugr::{Node, Port, Wire};

//#[cfg(feature = "pyo3")]
//pub mod py_circuit;

//#[cfg(feature = "tkcxx")]
//pub mod unitarybox;

// TODO: Move TKET1's custom op definition to tket-rs (or hugr?)
//mod tk1ops;

/// An object behaving like a quantum circuit.
//
// TODO: More methods:
// - other_{in,out}puts (for non-linear i/o + const inputs)?
// - Vertical slice iterator
// - Gate count map
// - Depth
pub trait Circuit {
    /// An iterator over the commands in the circuit.
    type Commands<'a>: Iterator<Item = Command<'a>>
    where
        Self: 'a;

    /// An iterator over the commands applied to an unit.
    type UnitCommands<'a>: Iterator<Item = Command<'a>>
    where
        Self: 'a;

    /// Return the name of the circuit
    fn name(&self) -> Option<&str>;

    /// Get the linear inputs of the circuit and their types.
    fn units(&self) -> Vec<(Unit, SimpleType)>;

    /// Returns the ports corresponding to qubits inputs to the circuit.
    #[inline]
    fn qubits(&self) -> Vec<Unit> {
        self.units()
            .iter()
            .filter(|(_, typ)| typ == &LinearType::Qubit.into())
            .map(|(unit, _)| *unit)
            .collect()
    }

    /// Given a linear port in a node, returns the corresponding port on the other side of the node (if any).
    fn follow_linear_port(&self, node: Node, port: Port) -> Option<Port>;

    /// Returns all the commands in the circuit, in some topological order.
    fn commands(&self) -> Self::Commands<'_>;

    /// Returns all the commands applied to the given unit, in order.
    fn unit_commands(&self) -> Self::UnitCommands<'_>;
}

// TODO: Define a Region trait in Hugr that implies all these traits.
impl<T> Circuit for T
where
    T: HugrView
        + petgraph::visit::IntoNeighborsDirected
        + petgraph::visit::IntoNodeIdentifiers
        + petgraph::visit::Visitable
        + petgraph::visit::GraphBase<NodeId = Node>,
{
    type Commands<'a> = CommandIterator<'a, T> where Self: 'a;
    type UnitCommands<'a> = std::iter::Empty<Command<'a>> where Self: 'a;

    #[inline]
    fn name(&self) -> Option<&str> {
        let meta = self.get_metadata(self.root()).as_object()?;
        meta.get("name")?.as_str()
    }

    #[inline]
    fn units(&self) -> Vec<(Unit, SimpleType)> {
        let root = self.root();
        let optype = self.get_optype(root);
        optype
            .signature()
            .input_df_types()
            .iter()
            .filter(|typ| typ.is_linear())
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

    fn commands(&self) -> Self::Commands<'_> {
        // Traverse the circuit in topological order.
        CommandIterator::new(self)
    }

    fn unit_commands(&self) -> Self::UnitCommands<'_> {
        // TODO Can we associate linear i/o with the corresponding unit without
        // doing the full toposort?
        todo!()
    }
}

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
