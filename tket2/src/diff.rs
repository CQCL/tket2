//! Persistent data structures for circuit rewriting.
//!
//! This module contains the definition of the [`CircuitDiff`] type, which
//! represent local circuit transformations. This enables persistent rewriting,
//! in which the original data is never changed, but new diffs are created to
//! represent the changes.
//!
//! Diffs can be created on top of existing diffs, resulting in an acyclic
//! history of circuit transformations.

pub mod experimental;
mod history;
mod replacement;

pub use history::CircuitHistory;

use std::{
    cell::RefCell,
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet},
    fmt,
};

use derive_more::{Display, Error, From};
use derive_where::derive_where;
use hugr::{
    hugr::SimpleReplacementError, Direction, Hugr, HugrView, IncomingPort, Node, Port, Wire,
};
use itertools::Itertools;
use relrc::RelRc;

use crate::{
    circuit::{HashError, HashedCircuit},
    Circuit, CircuitError,
};

/// A persistent data structure for circuit transformations.
///
/// A circuit diff stores a circuit along with the subgraphs in its parent
/// diffs that it overwrites. It is an immutable, reference counted type that
/// will remain alive as long as a reference is kept to it or any of its
/// descendants.
///
/// Use [`CircuitDiff::try_from_circuit`] to create a new "root" diff, i.e.
/// without any parents, and use [`CircuitDiff::apply_replacement`] to create
/// new diffs as children of the current diff.
#[derive(From)]
#[derive_where(Clone)]
pub struct CircuitDiff<H = Hugr>(RelRc<CircuitDiffData<H>, InvalidNodes>);

#[derive(Clone)]
#[derive_where(Hash; H: HugrView)]
#[derive_where(Debug; H: HugrView + fmt::Debug)]
struct CircuitDiffData<H> {
    circuit: HashedCircuit<H>,
    equivalent_wires: WireEquivalence<H>,
}

impl<H: HugrView> fmt::Debug for CircuitDiff<H> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CircuitDiff")
            .field("circuit_hash", &self.0.hash_id())
            .field("equivalent_wires", &self.0.value().equivalent_wires)
            .field(
                "parents",
                &self
                    .0
                    .all_incoming()
                    .iter()
                    .map(|e| (e.source().hash_id(), e.value()))
                    .collect_vec(),
            )
            .finish()
    }
}

/// Nodes that have been invalidated by the parent-child rewrite
pub type InvalidNodes = BTreeSet<Node>;

type WeakEdge<H> = relrc::WeakEdge<CircuitDiffData<H>, InvalidNodes>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct ParentWire {
    /// The index of the incoming edge to the parent node
    incoming_index: usize,
    /// The wire in the parent node
    wire: Wire,
}

#[derive(Clone)]
struct ChildWire<H> {
    /// Edge to the child node
    // (always use weak references to children)
    edge: WeakEdge<H>,
    /// The wire in the child node
    wire: Wire,
}

// TODO: RelRc currently implements Hash based on pointer values, might need
// to change this in the future (see https://github.com/lmondada/relrc/issues/4)
// Cannot use `derive_where` because clippy linting fails within the macro, so
// cannot be allowed
impl<H> std::hash::Hash for ChildWire<H> {
    fn hash<St>(&self, state: &mut St)
    where
        St: std::hash::Hasher,
    {
        self.edge.hash(state);
        self.wire.hash(state);
    }
}
impl<H: HugrView> PartialEq for ChildWire<H> {
    fn eq(&self, other: &Self) -> bool {
        self.edge.ptr_eq(&other.edge) && self.wire == other.wire
    }
}
impl<H: HugrView> Eq for ChildWire<H> {}

impl<H: HugrView> PartialOrd for ChildWire<H> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<H: HugrView> Ord for ChildWire<H> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.edge
            .target()
            .as_ptr()
            .cmp(&other.edge.target().as_ptr())
            .then(self.wire.cmp(&other.wire))
    }
}

impl<H: HugrView> fmt::Debug for ChildWire<H> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let target = self.edge.upgrade().map(|e| e.target().hash_id());
        f.debug_struct("ChildWire")
            .field("edge target", &target)
            .field("wire", &self.wire)
            .finish()
    }
}

/// A data structure that maps wires to their equivalent wires in parents
/// and children
#[derive(Clone)]
#[derive_where(Default, Hash)]
#[derive_where(Debug; H: HugrView)]
pub struct WireEquivalence<H = Hugr> {
    /// Map each input wire to a wire in one of the parents
    input_to_parent: BTreeMap<Wire, ParentWire>,

    /// Map each output wire to zero, one or several wires in parents
    output_to_parent: BTreeMap<Wire, BTreeSet<ParentWire>>,

    /// Map wires to equivalent wires in children
    #[derive_where(skip)] // skip hashing this field
    wire_to_children: RefCell<BTreeMap<Wire, BTreeSet<ChildWire<H>>>>,
}

type CircuitDiffPtr<H> = *const relrc::node::InnerData<CircuitDiffData<H>, InvalidNodes>;

impl<H> CircuitDiff<H> {
    /// Get the pointer to the inner data of the diff
    fn as_ptr(&self) -> CircuitDiffPtr<H> {
        self.0.as_ptr()
    }
}

impl<H: HugrView<Node = Node>> CircuitDiff<H> {
    /// Create a new circuit diff from a circuit
    pub fn try_from_circuit(circuit: Circuit<H>) -> Result<Self, HashError> {
        let data = CircuitDiffData {
            circuit: circuit.try_into()?,
            equivalent_wires: WireEquivalence::new(),
        };
        Ok(Self(RelRc::new(data)))
    }

    /// Get the circuit of the diff
    pub fn as_circuit(&self) -> &Circuit<H> {
        self.0.value().circuit.circuit()
    }

    /// Get the io nodes of the diff
    pub fn io_nodes(&self) -> [Node; 2] {
        self.as_circuit().io_nodes()
    }

    /// Get the diff circuit as a hugr
    pub fn as_hugr(&self) -> &H {
        self.as_circuit().hugr()
    }

    /// Create a new diff with the given parents
    ///
    /// This will update the wire equivalences in the parents with the
    /// new child wires.
    fn with_parents(
        data: CircuitDiffData<H>,
        parents: impl IntoIterator<Item = (CircuitDiff<H>, InvalidNodes)>,
    ) -> Self {
        let new = Self(RelRc::with_parents(
            data,
            parents.into_iter().map(|(d, i)| (d.0, i)),
        ));

        // register the new equivalent wires in the parents
        let new_eq_wires = &new.0.value().equivalent_wires;
        for (child_wire, parent_wire) in new_eq_wires.all_equivalent_wires_in_parents() {
            let ParentWire {
                incoming_index,
                wire: parent_wire,
            } = parent_wire;
            let edge = new
                .0
                .incoming_weak(incoming_index)
                .expect("invalid parent index");
            let parent = edge.upgrade().expect("child alive").source().clone();
            let wire_to_children = &parent.value().equivalent_wires.wire_to_children;
            let child_wire = ChildWire {
                edge,
                wire: child_wire,
            };
            wire_to_children
                .borrow_mut()
                .entry(parent_wire)
                .or_default()
                .insert(child_wire);
        }

        new
    }

    fn wire_to_children(&self, wire: Wire) -> Vec<(Self, Wire)> {
        let mut w = self
            .0
            .value()
            .equivalent_wires
            .wire_to_children
            .borrow_mut();
        let Some(wire_to_children_mut) = w.get_mut(&wire) else {
            return vec![];
        };
        let mut wire_to_children = Vec::new();
        wire_to_children_mut.retain(|child_wire| {
            // remove edges to no longer existing nodes
            let Some(target) = child_wire.edge.target().upgrade() else {
                return false;
            };
            wire_to_children.push((CircuitDiff(target), child_wire.wire));
            true
        });

        wire_to_children
    }

    fn input_to_parent(&self, wire: Wire) -> Option<ParentWire> {
        self.0
            .value()
            .equivalent_wires
            .input_to_parent
            .get(&wire)
            .copied()
    }

    fn output_to_parent(&self, wire: Wire) -> Option<&BTreeSet<ParentWire>> {
        self.0.value().equivalent_wires.output_to_parent.get(&wire)
    }

    fn get_parent(&self, parent_wire: &ParentWire) -> Self {
        let edge = self
            .0
            .incoming(parent_wire.incoming_index)
            .expect("invalid parent index");
        CircuitDiff(edge.source().clone())
    }

    fn all_parents(&self) -> impl ExactSizeIterator<Item = Self> + '_ {
        self.0.all_parents().cloned().map_into()
    }

    /// Get equivalent ports in children of a given port using the wire equivalences
    fn equivalent_children_ports<'a>(
        &'a self,
        node: Node,
        port: Port,
    ) -> impl Iterator<Item = Owned<H, (Node, Port)>> + 'a {
        let Ok(wire) = port_to_wire(node, port, self.as_hugr()) else {
            return None.into_iter().flatten();
        };
        let iter = self
            .wire_to_children(wire)
            .into_iter()
            .flat_map(move |(child, wire)| {
                let to_owned = |data| Owned {
                    owner: child.clone(),
                    data,
                };
                wire_to_ports(wire, port.direction(), child.as_hugr())
                    .map(to_owned)
                    .collect_vec()
            });
        Some(iter).into_iter().flatten()
    }

    /// Get equivalent ports in parents of a given port using the wire equivalences
    // TODO: make stronger assumptions on the kinds of wires in `input_to_parent`
    // and `output_to_parent` to make this more efficient
    fn equivalent_parent_ports<'a>(
        &'a self,
        node: Node,
        port: Port,
    ) -> impl Iterator<Item = Owned<H, (Node, Port)>> + 'a {
        let Ok(wire) = port_to_wire(node, port, self.as_hugr()) else {
            return None.into_iter().flatten();
        };
        let inputs = self.input_to_parent(wire).into_iter();
        let outputs = self.output_to_parent(wire).into_iter().flatten().copied();
        let iter = inputs.chain(outputs).flat_map(move |parent_wire| {
            let parent = self.get_parent(&parent_wire);
            let to_owned = |data| Owned {
                owner: parent.clone(),
                data,
            };
            wire_to_ports(parent_wire.wire, port.direction(), parent.as_hugr())
                .map(to_owned)
                .collect_vec()
        });
        Some(iter.unique_by(|o| (o.owner.as_ptr(), o.data)))
            .into_iter()
            .flatten()
    }
}

impl<H> WireEquivalence<H> {
    fn new() -> Self {
        Self::default()
    }

    /// All wires in `self` with their equivalent wires in parents
    fn all_equivalent_wires_in_parents(&self) -> impl Iterator<Item = (Wire, ParentWire)> + '_ {
        let input_wires = self.input_to_parent.iter();
        let output_wires = self
            .output_to_parent
            .iter()
            .flat_map(|(child_wire, parent_wires)| {
                parent_wires
                    .iter()
                    .map(move |parent_wire| (child_wire, parent_wire))
            });
        input_wires
            .chain(output_wires)
            .map(|(&wire, &parent_wire)| (wire, parent_wire))
    }
}

/// Errors that can occur when creating a circuit diff
#[derive(Display, Debug, Clone, Error, PartialEq, From)]
#[non_exhaustive]
pub enum CircuitDiffError {
    /// Error when hashing a circuit
    HashError(HashError),
    /// Error when converting a circuit to a hugr
    CircuitError(CircuitError),
    /// Error when applying a simple replacement
    SimpleReplacementError(SimpleReplacementError),
    /// Error when a node has no unique output
    #[display("no unique output for node {_0:?} and port {_1:?}")]
    NoUniqueOutput(Node, IncomingPort),
    /// Error when a cycle is detected in the dfg
    #[display("cycle detected in dfg")]
    Cycle,
    /// Error when merging two diffs
    #[display("conflicting diffs")]
    ConflictingDiffs,
    /// Error when a history is empty
    #[display("empty history")]
    EmptyHistory,
    /// Error when merging two diffs with different roots
    #[display("distinct roots")]
    DistinctRoots,
}

fn port_to_wire(
    node: Node,
    port: impl Into<Port>,
    hugr: &impl HugrView<Node = Node>,
) -> Result<Wire, CircuitDiffError> {
    let port: Port = port.into();

    use itertools::Either::{Left, Right};
    match port.as_directed() {
        Left(incoming) => {
            let (node, outgoing) = hugr
                .single_linked_output(node, incoming)
                .ok_or(CircuitDiffError::NoUniqueOutput(node, incoming))?;
            Ok(Wire::new(node, outgoing))
        }
        Right(outgoing) => Ok(Wire::new(node, outgoing)),
    }
}

fn wire_to_ports(
    wire: Wire,
    dir: Direction,
    hugr: &impl HugrView<Node = Node>,
) -> impl Iterator<Item = (Node, Port)> + '_ {
    use itertools::Either::{Left, Right};
    let iter = match dir {
        Direction::Incoming => Left(
            hugr.linked_inputs(wire.node(), wire.source())
                .map(|(node, port)| (node, port.into())),
        ),
        Direction::Outgoing => Right([(wire.node(), wire.source().into())]),
    };
    iter.into_iter()
}

/// Data in a circuit diff, along with its owner [`CircuitDiff`]
#[derive_where(Clone; D)]
pub struct Owned<H, D> {
    /// The owner of the data
    pub owner: CircuitDiff<H>,
    /// The data
    pub data: D,
}
