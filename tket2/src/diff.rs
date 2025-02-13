//! Persistent data structures for circuit rewriting.
//!
//! This module contains the definition of the [`CircuitDiff`] type, which
//! represent local circuit transformations. This enables persistent rewriting,
//! in which the original data is never changed, but new diffs are created to
//! represent the changes.
//!
//! Diffs can be created on top of existing diffs, resulting in an acyclic
//! history of circuit transformations.

mod replacement;

use std::{
    cell::RefCell,
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet},
    fmt,
};

use derive_more::{Display, Error, From};
use derive_where::derive_where;
use hugr::{hugr::SimpleReplacementError, Hugr, HugrView, IncomingPort, Node, Wire};
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
#[derive(Clone)]
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
#[derive_where(Hash)]
struct ChildWire<H> {
    /// Edge to the child node
    // (always use weak references to children)
    edge: WeakEdge<H>,
    /// The wire in the child node
    wire: Wire,
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

impl<H: HugrView> CircuitDiff<H> {
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
}
