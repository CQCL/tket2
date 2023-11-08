//! Transform circuits using rewrite rules.

#[cfg(feature = "portmatching")]
pub mod ecc_rewriter;
pub mod strategy;
pub mod trace;

use bytemuck::TransparentWrapper;
#[cfg(feature = "portmatching")]
pub use ecc_rewriter::ECCRewriter;

use delegate::delegate;
use derive_more::{From, Into};
use hugr::hugr::views::sibling_subgraph::{InvalidReplacement, InvalidSubgraph};
use hugr::Node;
use hugr::{
    hugr::{hugrmut::HugrMut, views::SiblingSubgraph, Rewrite, SimpleReplacementError},
    Hugr, SimpleReplacement,
};

use crate::circuit::Circuit;

/// A subcircuit of a circuit.
#[derive(Debug, Clone, From, Into)]
#[repr(transparent)]
pub struct Subcircuit {
    pub(crate) subgraph: SiblingSubgraph,
}

unsafe impl TransparentWrapper<SiblingSubgraph> for Subcircuit {}

impl Subcircuit {
    /// Create a new subcircuit induced from a set of nodes.
    pub fn try_from_nodes(
        nodes: impl Into<Vec<Node>>,
        hugr: &Hugr,
    ) -> Result<Self, InvalidSubgraph> {
        let subgraph = SiblingSubgraph::try_from_nodes(nodes, hugr)?;
        Ok(Self { subgraph })
    }

    /// Nodes in the subcircuit.
    pub fn nodes(&self) -> &[Node] {
        self.subgraph.nodes()
    }

    /// Number of nodes in the subcircuit.
    pub fn node_count(&self) -> usize {
        self.subgraph.node_count()
    }

    /// Create a rewrite rule to replace the subcircuit.
    pub fn create_rewrite(
        &self,
        source: &Hugr,
        target: Hugr,
    ) -> Result<CircuitRewrite, InvalidReplacement> {
        Ok(CircuitRewrite(
            self.subgraph.create_simple_replacement(source, target)?,
        ))
    }
}

/// A rewrite rule for circuits.
#[derive(Debug, Clone, From, Into)]
pub struct CircuitRewrite(SimpleReplacement);

impl CircuitRewrite {
    /// Create a new rewrite rule.
    pub fn try_new(
        source_position: &Subcircuit,
        source: &Hugr,
        target: Hugr,
    ) -> Result<Self, InvalidReplacement> {
        source_position
            .subgraph
            .create_simple_replacement(source, target)
            .map(Self)
    }

    /// Number of nodes added or removed by the rewrite.
    ///
    /// The difference between the new number of nodes minus the old. A positive
    /// number is an increase in node count, a negative number is a decrease.
    pub fn node_count_delta(&self) -> isize {
        let new_count = self.replacement().num_gates() as isize;
        let old_count = self.subcircuit().node_count() as isize;
        new_count - old_count
    }

    /// The subcircuit that is replaced.
    pub fn subcircuit(&self) -> &Subcircuit {
        Subcircuit::wrap_ref(self.0.subgraph())
    }

    /// The replacement subcircuit.
    pub fn replacement(&self) -> &Hugr {
        self.0.replacement()
    }

    /// Returns a set of nodes referenced by the rewrite. Modifying any these
    /// nodes will invalidate it.
    ///
    /// Two `CircuitRewrite`s can be composed if their invalidation sets are
    /// disjoint.
    #[inline]
    pub fn invalidation_set(&self) -> impl Iterator<Item = Node> + '_ {
        self.0.invalidation_set()
    }

    delegate! {
        to self.0 {
            /// Apply the rewrite rule to a circuit.
            pub fn apply(self, circ: &mut impl HugrMut) -> Result<(), SimpleReplacementError>;
        }
    }
}

/// Generate rewrite rules for circuits.
pub trait Rewriter {
    /// Get the rewrite rules for a circuit.
    fn get_rewrites<C: Circuit + Clone>(&self, circ: &C) -> Vec<CircuitRewrite>;
}
