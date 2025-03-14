//! Transform circuits using rewrite rules.

#[cfg(feature = "portmatching")]
pub mod ecc_rewriter;
pub mod strategy;
pub mod trace;

use bytemuck::TransparentWrapper;
#[cfg(feature = "portmatching")]
pub use ecc_rewriter::ECCRewriter;

use derive_more::{From, Into};
use hugr::core::HugrNode;
use hugr::hugr::hugrmut::HugrMut;
use hugr::hugr::views::sibling_subgraph::{InvalidReplacement, InvalidSubgraph};
use hugr::hugr::views::ExtractHugr;
use hugr::ops::OpType;
use hugr::types::Signature;
use hugr::{
    hugr::{views::SiblingSubgraph, Rewrite, SimpleReplacementError},
    SimpleReplacement,
};
use hugr::{Hugr, HugrView, Node};

use crate::circuit::Circuit;

/// A subcircuit of a circuit.
#[derive(Debug, Clone, From, Into)]
#[repr(transparent)]
pub struct Subcircuit<N = Node> {
    pub(crate) subgraph: SiblingSubgraph<N>,
}

unsafe impl<N> TransparentWrapper<SiblingSubgraph<N>> for Subcircuit<N> {}

impl<N: HugrNode> Subcircuit<N> {
    /// Create a new subcircuit induced from a set of nodes.
    pub fn try_from_nodes(
        nodes: impl Into<Vec<N>>,
        circ: &Circuit<impl HugrView<Node = N>, N>,
    ) -> Result<Self, InvalidSubgraph<N>> {
        let subgraph = SiblingSubgraph::try_from_nodes(nodes, circ.hugr())?;
        Ok(Self { subgraph })
    }

    /// Nodes in the subcircuit.
    pub fn nodes(&self) -> &[N] {
        self.subgraph.nodes()
    }

    /// Number of nodes in the subcircuit.
    pub fn node_count(&self) -> usize {
        self.subgraph.node_count()
    }

    /// The signature of the subcircuit.
    pub fn signature(&self, circ: &Circuit<impl HugrView<Node = N>, N>) -> Signature {
        self.subgraph.signature(circ.hugr())
    }
}

impl Subcircuit<Node> {
    /// Create a rewrite rule to replace the subcircuit with a new circuit.
    ///
    /// # Parameters
    /// * `circuit` - The base circuit that contains the subcircuit.
    /// * `replacement` - The new circuit to replace the subcircuit with.
    pub fn create_rewrite(
        &self,
        circuit: &Circuit<impl HugrView<Node = Node>>,
        replacement: Circuit<impl ExtractHugr<Node = Node>>,
    ) -> Result<CircuitRewrite, InvalidReplacement> {
        // The replacement must be a Dfg rooted hugr.
        let replacement = replacement
            .extract_dfg()
            .unwrap_or_else(|e| panic!("{}", e))
            .into_hugr();
        Ok(CircuitRewrite(
            self.subgraph
                .create_simple_replacement(circuit.hugr(), replacement)?,
        ))
    }
}

/// A rewrite rule for circuits.
#[derive(Debug, Clone, From, Into)]
pub struct CircuitRewrite<N = Node>(SimpleReplacement<N>);

impl CircuitRewrite {
    /// Create a new rewrite rule.
    pub fn try_new(
        circuit_position: &Subcircuit,
        circuit: &Circuit<impl HugrView<Node = Node>>,
        replacement: Circuit<impl ExtractHugr<Node = Node>>,
    ) -> Result<Self, InvalidReplacement> {
        let replacement = replacement
            .extract_dfg()
            .unwrap_or_else(|e| panic!("{}", e))
            .into_hugr();
        circuit_position
            .subgraph
            .create_simple_replacement(circuit.hugr(), replacement)
            .map(Self)
    }

    /// Number of nodes added or removed by the rewrite.
    ///
    /// The difference between the new number of nodes minus the old. A positive
    /// number is an increase in node count, a negative number is a decrease.
    pub fn node_count_delta(&self) -> isize {
        let new_count = self.replacement().num_operations() as isize;
        let old_count = self.subcircuit().node_count() as isize;
        new_count - old_count
    }

    /// The subcircuit that is replaced.
    pub fn subcircuit(&self) -> &Subcircuit {
        Subcircuit::wrap_ref(self.0.subgraph())
    }

    /// The replacement subcircuit.
    pub fn replacement(&self) -> Circuit<&Hugr> {
        self.0.replacement().into()
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

    /// Apply the rewrite rule to a circuit.
    #[inline]
    pub fn apply(
        self,
        circ: &mut Circuit<impl HugrMut>,
    ) -> Result<Vec<(Node, OpType)>, SimpleReplacementError> {
        circ.add_rewrite_trace(&self);
        self.0.apply(circ.hugr_mut())
    }

    /// Apply the rewrite rule to a circuit, without registering it in the rewrite trace.
    #[inline]
    pub fn apply_notrace(
        self,
        circ: &mut Circuit<impl HugrMut>,
    ) -> Result<Vec<(Node, OpType)>, SimpleReplacementError> {
        self.0.apply(circ.hugr_mut())
    }
}

/// Generate rewrite rules for circuits.
pub trait Rewriter {
    /// Get the rewrite rules for a circuit.
    fn get_rewrites(&self, circ: &Circuit<impl HugrView<Node = Node>>) -> Vec<CircuitRewrite>;
}
