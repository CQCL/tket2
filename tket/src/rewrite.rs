//! Transform circuits using rewrite rules.

#[cfg(feature = "portmatching")]
pub mod ecc_rewriter;
pub mod matcher;
pub mod strategy;
pub mod trace;

use derive_more::derive::{Display, Error};
#[cfg(feature = "portmatching")]
pub use ecc_rewriter::ECCRewriter;

use derive_more::{From, Into};
use hugr::core::HugrNode;
use hugr::hugr::hugrmut::HugrMut;
use hugr::hugr::patch::simple_replace;
use hugr::hugr::views::sibling_subgraph::InvalidSubgraph;
use hugr::hugr::Patch;
use hugr::types::Signature;
use hugr::{
    hugr::{views::SiblingSubgraph, SimpleReplacementError},
    SimpleReplacement,
};
use hugr::{Hugr, HugrView};
use itertools::Either;

use crate::circuit::Circuit;
use crate::resource::ResourceScope;
pub use crate::Subcircuit;

/// A rewrite rule for circuits.
///
/// As a temporary solution, it support both old school [`SimpleReplacement`]s
/// as well as the much more civilised approach using [`ResourceScope`] and
/// [`Subcircuit`].
#[derive(Debug, Clone, From)]
pub enum CircuitRewrite<N: HugrNode = hugr::Node> {
    /// A rewrite rule expressed as a subcircuit and replacement circuit.
    New(NewCircuitRewrite<N>),
    /// A rewrite rule expressed as a [`SimpleReplacement`].
    ///
    /// Prefer using [`NewCircuitRewrite`] instead. It is much faster (but is
    /// not yet supported in portmatching and the Python interface).
    Old(#[from] OldCircuitRewrite<N>),
}

/// A rewrite rule for circuits.
#[derive(Debug, Clone)]
pub struct NewCircuitRewrite<N: HugrNode = hugr::Node> {
    subcircuit: Subcircuit<N>,
    replacement: Circuit,
}

/// A rewrite rule for circuits, wrapping a HUGR [`SimpleReplacement`].
///
/// You should migrate to using [`NewCircuitRewrite`] instead. It is much faster.
#[derive(Debug, Clone, From, Into)]
pub struct OldCircuitRewrite<N = hugr::Node>(SimpleReplacement<N>);

impl<N: HugrNode> CircuitRewrite<N> {
    /// Create a new rewrite that can be applied to `hugr`.
    pub fn try_new(
        subcircuit: Subcircuit<N>,
        circuit: &ResourceScope<impl HugrView<Node = N>>,
        replacement: Circuit,
    ) -> Result<Self, InvalidRewrite> {
        subcircuit
            .validate_subgraph(circuit)
            .map_err(|err| InvalidRewrite::try_from(err).unwrap_or_else(|err| panic!("{err}")))?;

        let subcircuit_sig = subcircuit.dataflow_signature(circuit);
        let replacement_sig = replacement.circuit_signature();
        if subcircuit_sig != replacement_sig {
            return Err(InvalidRewrite::InvalidSignature {
                expected: subcircuit_sig,
                actual: replacement_sig.into_owned(),
            });
        }

        Ok(Self::New(NewCircuitRewrite {
            subcircuit,
            replacement,
        }))
    }

    /// Number of nodes added or removed by the rewrite.
    ///
    /// The difference between the new number of nodes minus the old. A positive
    /// number is an increase in node count, a negative number is a decrease.
    pub fn node_count_delta(&self, circuit: &ResourceScope<impl HugrView<Node = N>>) -> isize {
        match self {
            Self::New(rewrite) => {
                compute_node_count_delta(&rewrite.subcircuit, rewrite.replacement.hugr(), circuit)
            }
            Self::Old(OldCircuitRewrite(simple_replacement)) => {
                let old_count = simple_replacement.subgraph().node_count() as isize;
                let new_count =
                    Circuit::new(simple_replacement.replacement()).num_operations() as isize;
                new_count - old_count
            }
        }
    }

    /// Construct a [`SiblingSubgraph`] that represents the subcircuit being
    /// replaced.
    pub fn to_subgraph(
        &self,
        circuit: &ResourceScope<impl HugrView<Node = N>>,
    ) -> SiblingSubgraph<N> {
        match self {
            Self::New(rewrite) => rewrite
                .subcircuit
                .try_to_subgraph(circuit)
                .expect("subcircuit is valid subgraph"),
            Self::Old(rewrite) => rewrite.0.subgraph().to_owned(),
        }
    }

    /// The replacement subcircuit.
    pub fn replacement(&self) -> &Hugr {
        match self {
            Self::New(rewrite) => rewrite.replacement.hugr(),
            Self::Old(rewrite) => rewrite.0.replacement(),
        }
    }

    /// Construct a [`SimpleReplacement`] that executes the rewrite as a HUGR
    /// operation.
    pub fn to_simple_replacement(
        &self,
        circuit: &ResourceScope<impl HugrView<Node = N>>,
    ) -> SimpleReplacement<N> {
        self.to_subgraph(circuit)
            .create_simple_replacement(circuit.hugr(), self.replacement().to_owned())
            .expect("rewrite is valid simple replacement")
    }

    /// Returns a set of nodes referenced by the rewrite. Modifying any these
    /// nodes will invalidate it.
    ///
    /// Two `CircuitRewrite`s can be composed if their invalidation sets are
    /// disjoint.
    #[inline]
    pub fn invalidation_set<'a>(
        &'a self,
        circuit: &'a ResourceScope<impl HugrView<Node = N>>,
    ) -> impl Iterator<Item = N> + 'a {
        match self {
            Self::New(rewrite) => Either::Left(rewrite.subcircuit.nodes(circuit)),
            Self::Old(rewrite) => Either::Right(rewrite.0.subgraph().nodes().iter().copied()),
        }
    }

    /// Apply the rewrite rule to a circuit.
    #[inline]
    pub fn apply(
        self,
        circ: &mut ResourceScope<impl HugrMut<Node = N>>,
    ) -> Result<simple_replace::Outcome<N>, SimpleReplacementError> {
        circ.as_circuit_mut().add_rewrite_trace(&self);
        self.to_simple_replacement(circ).apply(circ.hugr_mut())
    }

    /// Apply the rewrite rule to a circuit, without registering it in the rewrite trace.
    #[inline]
    pub fn apply_notrace(
        self,
        circ: &mut ResourceScope<impl HugrMut<Node = N>>,
    ) -> Result<simple_replace::Outcome<N>, SimpleReplacementError> {
        self.to_simple_replacement(circ).apply(circ.hugr_mut())
    }
}

impl<N: HugrNode> From<SimpleReplacement<N>> for CircuitRewrite<N> {
    fn from(value: SimpleReplacement<N>) -> Self {
        OldCircuitRewrite(value).into()
    }
}

/// Generate rewrite rules for circuits.
///
/// The generic argument `C` (default: [`Circuit`]) is the type of circuit to
/// find rewrites on. Currently, only arguments of type [`Circuit<H>`] are
/// supported.
pub trait Rewriter<C: CircuitLike = Circuit> {
    /// Get the rewrite rules for a circuit.
    fn get_rewrites(&self, circ: &C) -> Vec<CircuitRewrite<C::Node>>;
}

// A simple trait to get the node type of a circuit. This will allow us to
// support circuit-like types (e.g. persistent circuits) in the future.
mod hidden {
    use hugr::{core::HugrNode, HugrView};

    use crate::{resource::ResourceScope, Circuit};

    pub trait CircuitLike {
        type Node: HugrNode;
    }

    impl<H: HugrView> CircuitLike for H {
        type Node = H::Node;
    }

    impl<H: HugrView> CircuitLike for Circuit<H> {
        type Node = H::Node;
    }

    impl<H: HugrView> CircuitLike for ResourceScope<H> {
        type Node = H::Node;
    }
}
use hidden::CircuitLike;

/// An error that can occur when constructing a rewrite rule.
#[derive(Debug, Clone, PartialEq, Display, Error)]
#[non_exhaustive]
pub enum InvalidRewrite {
    /// The LHS subcircuit is not convex.
    #[display("The LHS subcircuit is not convex.")]
    NonConvexSubgraph,
    /// The LHS subcircuit is empty.
    #[display("The LHS subcircuit is empty.")]
    EmptySubgraph,
    /// The left and right hand sides have mismatched signatures.
    #[display("The left and right hand sides have mismatched signatures: expected {expected:?}, got {actual:?}.")]
    InvalidSignature {
        /// The expected signature.
        expected: Signature,
        /// The actual signature.
        actual: Signature,
    },
}

impl<N: HugrNode> TryFrom<InvalidSubgraph<N>> for InvalidRewrite {
    type Error = &'static str;

    fn try_from(value: InvalidSubgraph<N>) -> Result<Self, Self::Error> {
        match value {
            InvalidSubgraph::NotConvex => Ok(InvalidRewrite::NonConvexSubgraph),
            InvalidSubgraph::EmptySubgraph => Ok(InvalidRewrite::EmptySubgraph),
            _ => return Err("Unexpected InvalidSubgraph error"),
        }
    }
}

fn compute_node_count_delta<N: HugrNode>(
    subcircuit: &Subcircuit<N>,
    replacement: &Hugr,
    circuit: &ResourceScope<impl HugrView<Node = N>>,
) -> isize {
    let new_count = Circuit::new(replacement).num_operations() as isize;
    let old_count = subcircuit.nodes(circuit).count() as isize;
    new_count - old_count
}
