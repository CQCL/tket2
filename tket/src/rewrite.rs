//! Transform circuits using rewrite rules.

#[cfg(feature = "portmatching")]
pub mod ecc_rewriter;
pub mod matcher;
pub mod replacer;
pub mod strategy;
pub mod trace;

#[cfg(feature = "portmatching")]
pub use ecc_rewriter::ECCRewriter;

use derive_more::{From, Into};
use hugr::core::HugrNode;
use hugr::hugr::hugrmut::HugrMut;
use hugr::hugr::patch::simple_replace;
use hugr::hugr::views::sibling_subgraph::InvalidReplacement;
use hugr::hugr::Patch;
use hugr::{
    hugr::{views::SiblingSubgraph, SimpleReplacementError},
    SimpleReplacement,
};
use hugr::{Hugr, HugrView};
use matcher::{CircuitMatcher, MatchingOptions};
use replacer::CircuitReplacer;

use crate::circuit::Circuit;
pub use crate::Subcircuit;

/// A rewrite rule for circuits.
#[derive(Debug, Clone, From, Into)]
pub struct CircuitRewrite<N = hugr::Node>(SimpleReplacement<N>);

impl<N: HugrNode> CircuitRewrite<N> {
    /// Create a new rewrite rule.
    pub fn try_new(
        subgraph: &SiblingSubgraph<N>,
        hugr: &impl HugrView<Node = N>,
        replacement: Circuit<impl HugrView<Node = hugr::Node>>,
    ) -> Result<Self, InvalidReplacement> {
        let replacement = replacement
            .extract_dfg()
            .unwrap_or_else(|e| panic!("{}", e))
            .into_hugr();
        Ok(Self(subgraph.create_simple_replacement(hugr, replacement)?))
    }

    /// Number of nodes added or removed by the rewrite.
    ///
    /// The difference between the new number of nodes minus the old. A positive
    /// number is an increase in node count, a negative number is a decrease.
    pub fn node_count_delta(&self) -> isize {
        let new_count = self.replacement().num_operations() as isize;
        let old_count = self.subgraph().node_count() as isize;
        new_count - old_count
    }

    /// The subgraph that is replaced.
    pub fn subgraph(&self) -> &SiblingSubgraph<N> {
        self.0.subgraph()
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
    pub fn invalidation_set(&self) -> impl Iterator<Item = N> + '_ {
        self.0.invalidation_set()
    }

    /// Apply the rewrite rule to a circuit.
    #[inline]
    pub fn apply(
        self,
        circ: &mut Circuit<impl HugrMut<Node = N>>,
    ) -> Result<simple_replace::Outcome<N>, SimpleReplacementError> {
        circ.add_rewrite_trace(&self);
        self.0.apply(circ.hugr_mut())
    }

    /// Apply the rewrite rule to a circuit, without registering it in the rewrite trace.
    #[inline]
    pub fn apply_notrace(
        self,
        circ: &mut Circuit<impl HugrMut<Node = N>>,
    ) -> Result<simple_replace::Outcome<N>, SimpleReplacementError> {
        self.0.apply(circ.hugr_mut())
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
    use hugr::HugrView;

    use crate::Circuit;

    pub trait CircuitLike {
        type Node;
    }

    impl<H: HugrView> CircuitLike for Circuit<H> {
        type Node = H::Node;
    }
}
use hidden::CircuitLike;

/// A rewriter made of a [`CircuitMatcher`] and a [`CircuitReplacer`].
///
/// The [`CircuitMatcher`] is used to find matches in the circuit, and the
/// [`CircuitReplacer`] is used to create [`CircuitRewrite`]s for each match.
#[derive(Clone, Debug)]
pub struct MatchReplaceRewriter<C, R> {
    matcher: C,
    replacer: R,
}

impl<C, R> MatchReplaceRewriter<C, R> {
    /// Create a new [`MatchReplaceRewriter`].
    pub fn new(matcher: C, replacement: R) -> Self {
        Self {
            matcher,
            replacer: replacement,
        }
    }
}

impl<C, R, H: HugrView<Node = hugr::Node>> Rewriter<Circuit<H>> for MatchReplaceRewriter<C, R>
where
    C: CircuitMatcher,
    R: CircuitReplacer<C::MatchInfo>,
{
    fn get_rewrites(&self, circ: &Circuit<H>) -> Vec<CircuitRewrite<H::Node>> {
        let hugr = circ.hugr();
        let matches = self
            .matcher
            .as_hugr_matcher()
            .get_all_matches(circ, &MatchingOptions::default());
        matches
            .into_iter()
            .flat_map(|(subgraph, match_info)| {
                self.replacer
                    .replace_match(&subgraph, hugr, match_info)
                    .into_iter()
                    .filter_map(move |repl| CircuitRewrite::try_new(&subgraph, hugr, repl).ok())
            })
            .collect()
    }
}
