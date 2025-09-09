//! Transform circuits using rewrite rules.

#[cfg(feature = "portmatching")]
pub mod ecc_rewriter;
pub mod matcher;
pub mod replacer;
pub mod strategy;
pub mod trace;

use std::mem;

use derive_more::derive::{Display, Error};
#[cfg(feature = "portmatching")]
pub use ecc_rewriter::ECCRewriter;

use derive_more::{From, Into};
use hugr::core::HugrNode;
use hugr::hugr::hugrmut::HugrMut;
use hugr::hugr::patch::{simple_replace, PatchVerification};
use hugr::hugr::views::sibling_subgraph::InvalidSubgraph;
use hugr::hugr::views::NodesIter;
use hugr::hugr::Patch;
use hugr::types::Signature;
use hugr::{hugr::views::SiblingSubgraph, SimpleReplacement};
use hugr::{Hugr, HugrView};
use itertools::{Either, Itertools};
use matcher::{CircuitMatcher, MatchingOptions};
use replacer::CircuitReplacer;

use crate::circuit::Circuit;
use crate::resource::{CircuitRewriteError, ResourceScope};
pub use crate::Subcircuit;

/// A rewrite rule for circuits.
///
/// As a temporary solution, it support both old school [`SimpleReplacement`]s
/// as well as the much more civilised approach using [`ResourceScope`] and
/// [`Subcircuit`].
// TODO: get rid of OldCircuitRewrite, Rename NewCircuitRewrite to CircuitRewrite
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

impl PatchVerification for CircuitRewrite {
    type Error = CircuitRewriteError;

    type Node = hugr::Node;

    fn verify(&self, h: &impl HugrView<Node = Self::Node>) -> Result<(), Self::Error> {
        match self {
            CircuitRewrite::New(rewrite) => rewrite.verify(h),
            CircuitRewrite::Old(OldCircuitRewrite(repl)) => repl.verify(h).map_err(Into::into),
        }
    }
}

impl PatchVerification for NewCircuitRewrite {
    type Error = CircuitRewriteError;

    type Node = hugr::Node;

    fn verify(&self, h: &impl HugrView<Node = Self::Node>) -> Result<(), Self::Error> {
        let circ = ResourceScope::from_circuit(Circuit::new(h));
        self.to_simple_replacement(&circ)
            .verify(circ.hugr())
            .map_err(Into::into)
    }
}

/// A rewrite rule for circuits.
///
/// The following invariants hold:
///  - the subcircuit is not empty
///  - the subcircuit is convex
#[derive(Debug, Clone)]
pub struct NewCircuitRewrite<N: HugrNode = hugr::Node> {
    pub(crate) subcircuit: Subcircuit<N>,
    pub(crate) replacement: Circuit,
}

impl<N: HugrNode> NewCircuitRewrite<N> {
    /// Construct a [`SimpleReplacement`] that executes the rewrite as a HUGR
    /// operation.
    pub fn to_simple_replacement(
        &self,
        circuit: &ResourceScope<impl HugrView<Node = N>>,
    ) -> SimpleReplacement<N> {
        let subgraph = self
            .subcircuit
            .try_to_subgraph(circuit)
            .expect("subcircuit is valid subgraph");
        subgraph
            .create_simple_replacement(circuit.hugr(), self.replacement.clone().into_hugr())
            .expect("rewrite is valid simple replacement")
    }
}

/// A rewrite rule for circuits, wrapping a HUGR [`SimpleReplacement`].
///
/// You should migrate to using [`NewCircuitRewrite`] instead. It is much
/// faster.
#[derive(Debug, Clone, From, Into)]
pub struct OldCircuitRewrite<N = hugr::Node>(pub(crate) SimpleReplacement<N>);

impl<N: HugrNode> CircuitRewrite<N> {
    /// Create a new rewrite that can be applied to `hugr`.
    pub fn try_new(
        subcircuit: Subcircuit<N>,
        circuit: &ResourceScope<impl HugrView<Node = N>>,
        replacement: Circuit,
    ) -> Result<Self, InvalidRewrite> {
        subcircuit.validate_subgraph(circuit).map_err(|err| {
            InvalidRewrite::try_from(err).unwrap_or_else(|err| panic!("unknown error: {err}"))
        })?;

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
}

impl CircuitRewrite {
    /// Apply the rewrite rule to a circuit.
    #[inline]
    pub fn apply(
        self,
        circ: &mut ResourceScope<impl HugrMut<Node = hugr::Node>>,
    ) -> Result<simple_replace::Outcome, CircuitRewriteError> {
        circ.add_rewrite_trace(&self);
        circ.apply_rewrite(self)
    }

    /// Apply the rewrite rule to a circuit, without registering it in the
    /// rewrite trace.
    #[inline]
    pub fn apply_notrace(
        self,
        circ: &mut ResourceScope<impl HugrMut<Node = hugr::Node>>,
    ) -> Result<simple_replace::Outcome, CircuitRewriteError> {
        circ.apply_rewrite(self)
    }
}

impl<H: HugrMut<Node = hugr::Node>> Patch<H> for CircuitRewrite {
    type Outcome = simple_replace::Outcome;

    const UNCHANGED_ON_FAILURE: bool = true;

    fn apply(self, h: &mut H) -> Result<Self::Outcome, Self::Error> {
        let mut circ = Circuit::new(h);
        <Self as Patch<Circuit<&mut H>>>::apply(self, &mut circ)
    }
}

impl<H: HugrMut<Node = hugr::Node>> Patch<Circuit<H>> for CircuitRewrite {
    type Outcome = simple_replace::Outcome;

    const UNCHANGED_ON_FAILURE: bool = true;

    fn apply(self, h: &mut Circuit<H>) -> Result<Self::Outcome, Self::Error> {
        match self {
            CircuitRewrite::New(..) => {
                unimplemented!("use ResourceScope with new rewrite type")
            }
            CircuitRewrite::Old(OldCircuitRewrite(repl)) => {
                repl.apply(h.hugr_mut()).map_err(Into::into)
            }
        }
    }
}

impl<N: HugrNode> From<SimpleReplacement<N>> for CircuitRewrite<N> {
    fn from(value: SimpleReplacement<N>) -> Self {
        OldCircuitRewrite(value).into()
    }
}

/// Generate rewrite rules for circuits.
///
/// Every rewrite must have one designated root node in its left hand side (the
/// match). The API allows for queries that only find rewrites with a specific
/// root node.
///
/// The generic argument `C` (default: [`Circuit`]) is the type of
/// circuit to find rewrites on.
pub trait Rewriter<C: NodesIter = Circuit> {
    /// The type of rewrite rule to return.
    type Rewrite;

    /// Get the rewrite rules for a circuit with the given root node.
    fn get_rewrites(&self, circ: &C, root_node: C::Node) -> Vec<Self::Rewrite>;

    /// Get the rewrite rules for a circuit with all root nodes.
    fn get_all_rewrites(&self, circ: &C) -> Vec<Self::Rewrite> {
        circ.nodes()
            .flat_map(|n| self.get_rewrites(circ, n))
            .collect()
    }
}

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

fn compute_node_count_delta<N: HugrNode>(
    subcircuit: &Subcircuit<N>,
    replacement: &Hugr,
    circuit: &ResourceScope<impl HugrView<Node = N>>,
) -> isize {
    let new_count = Circuit::new(replacement).num_operations() as isize;
    let old_count = subcircuit.nodes(circuit).count() as isize;
    new_count - old_count
}

impl<C, R, H: HugrView<Node = hugr::Node>> Rewriter<ResourceScope<H>> for MatchReplaceRewriter<C, R>
where
    C: CircuitMatcher,
    R: CircuitReplacer<C::MatchInfo>,
{
    type Rewrite = CircuitRewrite;

    fn get_rewrites(&self, circ: &ResourceScope<H>, root_node: H::Node) -> Vec<CircuitRewrite> {
        let matches = self.matcher.as_hugr_matcher().get_matches(
            circ,
            root_node,
            &MatchingOptions::default(),
        );
        matches
            .into_iter()
            .flat_map(|(subcirc, match_info)| {
                self.replacer
                    .replace_match(&subcirc, circ, match_info)
                    .into_iter()
                    .filter_map(move |repl| {
                        match CircuitRewrite::try_new(subcirc.clone(), circ, repl) {
                            Ok(ok) => Some(ok),
                            Err(err) => {
                                eprintln!("Error: failed to create rewrite, skipping:\n{}", err);
                                None
                            }
                        }
                    })
            })
            .collect()
    }

    fn get_all_rewrites(&self, circ: &ResourceScope<H>) -> Vec<CircuitRewrite> {
        let matches = self
            .matcher
            .as_hugr_matcher()
            .get_all_matches(circ, &MatchingOptions::default());
        matches
            .into_iter()
            .flat_map(|(subcirc, match_info)| {
                self.replacer
                    .replace_match(&subcirc, circ, match_info)
                    .into_iter()
                    .filter_map(move |repl| {
                        match CircuitRewrite::try_new(subcirc.clone(), circ, repl) {
                            Ok(ok) => Some(ok),
                            Err(err) => {
                                eprintln!("Error: failed to create rewrite, skipping:\n{}", err);
                                None
                            }
                        }
                    })
            })
            .collect()
    }
}

#[cfg(feature = "badgerv2_unstable")]
mod badgerv2_unstable {
    use super::*;
    use crate::{rewrite::matcher::ImMatchResult, rewrite_space::RewriteSpace};
    use hugr::persistent::{Commit, CommitStateSpace, PatchNode, Walker};

    impl<'w, C, R> Rewriter<Walker<'w>> for MatchReplaceRewriter<C, R>
    where
        C: CircuitMatcher,
        R: CircuitReplacer<C::MatchInfo>,
    {
        type Rewrite = Commit<'w>;

        fn get_rewrites(&self, walker: &Walker<'w>, root_node: PatchNode) -> Vec<Commit<'w>> {
            let matches = self
                .matcher
                .as_rewrite_space_matcher()
                .get_matches(walker.clone(), root_node);
            self.im_matches_to_commits(matches, walker.as_hugr_view().state_space())
                // SAFETY: the commit is valid for the lifetime of the walker
                .map(|commit| unsafe { commit.upgrade_lifetime() })
                .collect()
        }
    }

    impl<'c, C, R, Cost> Rewriter<&'c RewriteSpace<Cost>> for MatchReplaceRewriter<C, R>
    where
        C: CircuitMatcher,
        R: CircuitReplacer<C::MatchInfo>,
    {
        type Rewrite = Commit<'c>;

        fn get_rewrites(
            &self,
            space: &&'c RewriteSpace<Cost>,
            root_node: PatchNode,
        ) -> Vec<Commit<'c>> {
            let walker = Walker::from_pinned_node(root_node, space.state_space());
            self.get_rewrites(&walker, root_node)
                .into_iter()
                // SAFETY: the commit is valid for the lifetime of the rewrite space
                .map(|commit| unsafe { commit.upgrade_lifetime() })
                .collect()
        }

        fn get_all_rewrites(&self, space: &&'c RewriteSpace<Cost>) -> Vec<Self::Rewrite> {
            let matches = self
                .matcher
                .as_rewrite_space_matcher()
                .get_all_matches(space);
            self.im_matches_to_commits(matches, space.state_space())
                .into_iter()
                // SAFETY: the commit is valid for the lifetime of the rewrite space
                .map(|commit| unsafe { commit.upgrade_lifetime() })
                .collect()
        }
    }

    impl<C, R> MatchReplaceRewriter<C, R>
    where
        C: CircuitMatcher,
        R: CircuitReplacer<C::MatchInfo>,
    {
        fn im_matches_to_commits<'c>(
            &'c self,
            matches: Vec<ImMatchResult<C::MatchInfo>>,
            state_space: &'c CommitStateSpace,
        ) -> impl Iterator<Item = Commit<'c>> + 'c {
            matches.into_iter().flat_map(
                move |ImMatchResult {
                          subcircuit,
                          subgraph,
                          match_info,
                          hugr,
                      }| {
                    self.replacer
                        .replace_match(&subcircuit, &hugr, match_info)
                        .into_iter()
                        .filter_map(move |repl| {
                            let rw = match CircuitRewrite::try_new(subcircuit.clone(), &hugr, repl)
                            {
                                Ok(ok) => Some(ok),
                                Err(err) => {
                                    eprintln!(
                                        "Error: failed to create rewrite, skipping:\n{}",
                                        err
                                    );
                                    None
                                }
                            }?;
                            let parents = subgraph
                                .selected_commits()
                                .map(|id| hugr.hugr().get_commit(id).clone());
                            let repl = rw.to_simple_replacement(&hugr);
                            Commit::try_new(repl, parents, state_space).ok()
                        })
                },
            )
        }
    }
}

/// A rewriter that combines multiple [`CircuitMatcher`]s before passing the
/// combined match to a [`CircuitReplacer`].
///
/// The [`CircuitMatcher`]s are used to find matches in the circuit. All
/// cartesian products of the matches that are convex are then passed to the
/// [`CircuitReplacer`] to create [`CircuitRewrite`]s.
#[derive(Clone, Debug)]
pub struct CombineMatchReplaceRewriter<C, R> {
    matchers: Vec<C>,
    replacer: R,
}

impl<C, R> CombineMatchReplaceRewriter<C, R> {
    /// Create a new [`MatchReplaceRewriter`].
    pub fn new(matchers: Vec<C>, replacement: R) -> Self {
        Self {
            matchers,
            replacer: replacement,
        }
    }
}

impl<C, R, H: HugrView<Node = hugr::Node>> Rewriter<ResourceScope<H>>
    for CombineMatchReplaceRewriter<C, R>
where
    C: CircuitMatcher,
    C::MatchInfo: Clone,
    R: CircuitReplacer<Vec<C::MatchInfo>>,
{
    type Rewrite = CircuitRewrite;

    fn get_rewrites(&self, circ: &ResourceScope<H>, root_node: hugr::Node) -> Vec<CircuitRewrite> {
        let mut is_first = true;
        let all_matches = self
            .matchers
            .iter()
            .map(|m| {
                if is_first {
                    is_first = false;
                    m.as_hugr_matcher()
                        .get_matches(circ, root_node, &MatchingOptions::default())
                } else {
                    m.as_hugr_matcher()
                        .get_all_matches(circ, &MatchingOptions::default())
                }
            })
            .collect_vec();
        convex_cartesian_product(all_matches, circ)
            .into_iter()
            .flat_map(|(subcirc, match_info)| {
                self.replacer
                    .replace_match(&subcirc, circ, match_info)
                    .into_iter()
                    .filter_map(move |repl| {
                        match CircuitRewrite::try_new(subcirc.clone(), circ, repl) {
                            Ok(ok) => Some(ok),
                            Err(err) => {
                                eprintln!("Error: failed to create rewrite, skipping:\n{}", err);
                                None
                            }
                        }
                    })
            })
            .collect()
    }

    fn get_all_rewrites(&self, circ: &ResourceScope<H>) -> Vec<CircuitRewrite> {
        let all_matches = self
            .matchers
            .iter()
            .map(|m| {
                m.as_hugr_matcher()
                    .get_all_matches(circ, &MatchingOptions::default())
            })
            .collect_vec();
        convex_cartesian_product(all_matches, circ)
            .into_iter()
            .flat_map(|(subcirc, match_info)| {
                self.replacer
                    .replace_match(&subcirc, circ, match_info)
                    .into_iter()
                    .filter_map(move |repl| {
                        match CircuitRewrite::try_new(subcirc.clone(), circ, repl) {
                            Ok(ok) => Some(ok),
                            Err(err) => {
                                eprintln!("Error: failed to create rewrite, skipping:\n{}", err);
                                None
                            }
                        }
                    })
            })
            .collect()
    }
}

fn convex_cartesian_product<N: HugrNode, M: Clone>(
    all_matches: Vec<Vec<(Subcircuit<N>, M)>>,
    circ: &ResourceScope<impl HugrView<Node = N>>,
) -> Vec<(Subcircuit<N>, Vec<M>)> {
    let mut combined_matches = vec![(Subcircuit::new_empty(), vec![])];

    let mut new_combined_matches = Vec::new();
    for matches in all_matches {
        for (subcirc, match_info) in combined_matches {
            // combine with each match in matches
            for (subcirc2, match_info2) in matches.clone() {
                let mut new_subcirc = subcirc.clone();
                let mut new_match_info = match_info.clone();
                new_subcirc.try_extend(subcirc2);

                // Check that the combined subcircuit is convex
                if new_subcirc.validate_subgraph(circ).is_ok() {
                    new_match_info.push(match_info2);
                    new_combined_matches.push((new_subcirc, new_match_info));
                }
            }
        }

        combined_matches = mem::take(&mut new_combined_matches);
    }

    combined_matches
}
