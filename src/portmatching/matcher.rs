//! Pattern and matcher objects for circuit matching

use std::fmt::Debug;

use super::{CircuitPattern, MatchOp, PEdge, PNode};
use derive_more::{From, Into};
use hugr::{
    hugr::views::{
        sibling::{
            InvalidReplacement,
            InvalidSubgraph::{self},
        },
        SiblingSubgraph,
    },
    Hugr, Node, Port, SimpleReplacement,
};
use itertools::Itertools;
use portmatching::{
    automaton::{LineBuilder, ScopeAutomaton},
    PatternID,
};
use thiserror::Error;

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

use crate::circuit::Circuit;

/// A convex pattern match in a circuit.
#[derive(Clone)]
pub struct CircuitMatch<'a, 'p, C> {
    subgraph: SiblingSubgraph<'a, C>,
    pub(super) pattern: &'p CircuitPattern,
    /// The root of the pattern in the circuit.
    ///
    /// This is redundant with the subgraph attribute, but is a more concise
    /// representation of the match useful for `PyCircuitMatch` or serialisation.
    pub(super) root: Node,
}

impl<'a, 'p, C: Circuit<'a>> CircuitMatch<'a, 'p, C> {
    /// Create a pattern match from the image of a pattern root.
    ///
    /// This checks at construction time that the match is convex. This will
    /// have runtime linear in the size of the circuit.
    ///
    /// TODO: Support passing a pre-computed [`portgraph::algorithms::ConvexChecker`]
    /// at construction time for faster convexity checking.
    ///
    /// Returns an error if
    ///  - the match is not convex
    ///  - the subcircuit does not match the pattern
    ///  - the subcircuit is empty
    ///  - the subcircuit obtained is not a valid circuit region
    pub fn try_from_root_match(
        root: Node,
        pattern: &'p CircuitPattern,
        circ: &'a C,
    ) -> Result<Self, InvalidCircuitMatch> {
        let map = pattern
            .get_match_map(root, circ)
            .ok_or(InvalidCircuitMatch::MatchNotFound)?;
        let inputs = pattern
            .inputs
            .iter()
            .map(|p| p.iter().map(|(n, p)| (map[n], *p)).collect_vec())
            .collect_vec();
        let outputs = pattern
            .outputs
            .iter()
            .map(|(n, p)| (map[n], *p))
            .collect_vec();
        let subgraph = SiblingSubgraph::try_from_boundary_ports(circ, inputs, outputs)?;
        Ok(Self {
            subgraph,
            pattern,
            root,
        })
    }

    /// Construct a rewrite to replace `self` with `repl`.
    pub fn to_rewrite(&self, repl: Hugr) -> Result<CircuitRewrite, InvalidReplacement> {
        self.subgraph
            .create_simple_replacement(repl)
            .map(Into::into)
    }
}

impl<'a, 'p, C: Circuit<'a>> Debug for CircuitMatch<'a, 'p, C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CircuitMatch")
            .field("root", &self.root)
            .field("nodes", &self.subgraph.nodes())
            .finish()
    }
}

/// A rewrite object for circuit matching.
#[cfg_attr(feature = "pyo3", pyclass)]
#[derive(Debug, Clone, From, Into)]
pub struct CircuitRewrite(SimpleReplacement);

/// A matcher object for fast pattern matching on circuits.
///
/// This uses a state automaton internally to match against a set of patterns
/// simultaneously.
#[cfg_attr(feature = "pyo3", pyclass)]
pub struct CircuitMatcher {
    automaton: ScopeAutomaton<PNode, PEdge, Port>,
    patterns: Vec<CircuitPattern>,
}

impl Debug for CircuitMatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CircuitMatcher")
            .field("patterns", &self.patterns)
            .finish()
    }
}

impl CircuitMatcher {
    /// Construct a matcher from a set of patterns
    pub fn from_patterns(patterns: impl Into<Vec<CircuitPattern>>) -> Self {
        let patterns = patterns.into();
        let line_patterns = patterns
            .iter()
            .map(|p| {
                p.pattern
                    .clone()
                    .try_into_line_pattern(compatible_offsets)
                    .expect("Failed to express pattern as line pattern")
            })
            .collect_vec();
        let builder = LineBuilder::from_patterns(line_patterns);
        let automaton = builder.build();
        Self {
            automaton,
            patterns,
        }
    }

    /// Find all convex pattern matches in a circuit.
    pub fn find_matches<'a, 'm, C: Circuit<'a>>(
        &'m self,
        circuit: &'a C,
    ) -> Vec<CircuitMatch<'a, 'm, C>> {
        circuit
            .commands()
            .flat_map(|cmd| self.find_rooted_matches(circuit, cmd.node))
            .collect()
    }

    /// Find all convex pattern matches in a circuit rooted at a given node.
    pub fn find_rooted_matches<'a, 'm, C: Circuit<'a>>(
        &'m self,
        circ: &'a C,
        root: Node,
    ) -> Vec<CircuitMatch<'a, 'm, C>> {
        self.automaton
            .run(
                root,
                // Node weights (none)
                validate_weighted_node(circ),
                // Check edge exist
                validate_unweighted_edge(circ),
            )
            .filter_map(|m| {
                let p = &self.patterns[m.0];
                handle_match_error(CircuitMatch::try_from_root_match(root, p, circ), root)
            })
            .collect()
    }

    /// Get a pattern by ID.
    pub fn get_pattern(&self, id: PatternID) -> Option<&CircuitPattern> {
        self.patterns.get(id.0)
    }
}

/// Errors that can occur when constructing matches.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum InvalidCircuitMatch {
    /// The match is not convex.
    #[error("match is not convex")]
    NotConvex,
    /// The subcircuit does not match the pattern.
    #[error("invalid circuit region")]
    MatchNotFound,
    /// The subcircuit matched is not valid.
    ///
    /// This is typically a logic error in the code.
    #[error("invalid circuit region")]
    InvalidSubcircuit,
    /// Empty matches are not supported.
    ///
    /// This should never happen is the pattern is not itself empty (in which
    /// case an error would have been raised earlier on).
    #[error("empty match")]
    EmptyMatch,
}

impl From<InvalidSubgraph> for InvalidCircuitMatch {
    fn from(value: InvalidSubgraph) -> Self {
        match value {
            InvalidSubgraph::NotConvex => InvalidCircuitMatch::NotConvex,
            InvalidSubgraph::EmptySubgraph => InvalidCircuitMatch::EmptyMatch,
            InvalidSubgraph::NoSharedParent | InvalidSubgraph::InvalidBoundary => {
                InvalidCircuitMatch::InvalidSubcircuit
            }
        }
    }
}

fn compatible_offsets((_, pout): &(Port, Port), (pin, _): &(Port, Port)) -> bool {
    pout.direction() != pin.direction() && pout.index() == pin.index()
}

/// Check if an edge `e` is valid in a portgraph `g` without weights.
pub(crate) fn validate_unweighted_edge<'circ>(
    circ: &impl Circuit<'circ>,
) -> impl for<'a> Fn(Node, &'a PEdge) -> Option<Node> + '_ {
    move |src, &(src_port, tgt_port)| {
        let (next_node, _) = circ
            .linked_ports(src, src_port)
            .find(|&(_, tgt)| tgt == tgt_port)?;
        Some(next_node)
    }
}

/// Check if a node `n` is valid in a weighted portgraph `g`.
pub(crate) fn validate_weighted_node<'circ>(
    circ: &impl Circuit<'circ>,
) -> impl for<'a> Fn(Node, &PNode) -> bool + '_ {
    move |v, prop| {
        let v_weight = MatchOp::try_from(circ.get_optype(v).clone());
        v_weight.is_ok_and(|w| &w == prop)
    }
}

/// Unwraps match errors, ignoring benign errors and panicking otherwise.
///
/// Benign errors are non-convex matches, which are expected to occur.
/// Other errors are considered logic errors and should never occur.
fn handle_match_error<T>(match_res: Result<T, InvalidCircuitMatch>, root: Node) -> Option<T> {
    match_res
        .map_err(|err| match err {
            InvalidCircuitMatch::NotConvex => InvalidCircuitMatch::NotConvex,
            InvalidCircuitMatch::MatchNotFound
            | InvalidCircuitMatch::InvalidSubcircuit
            | InvalidCircuitMatch::EmptyMatch => {
                panic!("invalid match at root node {root:?}")
            }
        })
        .ok()
}

#[cfg(test)]
mod tests {
    use hugr::extension::prelude::QB_T;
    use hugr::hugr::views::{DescendantsGraph, HierarchyView};
    use hugr::ops::handle::DfgID;
    use hugr::types::FunctionType;
    use hugr::{
        builder::{DFGBuilder, Dataflow, DataflowHugr},
        Hugr, HugrView,
    };

    use crate::utils::{cx_gate, h_gate};

    use super::{CircuitMatcher, CircuitPattern};

    fn h_cx() -> Hugr {
        let qb = QB_T;
        let mut hugr = DFGBuilder::new(FunctionType::new_linear(vec![qb; 2])).unwrap();
        let mut circ = hugr.as_circuit(hugr.input_wires().collect());
        circ.append(cx_gate(), [0, 1]).unwrap();
        circ.append(h_gate(), [0]).unwrap();
        let out_wires = circ.finish();
        hugr.finish_hugr_with_outputs(out_wires).unwrap()
    }

    #[test]
    fn construct_matcher() {
        let hugr = h_cx();
        let circ: DescendantsGraph<'_, DfgID> = DescendantsGraph::new(&hugr, hugr.root());

        let p = CircuitPattern::from_circuit(&circ);
        let m = CircuitMatcher::from_patterns(vec![p]);

        let matches = m.find_matches(&circ);
        assert_eq!(matches.len(), 1);
    }
}
