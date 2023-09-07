//! Pattern and matcher objects for circuit matching

use std::{
    fmt::Debug,
    fs::File,
    io,
    path::{Path, PathBuf},
};

use super::{CircuitPattern, PEdge, PNode};
use hugr::{
    hugr::views::{
        sibling::{
            ConvexChecker, InvalidReplacement,
            InvalidSubgraph::{self},
        },
        SiblingSubgraph,
    },
    ops::OpType,
    Hugr, Node, Port,
};
use itertools::Itertools;
use portmatching::{
    automaton::{LineBuilder, ScopeAutomaton},
    PatternID,
};
use thiserror::Error;

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

use crate::{
    circuit::Circuit,
    ops::NotT2Op,
    rewrite::{CircuitRewrite, Subcircuit},
    T2Op,
};

/// Matchable operations in a circuit.
///
/// We currently support [`T2Op`] and a the HUGR load constant operation.
#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub(crate) enum MatchOp {
    /// A TKET2 operation.
    Op(T2Op),
    /// A HUGR load constant operation.
    LoadConstant,
}

impl From<T2Op> for MatchOp {
    fn from(op: T2Op) -> Self {
        Self::Op(op)
    }
}

impl TryFrom<OpType> for MatchOp {
    type Error = NotT2Op;

    fn try_from(value: OpType) -> Result<Self, Self::Error> {
        match value {
            OpType::LeafOp(op) => Ok(Self::Op(op.try_into()?)),
            OpType::LoadConstant(_) => Ok(Self::LoadConstant),
            _ => Err(NotT2Op),
        }
    }
}

/// A convex pattern match in a circuit.
#[derive(Clone)]
pub struct CircuitMatch<'a, C> {
    position: Subcircuit<'a, C>,
    pattern: PatternID,
    /// The root of the pattern in the circuit.
    ///
    /// This is redundant with the subgraph attribute, but is a more concise
    /// representation of the match useful for `PyCircuitMatch` or serialisation.
    pub(super) root: Node,
}

impl<'a, C: Circuit<'a>> CircuitMatch<'a, C> {
    /// The matcher's pattern ID of the match.
    pub fn pattern_id(&self) -> PatternID {
        self.pattern
    }

    /// Create a pattern match from the image of a pattern root.
    ///
    /// This checks at construction time that the match is convex. This will
    /// have runtime linear in the size of the circuit.
    ///
    /// For repeated convexity checking on the same circuit, use
    /// [`CircuitMatch::try_from_root_match_with_checker`] instead.
    ///
    /// Returns an error if
    ///  - the match is not convex
    ///  - the subcircuit does not match the pattern
    ///  - the subcircuit is empty
    ///  - the subcircuit obtained is not a valid circuit region
    pub fn try_from_root_match(
        root: Node,
        pattern_id: PatternID,
        pattern: &CircuitPattern,
        circ: &'a C,
    ) -> Result<Self, InvalidCircuitMatch> {
        let mut checker = ConvexChecker::new(circ);
        Self::try_from_root_match_with_checker(root, pattern_id, pattern, circ, &mut checker)
    }

    /// Create a pattern match from the image of a pattern root with a checker.
    ///
    /// This is the same as [`CircuitMatch::try_from_root_match`] but takes a
    /// checker object to speed up convexity checking.
    ///
    /// See [`CircuitMatch::try_from_root_match`] for more details.
    pub fn try_from_root_match_with_checker(
        root: Node,
        pattern_id: PatternID,
        pattern: &CircuitPattern,
        circ: &'a C,
        checker: &mut ConvexChecker<'a, C>,
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
        let subgraph =
            SiblingSubgraph::try_from_boundary_ports_with_checker(circ, inputs, outputs, checker)?;
        Ok(Self {
            position: subgraph.into(),
            pattern: pattern_id,
            root,
        })
    }

    /// Construct a rewrite to replace `self` with `repl`.
    pub fn to_rewrite(&self, repl: Hugr) -> Result<CircuitRewrite, InvalidReplacement> {
        CircuitRewrite::try_new(&self.position, repl)
    }
}

impl<'a, C: Circuit<'a>> Debug for CircuitMatch<'a, C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CircuitMatch")
            .field("root", &self.root)
            .field("nodes", &self.position.subgraph.nodes())
            .finish()
    }
}

/// A matcher object for fast pattern matching on circuits.
///
/// This uses a state automaton internally to match against a set of patterns
/// simultaneously.
#[cfg_attr(feature = "pyo3", pyclass)]
#[derive(Clone, serde::Serialize, serde::Deserialize)]
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
    pub fn find_matches<'a, C: Circuit<'a>>(&self, circuit: &'a C) -> Vec<CircuitMatch<'a, C>> {
        let mut checker = ConvexChecker::new(circuit);
        circuit
            .commands()
            .flat_map(|cmd| self.find_rooted_matches(circuit, cmd.node(), &mut checker))
            .collect()
    }

    /// Find all convex pattern matches in a circuit rooted at a given node.
    fn find_rooted_matches<'a, C: Circuit<'a>>(
        &self,
        circ: &'a C,
        root: Node,
        checker: &mut ConvexChecker<'a, C>,
    ) -> Vec<CircuitMatch<'a, C>> {
        self.automaton
            .run(
                root,
                // Node weights (none)
                validate_weighted_node(circ),
                // Check edge exist
                validate_unweighted_edge(circ),
            )
            .filter_map(|pattern_id| {
                let pattern = &self.patterns[pattern_id.0];
                handle_match_error(
                    CircuitMatch::try_from_root_match_with_checker(
                        root, pattern_id, pattern, circ, checker,
                    ),
                    root,
                )
            })
            .collect()
    }

    /// Get a pattern by ID.
    pub fn get_pattern(&self, id: PatternID) -> Option<&CircuitPattern> {
        self.patterns.get(id.0)
    }

    /// Serialise a matcher into an IO stream.
    ///
    /// Precomputed matchers can be serialised as binary and then loaded
    /// later using [`CircuitMatcher::load_binary_io`].
    pub fn save_binary_io<W: io::Write>(
        &self,
        writer: &mut W,
    ) -> Result<(), MatcherSerialisationError> {
        rmp_serde::encode::write(writer, &self)?;
        Ok(())
    }

    /// Loads a matcher from an IO stream.
    ///
    /// Loads streams as created by [`CircuitMatcher::save_binary_io`].
    pub fn load_binary_io<R: io::Read>(reader: &mut R) -> Result<Self, MatcherSerialisationError> {
        let matcher: Self = rmp_serde::decode::from_read(reader)?;
        Ok(matcher)
    }

    /// Save a matcher as a binary file.
    ///
    /// Precomputed matchers can be saved as binary files and then loaded
    /// later using [`CircuitMatcher::load_binary`].
    ///
    /// The extension of the file name will always be set or amended to be
    /// `.bin`.
    ///
    /// If successful, returns the path to the newly created file.
    pub fn save_binary(
        &self,
        name: impl AsRef<Path>,
    ) -> Result<PathBuf, MatcherSerialisationError> {
        let mut file_name = PathBuf::from(name.as_ref());
        file_name.set_extension("bin");
        let mut file = File::create(&file_name)?;
        self.save_binary_io(&mut file)?;
        Ok(file_name)
    }

    /// Loads a matcher saved using [`CircuitMatcher::save_binary`].
    pub fn load_binary(name: impl AsRef<Path>) -> Result<Self, MatcherSerialisationError> {
        let file = File::open(name)?;
        let mut reader = std::io::BufReader::new(file);
        Self::load_binary_io(&mut reader)
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

/// Errors that can occur when (de)serialising a matcher.
#[derive(Debug, Error)]
pub enum MatcherSerialisationError {
    /// An IO error occured
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
    /// An error occured during deserialisation
    #[error("Deserialisation error: {0}")]
    Deserialisation(#[from] rmp_serde::decode::Error),
    /// An error occured during serialisation
    #[error("Serialisation error: {0}")]
    Serialisation(#[from] rmp_serde::encode::Error),
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
    use std::sync::OnceLock;

    use hugr::hugr::views::{DescendantsGraph, HierarchyView};
    use hugr::ops::handle::DfgID;
    use hugr::{Hugr, HugrView};
    use itertools::Itertools;

    use crate::utils::build_simple_circuit;
    use crate::T2Op;

    use super::{CircuitMatcher, CircuitPattern};

    static H_CX: OnceLock<Hugr> = OnceLock::new();
    static CX_CX: OnceLock<Hugr> = OnceLock::new();

    fn h_cx<'a>() -> DescendantsGraph<'a, DfgID> {
        let circ = H_CX.get_or_init(|| {
            build_simple_circuit(2, |circ| {
                circ.append(T2Op::CX, [0, 1]).unwrap();
                circ.append(T2Op::H, [0]).unwrap();
                Ok(())
            })
            .unwrap()
        });
        DescendantsGraph::new(circ, circ.root())
    }

    fn cx_xc<'a>() -> DescendantsGraph<'a, DfgID> {
        let circ = CX_CX.get_or_init(|| {
            build_simple_circuit(2, |circ| {
                circ.append(T2Op::CX, [0, 1]).unwrap();
                circ.append(T2Op::CX, [1, 0]).unwrap();
                Ok(())
            })
            .unwrap()
        });
        DescendantsGraph::new(circ, circ.root())
    }

    #[test]
    fn construct_matcher() {
        let circ = h_cx();

        let p = CircuitPattern::try_from_circuit(&circ).unwrap();
        let m = CircuitMatcher::from_patterns(vec![p]);

        let matches = m.find_matches(&circ);
        assert_eq!(matches.len(), 1);
    }

    #[test]
    fn serialise_round_trip() {
        let circs = [h_cx(), cx_xc()];
        let patterns = circs
            .iter()
            .map(|circ| CircuitPattern::try_from_circuit(circ).unwrap())
            .collect_vec();

        // Estimate the size of the buffer based on the number of patterns and the size of each pattern
        let mut buf = Vec::with_capacity(patterns[0].n_edges() + patterns[1].n_edges());
        let m = CircuitMatcher::from_patterns(patterns);
        m.save_binary_io(&mut buf).unwrap();

        let m2 = CircuitMatcher::load_binary_io(&mut buf.as_slice()).unwrap();
        let mut buf2 = Vec::with_capacity(buf.len());
        m2.save_binary_io(&mut buf2).unwrap();

        assert_eq!(buf, buf2);
    }
}
