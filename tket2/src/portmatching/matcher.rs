//! Pattern and matcher objects for circuit matching

use std::{
    fmt::Debug,
    fs::File,
    io,
    path::{Path, PathBuf},
};

use super::{
    indexing::{PatternOpPosition, StaticIndexScheme},
    pattern::InvalidStaticPattern,
    predicate::Predicate,
};
use delegate::delegate;
use hugr::hugr::views::sibling_subgraph::{InvalidSubgraph, InvalidSubgraphBoundary};
use portmatching::{IndexingScheme, ManyMatcher, PatternFallback, PatternID, PortMatcher};
use thiserror::Error;

use crate::static_circ::StaticSizeCircuit;

/// A matcher object for fast pattern matching on circuits.
///
/// This uses a state automaton internally to match against a set of patterns
/// simultaneously.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CircuitMatcher(
    ManyMatcher<StaticSizeCircuit, PatternOpPosition, Predicate, StaticIndexScheme>,
);

impl PortMatcher<StaticSizeCircuit> for CircuitMatcher {
    type Match = <StaticIndexScheme as IndexingScheme<StaticSizeCircuit>>::Map;

    fn find_matches<'a>(
        &'a self,
        host: &'a StaticSizeCircuit,
    ) -> impl Iterator<Item = portmatching::PatternMatch<Self::Match>> + 'a {
        self.0.find_matches(host)
    }
}

impl CircuitMatcher {
    /// Create a matcher from a set of patterns, skipping disconnected patterns.
    pub fn try_from_patterns(
        patterns: Vec<StaticSizeCircuit>,
    ) -> Result<Self, InvalidStaticPattern> {
        Ok(CircuitMatcher(ManyMatcher::try_from_patterns(
            patterns,
            PatternFallback::Skip,
        )?))
    }

    /// Serialise a matcher into an IO stream.
    ///
    /// Precomputed matchers can be serialised as binary and then loaded
    /// later using [`PatternMatcher::load_binary_io`].
    pub fn save_binary_io<W: io::Write>(
        &self,
        writer: &mut W,
    ) -> Result<(), MatcherSerialisationError> {
        rmp_serde::encode::write(writer, &self)?;
        Ok(())
    }

    /// Loads a matcher from an IO stream.
    ///
    /// Loads streams as created by [`PatternMatcher::save_binary_io`].
    pub fn load_binary_io<R: io::Read>(reader: &mut R) -> Result<Self, MatcherSerialisationError> {
        let matcher: Self = rmp_serde::decode::from_read(reader)?;
        Ok(matcher)
    }

    /// Save a matcher as a binary file.
    ///
    /// Precomputed matchers can be saved as binary files and then loaded
    /// later using [`PatternMatcher::load_binary`].
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

    /// Loads a matcher saved using [`PatternMatcher::save_binary`].
    pub fn load_binary(name: impl AsRef<Path>) -> Result<Self, MatcherSerialisationError> {
        let file = File::open(name)?;
        let mut reader = std::io::BufReader::new(file);
        Self::load_binary_io(&mut reader)
    }

    delegate! {
        to self.0 {
            pub fn get_pattern(&self, id: PatternID) -> Option<&StaticSizeCircuit>;
            pub fn n_states(&self) -> usize;
            pub fn dot_string(&self) -> String;
            pub fn n_patterns(&self) -> usize;
        }
    }
}

/// Errors that can occur when constructing matches.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum InvalidPatternMatch {
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
    #[error(transparent)]
    #[allow(missing_docs)]
    Other(InvalidSubgraph),
}

/// Errors that can occur when (de)serialising a matcher.
#[derive(Debug, Error)]
pub enum MatcherSerialisationError {
    /// An IO error occurred
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
    /// An error occurred during deserialisation
    #[error("Deserialisation error: {0}")]
    Deserialisation(#[from] rmp_serde::decode::Error),
    /// An error occurred during serialisation
    #[error("Serialisation error: {0}")]
    Serialisation(#[from] rmp_serde::encode::Error),
}

impl From<InvalidSubgraph> for InvalidPatternMatch {
    fn from(value: InvalidSubgraph) -> Self {
        match value {
            // A non-convex subgraph might show itself as a disconnected boundary
            // in the subgraph
            InvalidSubgraph::NotConvex
            | InvalidSubgraph::InvalidBoundary(
                InvalidSubgraphBoundary::DisconnectedBoundaryPort(_, _),
            ) => InvalidPatternMatch::NotConvex,
            InvalidSubgraph::EmptySubgraph => InvalidPatternMatch::EmptyMatch,
            InvalidSubgraph::NoSharedParent { .. } | InvalidSubgraph::InvalidBoundary(_) => {
                InvalidPatternMatch::InvalidSubcircuit
            }
            other => InvalidPatternMatch::Other(other),
        }
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use portmatching::PortMatcher;
    use rstest::{fixture, rstest};

    use crate::static_circ::StaticSizeCircuit;
    use crate::utils::build_simple_circuit;
    use crate::{Circuit, Tk2Op};

    use super::CircuitMatcher;

    fn h_cx() -> StaticSizeCircuit {
        let circ = build_simple_circuit(2, |circ| {
            circ.append(Tk2Op::CX, [0, 1]).unwrap();
            circ.append(Tk2Op::H, [0]).unwrap();
            Ok(())
        })
        .unwrap();
        StaticSizeCircuit::try_from(&circ).unwrap()
    }

    fn cx_xc() -> StaticSizeCircuit {
        let circ = build_simple_circuit(2, |circ| {
            circ.append(Tk2Op::CX, [0, 1]).unwrap();
            circ.append(Tk2Op::CX, [1, 0]).unwrap();
            Ok(())
        })
        .unwrap();
        StaticSizeCircuit::try_from(&circ).unwrap()
    }

    #[fixture]
    fn cx_cx_3() -> StaticSizeCircuit {
        let circ = build_simple_circuit(3, |circ| {
            circ.append(Tk2Op::CX, [0, 1]).unwrap();
            circ.append(Tk2Op::CX, [2, 1]).unwrap();
            Ok(())
        })
        .unwrap();
        StaticSizeCircuit::try_from(&circ).unwrap()
    }

    #[fixture]
    fn cx_cx() -> StaticSizeCircuit {
        let circ = build_simple_circuit(2, |circ| {
            circ.append(Tk2Op::CX, [0, 1]).unwrap();
            circ.append(Tk2Op::CX, [0, 1]).unwrap();
            Ok(())
        })
        .unwrap();
        StaticSizeCircuit::try_from(&circ).unwrap()
    }

    #[test]
    fn construct_matcher() {
        let circ = h_cx();

        let m = CircuitMatcher::try_from_patterns(vec![circ.clone()]).unwrap();

        let matches = m.find_matches(&circ);
        assert_eq!(matches.count(), 1);
    }

    #[test]
    fn serialise_round_trip() {
        let circs = [h_cx(), cx_xc()];
        let patterns = circs.to_vec();

        // Estimate the size of the buffer based on the number of patterns and the size of each pattern
        let mut buf = Vec::with_capacity(patterns[0].n_ops() + patterns[1].n_ops());
        let m = CircuitMatcher::try_from_patterns(patterns).unwrap();
        m.save_binary_io(&mut buf).unwrap();

        let m2 = CircuitMatcher::load_binary_io(&mut buf.as_slice()).unwrap();
        let mut buf2 = Vec::with_capacity(buf.len());
        m2.save_binary_io(&mut buf2).unwrap();

        assert_eq!(buf, buf2);
    }

    #[rstest]
    fn cx_cx_replace_to_id(cx_cx: StaticSizeCircuit, cx_cx_3: StaticSizeCircuit) {
        let m = CircuitMatcher::try_from_patterns(vec![cx_cx_3]).unwrap();

        let matches = m.find_matches(&cx_cx);
        assert_eq!(matches.count(), 0);
    }
}
