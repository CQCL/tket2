//! Pattern and matcher objects for circuit matching

use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};

use delegate::delegate;
// use super::{CircuitPattern, NodeID, PEdge, PNode};
use derive_more::{Debug, Display, Error, From, Into};
use hugr::hugr::views::sibling_subgraph::TopoConvexChecker;
use hugr::hugr::views::ExtractHugr;
use hugr::ops::{NamedOp, OpType};
use hugr::HugrView;
use portmatching::{self as pm, PortMatcher};
use smol_str::SmolStr;

use crate::rewrite::{CircuitRewrite, InvalidReplacement};
use crate::Circuit;
use crate::{rewrite::Subcircuit, Tk2Op};

use super::constraint::Predicate;
use super::indexing::FlatHugrIndexingScheme;
use super::{CircuitPattern, HugrVariableID};

pub use pm::PatternID;

/// A match of a pattern in a circuit.
#[derive(Debug, Clone)]
pub struct PatternMatch {
    /// The matched pattern ID.
    pub pattern: PatternID,
    /// The region of the circuit that matched the pattern.
    pub subcircuit: Subcircuit,
}

impl PatternMatch {
    /// Construct a rewrite from a matching pattern.
    pub fn to_rewrite(
        &self,
        circuit: &Circuit<impl HugrView>,
        replacement: Circuit<impl ExtractHugr>,
    ) -> Result<CircuitRewrite, InvalidReplacement> {
        CircuitRewrite::try_new(&self.subcircuit, circuit, replacement)
    }
}

/// Matchable operations in a circuit.
#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct MatchOp {
    /// The operation identifier
    op_name: SmolStr,
    /// The encoded operation, if necessary for comparisons.
    ///
    /// This as a temporary hack for comparing parametric operations, since
    /// OpType doesn't implement Eq, Hash, or Ord.
    #[debug(skip)]
    encoded: Option<Vec<u8>>,
}

impl From<OpType> for MatchOp {
    fn from(op: OpType) -> Self {
        Self::from(&op)
    }
}

impl From<&OpType> for MatchOp {
    fn from(op: &OpType) -> Self {
        let op_name = op.name();
        let encoded = encode_op(op);
        Self { op_name, encoded }
    }
}

impl From<Tk2Op> for MatchOp {
    fn from(value: Tk2Op) -> Self {
        let op: OpType = value.into();
        op.into()
    }
}

/// Encode a unique identifier for an operation.
///
/// Avoids encoding some data if we know the operation can be uniquely
/// identified by their name.
fn encode_op(op: &OpType) -> Option<Vec<u8>> {
    match op {
        OpType::Module(_) => None,
        OpType::ExtensionOp(op) => encode_op(&OpType::OpaqueOp(op.make_opaque())),
        OpType::OpaqueOp(op) => {
            let mut encoded: Vec<u8> = Vec::new();
            // Ignore irrelevant fields
            rmp_serde::encode::write(&mut encoded, op.extension()).ok()?;
            rmp_serde::encode::write(&mut encoded, &op.name()).ok()?;
            rmp_serde::encode::write(&mut encoded, op.args()).ok()?;
            Some(encoded)
        }
        _ => rmp_serde::encode::to_vec(&op).ok(),
    }
}

/// A matcher object for fast pattern matching on circuits.
///
/// This uses a state automaton internally to match against a set of patterns
/// simultaneously.
#[derive(Debug, Clone, From, Into, serde::Serialize, serde::Deserialize)]
pub struct PatternMatcher(
    pm::ManyMatcher<CircuitPattern, HugrVariableID, Predicate, FlatHugrIndexingScheme>,
);

impl PatternMatcher {
    /// Construct a matcher from a set of patterns
    pub fn from_patterns(patterns: Vec<CircuitPattern>) -> Self {
        pm::ManyMatcher::try_from_patterns(patterns, Default::default())
            .expect("CircuitPattern conversions dont fail")
            .into()
    }

    /// Find all pattern matches in a circuit.
    pub fn find_matches<'a>(
        &'a self,
        circuit: &'a Circuit<impl HugrView>,
    ) -> impl Iterator<Item = PatternMatch> + 'a {
        let checker = TopoConvexChecker::new(circuit.hugr());
        self.0.find_matches(circuit).filter_map(move |m| {
            let pattern = self
                .get_pattern(m.pattern)
                .expect("invalid pattern ID in match");
            let subcircuit = pattern
                .get_subcircuit_with_checker(&m.match_data, circuit, &checker)
                .ok()?;
            Some(PatternMatch {
                pattern: m.pattern,
                subcircuit,
            })
        })
    }

    delegate! {
        to self.0 {
            /// Get a pattern by its ID.
            pub fn get_pattern(&self, id: PatternID) -> Option<&CircuitPattern>;

            /// Get the number of states in the automaton.
            pub fn n_states(&self) -> usize;

            /// Get the number of patterns in the matcher.
            pub fn n_patterns(&self) -> usize;

            /// Get a dot string representation of the matcher.
            pub fn dot_string(&self) -> String;
        }
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
}

/// Errors that can occur when (de)serialising a matcher.
#[derive(Debug, Display, Error, From)]
#[non_exhaustive]
pub enum MatcherSerialisationError {
    /// An IO error occurred
    #[display("IO error: {_0}")]
    Io(io::Error),
    /// An error occurred during deserialisation
    #[display("Deserialisation error: {_0}")]
    Deserialisation(rmp_serde::decode::Error),
    /// An error occurred during serialisation
    #[display("Serialisation error: {_0}")]
    Serialisation(rmp_serde::encode::Error),
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use hugr::builder::{DFGBuilder, Dataflow, DataflowHugr};
    use hugr::extension::prelude::QB_T;
    use hugr::types::Signature;
    use hugr::{IncomingPort, Node, OutgoingPort};
    use itertools::Itertools;
    use portgraph::NodeIndex;
    use rstest::{fixture, rstest};

    use crate::extension::rotation::ROTATION_TYPE;
    use crate::extension::REGISTRY;
    use crate::rewrite::{ECCRewriter, Rewriter};
    use crate::utils::build_simple_circuit;
    use crate::{Circuit, Tk2Op};

    use super::{CircuitPattern, PatternMatcher};

    fn h_cx() -> Circuit {
        build_simple_circuit(2, |circ| {
            circ.append(Tk2Op::CX, [0, 1]).unwrap();
            circ.append(Tk2Op::H, [0]).unwrap();
            Ok(())
        })
        .unwrap()
    }

    fn cx_xc() -> Circuit {
        build_simple_circuit(2, |circ| {
            circ.append(Tk2Op::CX, [0, 1]).unwrap();
            circ.append(Tk2Op::CX, [1, 0]).unwrap();
            Ok(())
        })
        .unwrap()
    }

    #[fixture]
    fn cx_cx_3() -> Circuit {
        build_simple_circuit(3, |circ| {
            circ.append(Tk2Op::CX, [0, 1]).unwrap();
            circ.append(Tk2Op::CX, [2, 1]).unwrap();
            Ok(())
        })
        .unwrap()
    }

    #[fixture]
    fn cx_cx() -> Circuit {
        build_simple_circuit(2, |circ| {
            circ.append(Tk2Op::CX, [0, 1]).unwrap();
            circ.append(Tk2Op::CX, [0, 1]).unwrap();
            Ok(())
        })
        .unwrap()
    }

    #[test]
    fn construct_matcher() {
        let circ = h_cx();

        let p = CircuitPattern::try_from_circuit(&circ).unwrap();
        let m = PatternMatcher::from_patterns(vec![p]);

        let matches = m.find_matches(&circ);
        assert_eq!(matches.count(), 1);
    }

    #[test]
    fn serialise_round_trip() {
        let circs = [h_cx(), cx_xc()];
        let patterns = circs
            .iter()
            .map(|circ| CircuitPattern::try_from_circuit(circ).unwrap())
            .collect_vec();

        // Estimate the size of the buffer based on the number of patterns and the size of each pattern
        let mut buf = Vec::with_capacity(patterns.iter().map(|p| p.n_constraints()).sum());
        let m = PatternMatcher::from_patterns(patterns);
        m.save_binary_io(&mut buf).unwrap();

        let m2 = PatternMatcher::load_binary_io(&mut buf.as_slice()).unwrap();
        let mut buf2 = Vec::with_capacity(buf.len());
        m2.save_binary_io(&mut buf2).unwrap();

        assert_eq!(buf, buf2);
    }

    #[rstest]
    fn cx_cx_replace_to_id(cx_cx: Circuit, cx_cx_3: Circuit) {
        let p = CircuitPattern::try_from_circuit(&cx_cx_3).unwrap();
        let m = PatternMatcher::from_patterns(vec![p]);

        let matches = m.find_matches(&cx_cx);
        assert_eq!(matches.count(), 0);
    }

    #[fixture]
    fn cx_rz() -> Circuit {
        let input_t = vec![QB_T, QB_T, ROTATION_TYPE];
        let output_t = vec![QB_T, QB_T];
        let mut h = DFGBuilder::new(Signature::new(input_t, output_t)).unwrap();

        let (qb1, qb2, f) = h.input_wires().collect_tuple().unwrap();

        let res = h.add_dataflow_op(Tk2Op::CX, [qb1, qb2]).unwrap();
        let (qb1, qb2) = res.outputs().collect_tuple().unwrap();
        let res = h.add_dataflow_op(Tk2Op::Rz, [qb2, f]).unwrap();
        let qb2 = res.outputs().next().unwrap();

        h.finish_hugr_with_outputs([qb1, qb2], &REGISTRY)
            .unwrap()
            .into()
    }

    #[rstest]
    fn cx_rz_replace_to_id(cx_rz: Circuit) {
        // let p = CircuitPattern::try_from_circuit(&cx_rz).unwrap();
        // let m = PatternMatcher::from_patterns(vec![p]);
        let rewriter =
            ECCRewriter::try_from_eccs_json_file(Path::new("../test_eccs.json")).unwrap();
        // println!("{}", rewriter.dot_string());
        // println!("{}", m.dot_string());

        // let matches = m.find_matches(&cx_rz);
        // assert_eq!(matches.count(), 1);

        // panic!();

        let matches = rewriter.get_rewrites(&cx_rz);
        assert_eq!(matches.len(), 1);
    }

    #[rstest]
    fn cx_match_xc() {
        let cx = build_simple_circuit(2, |circ| {
            circ.append(Tk2Op::CX, [0, 1]).unwrap();
            circ.append(Tk2Op::H, [1]).unwrap();
            Ok(())
        })
        .unwrap();
        let xc = build_simple_circuit(2, |circ| {
            circ.append(Tk2Op::CX, [1, 0]).unwrap();
            circ.append(Tk2Op::H, [0]).unwrap();
            Ok(())
        })
        .unwrap();

        let p = CircuitPattern::try_from_circuit(&cx).unwrap();
        let matcher = PatternMatcher::from_patterns(vec![p]);

        let m = matcher.find_matches(&xc).exactly_one().ok().unwrap();

        assert_eq!(
            m.subcircuit.subgraph.incoming_ports(),
            &[
                [(Node::from(NodeIndex::new(3)), IncomingPort::from(0)),],
                [(Node::from(NodeIndex::new(3)), IncomingPort::from(1)),]
            ]
        );
        assert_eq!(
            m.subcircuit.subgraph.outgoing_ports(),
            &[
                (Node::from(NodeIndex::new(3)), OutgoingPort::from(0)),
                (Node::from(NodeIndex::new(4)), OutgoingPort::from(0)),
            ]
        );
    }
}
