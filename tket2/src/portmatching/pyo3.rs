//! Python bindings for portmatching features

use std::fmt;

use derive_more::{From, Into};
use hugr::hugr::views::sibling_subgraph::PyInvalidReplacementError;
use hugr::{Hugr, IncomingPort, OutgoingPort};
use itertools::Itertools;
use portmatching::{HashMap, PatternID};
use pyo3::{prelude::*, types::PyIterator};
use tket_json_rs::circuit_json::SerialCircuit;

use super::{CircuitPattern, PatternMatch, PatternMatcher};
use crate::circuit::Circuit;
use crate::json::TKETDecode;
use crate::rewrite::CircuitRewrite;

#[pymethods]
impl CircuitPattern {
    /// Construct a pattern from a TKET1 circuit
    #[new]
    pub fn py_from_circuit(circ: PyObject) -> PyResult<CircuitPattern> {
        let circ = pyobj_as_hugr(circ)?;
        let pattern = CircuitPattern::try_from_circuit(&circ)?;
        Ok(pattern)
    }

    /// A string representation of the pattern.
    pub fn __repr__(&self) -> String {
        format!("{:?}", self)
    }
}

#[pymethods]
impl PatternMatcher {
    /// Construct a matcher from a list of patterns.
    #[new]
    pub fn py_from_patterns(patterns: &PyIterator) -> PyResult<Self> {
        Ok(PatternMatcher::from_patterns(
            patterns
                .iter()?
                .map(|p| p?.extract::<CircuitPattern>())
                .collect::<PyResult<Vec<_>>>()?,
        ))
    }
    /// A string representation of the pattern.
    pub fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self))
    }

    /// Find all convex matches in a circuit.
    #[pyo3(name = "find_matches")]
    pub fn py_find_matches(&self, circ: PyObject) -> PyResult<Vec<PyPatternMatch>> {
        let circ = pyobj_as_hugr(circ)?;
        self.find_matches(&circ)
            .into_iter()
            .map(|m| {
                let pattern_id = m.pattern_id();
                PyPatternMatch::try_from_rust(m, &circ, self).map_err(|e| {
                    PyInvalidReplacementError::new_err(format!(
                        "Invalid match for pattern {:?}: {}",
                        pattern_id, e
                    ))
                })
            })
            .collect()
    }
}

/// Python equivalent of [`PatternMatch`].
///
/// A convex pattern match in a circuit, available from Python.
///
/// This object is semantically equivalent to Rust's [`PatternMatch`] but
/// stores data differently, and in particular removes the lifetime-bound
/// references of Rust.
///
/// The data is stored in a way that favours a nice user-facing representation
/// over efficiency. It is provided for convenience and not recommended when
/// performance is a key concern.
///
/// TODO: can this be a wrapper for a [`PatternMatch`] instead?
///
/// [`PatternMatch`]: crate::portmatching::matcher::PatternMatch
#[pyclass]
#[derive(Debug, Clone)]
pub struct PyPatternMatch {
    /// The ID of the pattern in the matcher.
    pub pattern_id: usize,
    /// The root of the pattern within the circuit.
    pub root: Node,
    /// The input ports of the subcircuit.
    ///
    /// This is the incoming boundary of a [`hugr::hugr::views::SiblingSubgraph`].
    /// The input ports are grouped together if they are connected to the same
    /// source.
    pub inputs: Vec<Vec<(Node, IncomingPort)>>,
    /// The output ports of the subcircuit.
    ///
    /// This is the outgoing boundary of a [`hugr::hugr::views::SiblingSubgraph`].
    pub outputs: Vec<(Node, OutgoingPort)>,
    /// The node map from pattern to circuit.
    pub node_map: HashMap<Node, Node>,
}

#[pymethods]
impl PyPatternMatch {
    /// A string representation of the pattern.
    pub fn __repr__(&self) -> String {
        format!("CircuitMatch {:?}", self.node_map)
    }
}

impl PyPatternMatch {
    /// Construct a [`PyPatternMatch`] from a [`PatternMatch`].
    ///
    /// Requires references to the circuit and pattern to resolve indices
    /// into these objects.
    pub fn try_from_rust<C: Circuit + Clone>(
        m: PatternMatch,
        circ: &C,
        matcher: &PatternMatcher,
    ) -> PyResult<Self> {
        let pattern_id = m.pattern_id();
        let pattern = matcher.get_pattern(pattern_id).unwrap();
        let root = Node(m.root);

        let node_map: HashMap<Node, Node> = pattern
            .get_match_map(root.0, circ)
            .ok_or_else(|| PyInvalidReplacementError::new_err("Invalid match"))?
            .into_iter()
            .map(|(p, c)| (Node(p), Node(c)))
            .collect();
        let inputs = pattern
            .inputs
            .iter()
            .map(|ps| {
                ps.iter()
                    .map(|&(n, p)| (node_map[&Node(n)], p.as_incoming().unwrap()))
                    .collect_vec()
            })
            .collect_vec();
        let outputs = pattern
            .outputs
            .iter()
            .map(|&(n, p)| (node_map[&Node(n)], p.as_outgoing().unwrap()))
            .collect_vec();
        Ok(Self {
            pattern_id: pattern_id.0,
            inputs,
            outputs,
            node_map,
            root,
        })
    }

    /// Convert the pattern into a [`CircuitRewrite`].
    pub fn to_rewrite(&self, circ: &Hugr, replacement: Hugr) -> PyResult<CircuitRewrite> {
        let inputs = self
            .inputs
            .iter()
            .map(|p| p.iter().map(|&(n, p)| (n.0, p)).collect())
            .collect();
        let outputs = self.outputs.iter().map(|&(n, p)| (n.0, p)).collect();
        let rewrite = PatternMatch::try_from_io(
            self.root.0,
            PatternID(self.pattern_id),
            circ,
            inputs,
            outputs,
        )
        .expect("Invalid PyCircuitMatch object")
        .to_rewrite(circ, replacement)?;
        Ok(rewrite)
    }
}

/// A [`hugr::Node`] wrapper for Python.
///
/// Note: this will probably be useful outside of portmatching
#[pyclass]
#[derive(From, Into, PartialEq, Eq, Hash, Clone, Copy)]
pub struct Node(hugr::Node);

impl fmt::Debug for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[pymethods]
impl Node {
    /// A string representation of the pattern.
    pub fn __repr__(&self) -> String {
        format!("{:?}", self)
    }
}

fn pyobj_as_hugr(circ: PyObject) -> PyResult<Hugr> {
    let ser_c = SerialCircuit::_from_tket1(circ);
    let hugr: Hugr = ser_c.decode()?;
    Ok(hugr)
}
