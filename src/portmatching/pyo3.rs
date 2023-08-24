//! Python bindings for portmatching features

use std::{collections::HashMap, fmt};

use derive_more::{From, Into};
use hugr::hugr::views::{DescendantsGraph, HierarchyView};
use hugr::{Hugr, HugrView};
use portmatching::PortMatcher;
use pyo3::{create_exception, exceptions::PyException, prelude::*, types::PyIterator};
use tket_json_rs::circuit_json::SerialCircuit;

use super::{CircuitMatcher, CircuitPattern};
use crate::json::TKETDecode;

create_exception!(pyrs, PyValidateError, PyException);

#[pymethods]
impl CircuitPattern {
    /// Construct a pattern from a TKET1 circuit
    #[new]
    pub fn py_from_circuit(circ: PyObject) -> PyResult<CircuitPattern> {
        let ser_c = SerialCircuit::_from_tket1(circ);
        let hugr: Hugr = ser_c
            .decode()
            .map_err(|e| PyValidateError::new_err(e.to_string()))?;
        let circ = DescendantsGraph::new(&hugr, hugr.root());
        Ok(CircuitPattern::from_circuit(&circ))
    }

    /// A string representation of the pattern.
    pub fn __repr__(&self) -> String {
        format!("{:?}", self)
    }
}

#[pymethods]
impl CircuitMatcher {
    /// Construct a matcher from a list of patterns.
    #[new]
    pub fn py_from_patterns(patterns: &PyIterator) -> PyResult<Self> {
        Ok(CircuitMatcher::from_patterns(
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

    /// Find all matches in a circuit
    #[pyo3(name = "find_matches")]
    pub fn py_find_matches(&self, circ: PyObject) -> PyResult<Vec<HashMap<Node, Node>>> {
        let ser_c = SerialCircuit::_from_tket1(circ);
        let hugr: Hugr = ser_c
            .decode()
            .map_err(|e| PyValidateError::new_err(e.to_string()))?;
        let circ = DescendantsGraph::new(&hugr, hugr.root());
        let matches = self.find_matches(&circ);
        Ok(matches
            .into_iter()
            .map(|m| {
                self.get_match_map(m, &circ)
                    .unwrap()
                    .into_iter()
                    .map(|(n, m)| (n.into(), m.into()))
                    .collect()
            })
            .collect())
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
