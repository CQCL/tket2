//! PyO3 wrapper for rewriters.

use derive_more::From;
use itertools::Itertools;
use pyo3::prelude::*;
use std::path::PathBuf;
use tket2::rewrite::{CircuitRewrite, ECCRewriter, Rewriter, Subcircuit};

use crate::circuit::{PyNode, Tk2Circuit};

/// The module definition
pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "rewrite")?;
    m.add_class::<PyECCRewriter>()?;
    m.add_class::<PyCircuitRewrite>()?;
    m.add_class::<PySubcircuit>()?;
    Ok(m)
}

/// A rewrite rule for circuits.
///
/// Python equivalent of [`CircuitRewrite`].
///
/// [`CircuitRewrite`]: tket2::rewrite::CircuitRewrite
#[pyclass]
#[pyo3(name = "CircuitRewrite")]
#[derive(Debug, Clone, From)]
#[repr(transparent)]
pub struct PyCircuitRewrite {
    /// Rust representation of the circuit chunks.
    pub rewrite: CircuitRewrite,
}

#[pymethods]
impl PyCircuitRewrite {
    /// Number of nodes added or removed by the rewrite.
    ///
    /// The difference between the new number of nodes minus the old. A positive
    /// number is an increase in node count, a negative number is a decrease.
    pub fn node_count_delta(&self) -> isize {
        self.rewrite.node_count_delta()
    }

    /// The replacement subcircuit.
    pub fn replacement(&self) -> Tk2Circuit {
        self.rewrite.replacement().to_owned().into()
    }

    #[new]
    fn try_new(
        source_position: PySubcircuit,
        source_circ: PyRef<Tk2Circuit>,
        replacement: Tk2Circuit,
    ) -> PyResult<Self> {
        Ok(Self {
            rewrite: CircuitRewrite::try_new(
                &source_position.0,
                &source_circ.circ,
                replacement.circ,
            )
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?,
        })
    }
}

/// A subcircuit specification.
///
/// Python equivalent of [`Subcircuit`].
///
/// [`Subcircuit`]: tket2::rewrite::Subcircuit
#[pyclass]
#[pyo3(name = "Subcircuit")]
#[derive(Debug, Clone, From)]
#[repr(transparent)]
pub struct PySubcircuit(Subcircuit);

#[pymethods]
impl PySubcircuit {
    #[new]
    fn from_nodes(nodes: Vec<PyNode>, circ: &Tk2Circuit) -> PyResult<Self> {
        let nodes: Vec<_> = nodes.into_iter().map_into().collect();
        Ok(Self(
            Subcircuit::try_from_nodes(nodes, &circ.circ)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?,
        ))
    }
}

/// A rewriter based on circuit equivalence classes.
///
/// In every equivalence class, one circuit is chosen as the representative.
/// Valid rewrites turn a non-representative circuit into its representative,
/// or a representative circuit into any of the equivalent non-representative
#[pyclass(name = "ECCRewriter")]
pub struct PyECCRewriter(ECCRewriter);

#[pymethods]
impl PyECCRewriter {
    /// Load a precompiled ecc rewriter from a file.
    #[staticmethod]
    pub fn load_precompiled(path: PathBuf) -> PyResult<Self> {
        Ok(Self(ECCRewriter::load_binary(path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string())
        })?))
    }

    /// Returns a list of circuit rewrites that can be applied to the given Tk2Circuit.
    pub fn get_rewrites(&self, circ: &Tk2Circuit) -> Vec<PyCircuitRewrite> {
        self.0
            .get_rewrites(&circ.circ)
            .into_iter()
            .map_into()
            .collect()
    }
}
