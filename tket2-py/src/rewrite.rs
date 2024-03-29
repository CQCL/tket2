//! PyO3 wrapper for rewriters.

use derive_more::From;
use itertools::Itertools;
use pyo3::prelude::*;
use std::path::PathBuf;
use tket2::rewrite::{CircuitRewrite, ECCRewriter, Rewriter};

use crate::circuit::Tk2Circuit;

/// The module definition
pub fn module(py: Python) -> PyResult<&PyModule> {
    let m = PyModule::new(py, "_rewrite")?;
    m.add_class::<PyECCRewriter>()?;
    m.add_class::<PyCircuitRewrite>()?;
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
        self.rewrite.replacement().clone().into()
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
            .get_rewrites(&circ.hugr)
            .into_iter()
            .map_into()
            .collect()
    }
}
