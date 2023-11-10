//! Bindings for circuit rewrites.

use derive_more::From;
use pyo3::prelude::*;
use tket2::rewrite::CircuitRewrite;

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
