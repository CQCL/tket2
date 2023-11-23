//! Circuit chunking utilities.

use derive_more::From;
use pyo3::exceptions::PyAttributeError;
use pyo3::prelude::*;
use tket2::passes::CircuitChunks;
use tket2::Circuit;

use crate::circuit::convert::CircuitType;
use crate::circuit::{try_with_hugr, with_hugr};
use crate::utils::ConvertPyErr;

/// Split a circuit into chunks of a given size.
#[pyfunction]
pub fn chunks(c: &PyAny, max_chunk_size: usize) -> PyResult<PyCircuitChunks> {
    with_hugr(c, |hugr, typ| {
        // TODO: Detect if the circuit is in tket1 format or Tk2Circuit.
        let chunks = CircuitChunks::split(&hugr, max_chunk_size);
        (chunks, typ).into()
    })
}

/// A pattern that match a circuit exactly
///
/// Python equivalent of [`CircuitChunks`].
///
/// [`CircuitChunks`]: tket2::passes::chunks::CircuitChunks
#[pyclass]
#[pyo3(name = "CircuitChunks")]
#[derive(Debug, Clone, From)]
pub struct PyCircuitChunks {
    /// Rust representation of the circuit chunks.
    pub chunks: CircuitChunks,
    /// Whether to reassemble the circuit in the tket1 or tket2 format.
    pub original_type: CircuitType,
}

#[pymethods]
impl PyCircuitChunks {
    /// Reassemble the chunks into a circuit.
    fn reassemble<'py>(&self, py: Python<'py>) -> PyResult<&'py PyAny> {
        let hugr = self.clone().chunks.reassemble().convert_pyerrs()?;
        self.original_type.convert(py, hugr)
    }

    /// Returns clones of the split circuits.
    fn circuits<'py>(&self, py: Python<'py>) -> PyResult<Vec<&'py PyAny>> {
        self.chunks
            .iter()
            .map(|hugr| self.original_type.convert(py, hugr.clone()))
            .collect()
    }

    /// Replaces a chunk's circuit with an updated version.
    fn update_circuit(&mut self, index: usize, new_circ: &PyAny) -> PyResult<()> {
        try_with_hugr(new_circ, |hugr, _| {
            if hugr.circuit_signature() != self.chunks[index].circuit_signature() {
                return Err(PyAttributeError::new_err(
                    "The new circuit has a different signature.",
                ));
            }
            self.chunks[index] = hugr;
            Ok(())
        })
    }
}
