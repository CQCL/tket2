//! Circuit chunking utilities.

use derive_more::From;
use pyo3::exceptions::PyAttributeError;
use pyo3::prelude::*;
use tket2::json::TKETDecode;
use tket2::passes::CircuitChunks;
use tket2::Circuit;
use tket_json_rs::circuit_json::SerialCircuit;

use crate::circuit::{with_hugr, T2Circuit};

/// Split a circuit into chunks of a given size.
#[pyfunction]
pub fn chunks(c: Py<PyAny>, max_chunk_size: usize) -> PyResult<PyCircuitChunks> {
    with_hugr(c, |hugr| {
        // TODO: Detect if the circuit is in tket1 format or T2Circuit.
        let is_tket1 = true;
        let chunks = CircuitChunks::split(&hugr, max_chunk_size);
        (chunks, is_tket1).into()
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
    /// Whether to reassemble the circuit in the tket1 format.
    pub in_tket1: bool,
}

#[pymethods]
impl PyCircuitChunks {
    /// Reassemble the chunks into a circuit.
    fn reassemble(&self) -> PyResult<Py<PyAny>> {
        let hugr = self.clone().chunks.reassemble()?;
        Python::with_gil(|py| match self.in_tket1 {
            true => Ok(SerialCircuit::encode(&hugr)?.to_tket1(py)?.into_py(py)),
            false => Ok(T2Circuit { hugr }.into_py(py)),
        })
    }

    /// Returns clones of the split circuits.
    fn circuits(&self) -> PyResult<Vec<Py<PyAny>>> {
        self.chunks
            .iter()
            .map(|hugr| SerialCircuit::encode(hugr)?.to_tket1_with_gil())
            .collect()
    }

    /// Replaces a chunk's circuit with an updated version.
    fn update_circuit(&mut self, index: usize, new_circ: Py<PyAny>) -> PyResult<()> {
        let hugr = SerialCircuit::from_tket1_with_gil(new_circ)?.decode()?;
        if hugr.circuit_signature() != self.chunks[index].circuit_signature() {
            return Err(PyAttributeError::new_err(
                "The new circuit has a different signature.",
            ));
        }
        self.chunks[index] = hugr;
        Ok(())
    }
}
