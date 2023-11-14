//! Utilities for calling Hugr functions on generic python objects.

use pyo3::exceptions::PyAttributeError;
use pyo3::{prelude::*, PyTypeInfo};

use derive_more::From;
use hugr::{Hugr, HugrView};
use serde::Serialize;
use tket2::extension::REGISTRY;
use tket2::json::TKETDecode;
use tket2::passes::CircuitChunks;
use tket_json_rs::circuit_json::SerialCircuit;

use crate::pattern::rewrite::PyCircuitRewrite;

/// A manager for tket 2 operations on a tket 1 Circuit.
#[pyclass]
#[derive(Clone, Debug, PartialEq, From)]
pub struct T2Circuit {
    /// Rust representation of the circuit.
    pub hugr: Hugr,
}

#[pymethods]
impl T2Circuit {
    /// Cast a tket1 circuit to a [`T2Circuit`].
    #[new]
    pub fn from_circuit(circ: PyObject) -> PyResult<Self> {
        Ok(Self {
            hugr: with_hugr(circ, |hugr| hugr)?,
        })
    }

    /// Cast the [`T2Circuit`] to a tket1 circuit.
    pub fn finish(&self) -> PyResult<PyObject> {
        SerialCircuit::encode(&self.hugr)?.to_tket1_with_gil()
    }

    /// Apply a rewrite on the circuit.
    pub fn apply_match(&mut self, rw: PyCircuitRewrite) {
        rw.rewrite.apply(&mut self.hugr).expect("Apply error.");
    }

    /// Encode the circuit as a HUGR json string.
    pub fn to_hugr_json(&self) -> PyResult<String> {
        Ok(serde_json::to_string(&self.hugr).unwrap())
    }

    /// Decode a HUGR json string to a circuit.
    #[staticmethod]
    pub fn from_hugr_json(json: &str) -> PyResult<Self> {
        let hugr = serde_json::from_str(json)
            .map_err(|e| PyErr::new::<PyAttributeError, _>(format!("Invalid encoded HUGR: {e}")))?;
        Ok(T2Circuit { hugr })
    }

    /// Encode the circuit as a tket1 json string.
    ///
    /// FIXME: Currently the encoded circuit cannot be loaded back due to
    /// [https://github.com/CQCL/hugr/issues/683]
    pub fn to_tket1_json(&self) -> PyResult<String> {
        Ok(serde_json::to_string(&SerialCircuit::encode(&self.hugr)?).unwrap())
    }

    /// Decode a tket1 json string to a circuit.
    #[staticmethod]
    pub fn from_tket1_json(json: &str) -> PyResult<Self> {
        let tk1: SerialCircuit = serde_json::from_str(json)
            .map_err(|e| PyErr::new::<PyAttributeError, _>(format!("Invalid encoded HUGR: {e}")))?;
        Ok(T2Circuit {
            hugr: tk1.decode()?,
        })
    }
}
impl T2Circuit {
    /// Tries to extract a T2Circuit from a python object.
    ///
    /// Returns an error if the py object is not a T2Circuit.
    pub fn try_extract(circ: Py<PyAny>) -> PyResult<Self> {
        Python::with_gil(|py| circ.as_ref(py).extract::<T2Circuit>())
    }
}

/// Apply a fallible function expecting a hugr on a pytket circuit.
pub fn try_with_hugr<T, E, F>(circ: Py<PyAny>, f: F) -> PyResult<T>
where
    E: Into<PyErr>,
    F: FnOnce(Hugr) -> Result<T, E>,
{
    let hugr = Python::with_gil(|py| -> PyResult<Hugr> {
        let circ = circ.as_ref(py);
        match T2Circuit::extract(circ) {
            // hugr circuit
            Ok(t2circ) => Ok(t2circ.hugr),
            // tket1 circuit
            Err(_) => Ok(SerialCircuit::from_tket1(circ)?.decode()?),
        }
    })?;
    (f)(hugr).map_err(|e| e.into())
}

/// Apply a function expecting a hugr on a pytket circuit.
pub fn with_hugr<T, F>(circ: Py<PyAny>, f: F) -> PyResult<T>
where
    F: FnOnce(Hugr) -> T,
{
    try_with_hugr(circ, |hugr| Ok::<T, PyErr>((f)(hugr)))
}

/// Apply a hugr-to-hugr function on a pytket circuit, and return the modified circuit.
pub fn try_update_hugr<E, F>(circ: Py<PyAny>, f: F) -> PyResult<Py<PyAny>>
where
    E: Into<PyErr>,
    F: FnOnce(Hugr) -> Result<Hugr, E>,
{
    let hugr = try_with_hugr(circ, f)?;
    SerialCircuit::encode(&hugr)?.to_tket1_with_gil()
}

/// Apply a hugr-to-hugr function on a pytket circuit, and return the modified circuit.
pub fn update_hugr<F>(circ: Py<PyAny>, f: F) -> PyResult<Py<PyAny>>
where
    F: FnOnce(Hugr) -> Hugr,
{
    let hugr = with_hugr(circ, f)?;
    SerialCircuit::encode(&hugr)?.to_tket1_with_gil()
}
