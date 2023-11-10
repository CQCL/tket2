//! Utilities for calling Hugr functions on generic python objects.

use pyo3::{prelude::*, PyTypeInfo};

use hugr::{Hugr, HugrView};
use tket2::extension::REGISTRY;
use tket2::json::TKETDecode;
use tket2::passes::CircuitChunks;
use tket_json_rs::circuit_json::SerialCircuit;

use crate::pattern::rewrite::PyCircuitRewrite;

/// A manager for tket 2 operations on a tket 1 Circuit.
#[pyclass]
#[derive(Clone, Debug, PartialEq)]
pub struct T2Circuit {
    /// Rust representation of the circuit.
    pub hugr: Hugr,
}

#[pymethods]
impl T2Circuit {
    #[new]
    fn from_circuit(circ: PyObject) -> PyResult<Self> {
        Ok(Self {
            hugr: super::to_hugr(circ)?,
        })
    }

    fn finish(&self) -> PyResult<PyObject> {
        SerialCircuit::encode(&self.hugr)?.to_tket1_with_gil()
    }

    fn apply_match(&mut self, rw: PyCircuitRewrite) {
        rw.rewrite.apply(&mut self.hugr).expect("Apply error.");
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
