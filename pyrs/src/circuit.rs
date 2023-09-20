//! Circuit-related functionality and utilities.
#![allow(unused)]

use pyo3::prelude::*;

use hugr::{Hugr, HugrView};
use tket2::extension::REGISTRY;
use tket2::json::TKETDecode;
use tket_json_rs::circuit_json::SerialCircuit;

/// Apply a fallible function expecting a hugr on a pytket circuit.
pub fn try_with_hugr<T, E, F>(circ: Py<PyAny>, f: F) -> PyResult<T>
where
    E: Into<PyErr>,
    F: FnOnce(Hugr) -> Result<T, E>,
{
    let hugr = SerialCircuit::_from_tket1(circ).decode()?;
    (f)(hugr).map_err(|e| e.into())
}

/// Apply a function expecting a hugr on a pytket circuit.
pub fn with_hugr<T, F: FnOnce(Hugr) -> T>(circ: Py<PyAny>, f: F) -> PyResult<T> {
    try_with_hugr(circ, |hugr| Ok::<T, PyErr>((f)(hugr)))
}

/// Apply a hugr-to-hugr function on a pytket circuit, and return the modified circuit.
pub fn try_update_hugr<E: Into<PyErr>, F: FnOnce(Hugr) -> Result<Hugr, E>>(
    circ: Py<PyAny>,
    f: F,
) -> PyResult<Py<PyAny>> {
    let hugr = try_with_hugr(circ, f)?;
    SerialCircuit::encode(&hugr)?.to_tket1()
}

/// Apply a hugr-to-hugr function on a pytket circuit, and return the modified circuit.
pub fn update_hugr<F: FnOnce(Hugr) -> Hugr>(circ: Py<PyAny>, f: F) -> PyResult<Py<PyAny>> {
    let hugr = with_hugr(circ, f)?;
    SerialCircuit::encode(&hugr)?.to_tket1()
}

#[pyfunction]
pub fn validate_hugr(c: Py<PyAny>) -> PyResult<()> {
    try_with_hugr(c, |hugr| hugr.validate(&REGISTRY))
}

#[pyfunction]
pub fn to_hugr_dot(c: Py<PyAny>) -> PyResult<String> {
    with_hugr(c, |hugr| hugr.dot_string())
}

#[pyfunction]
pub fn to_hugr(c: Py<PyAny>) -> PyResult<Hugr> {
    with_hugr(c, |hugr| hugr)
}
