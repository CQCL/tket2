//! Circuit-related functionality.

use pyo3::prelude::*;

use hugr::{Hugr, HugrView};
use tket2::extension::REGISTRY;
use tket2::json::TKETDecode;
use tket_json_rs::circuit_json::SerialCircuit;

/// Apply a function expecting a hugr on a pytket circuit.
fn with_hugr<T, F: FnOnce(Hugr) -> T>(
    circ: Py<PyAny>,
    f: F,
) -> T {
    let hugr = SerialCircuit::_from_tket1(circ).decode().unwrap();
    (f)(hugr)
}

/// Apply a hugr-to-hugr function on a pytket circuit, and return the modified circuit.
fn update_hugr<F: FnOnce(Hugr) -> Hugr>(
    circ: Py<PyAny>,
    f: F,
) -> PyResult<Py<PyAny>> {
    // TODO: Export errors as python errors
    let hugr = SerialCircuit::_from_tket1(circ).decode().unwrap();
    let hugr = (f)(hugr);
    let reser: SerialCircuit = SerialCircuit::encode(&hugr).unwrap();
    reser.to_tket1()
}

#[pyfunction]
fn validate_hugr(c: Py<PyAny>) -> PyResult<()> {
    with_hugr(c, |hugr| {
        hugr.validate(&REGISTRY).unwrap();
            //.map_err(|e| PyValidateError::new_err(e.to_string()))
    });
    Ok(())
}

#[pyfunction]
fn to_hugr_dot(c: Py<PyAny>) -> PyResult<String> {
    let ser_c = SerialCircuit::_from_tket1(c);
    let hugr: Hugr = ser_c.decode().unwrap();
    Ok(hugr.dot_string())
}

#[pyfunction]
fn to_hugr(c: Py<PyAny>) -> PyResult<Hugr> {
    let ser_c = SerialCircuit::_from_tket1(c);
    let hugr: Hugr = ser_c.decode().unwrap();
    Ok(hugr)
}