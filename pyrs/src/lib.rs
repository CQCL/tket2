use pyo3::prelude::*;
use pythonize::{depythonize, pythonize};
use tket_json_rs::circuit_json::SerialCircuit;
use tket_rs::circuit::circuit::Circuit;

#[pyfunction]
fn remove_redundancies(c: Py<PyAny>) -> PyResult<Py<PyAny>> {
    let ser: SerialCircuit = Python::with_gil(|py| depythonize(c.as_ref(py)))?;

    let circ: Circuit = ser.clone().into();

    let (circ, _) = tket_rs::passes::squash::cx_cancel_pass(circ);
    let reser: SerialCircuit = circ.into();
    Ok(Python::with_gil(|py| pythonize(py, &reser).unwrap()))
}

/// A Python module implemented in Rust.
#[pymodule]
fn pyrs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(remove_redundancies, m)?)?;
    Ok(())
}
