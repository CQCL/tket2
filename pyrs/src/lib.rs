use pyo3::prelude::*;
use tket_rs::circuit::circuit::Circuit;
use tket_rs::circuit_json;

#[pyfunction]
fn remove_redundancies(s: String) -> PyResult<String> {
    let ser: circuit_json::SerialCircuit = serde_json::from_str(&s).unwrap();

    let circ: Circuit = ser.clone().into();

    let circ = tket_rs::passes::redundancy::remove_redundancies(circ);

    let reser: circuit_json::SerialCircuit = circ.into();

    Ok(serde_json::to_string(&reser).unwrap())
}

/// A Python module implemented in Rust.
#[pymodule]
fn pyrs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(remove_redundancies, m)?)?;
    Ok(())
}
