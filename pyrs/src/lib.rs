use hugr::Hugr;
use portgraph::graph::{Direction, NodeIndex};
use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use tket2::json::json_convert::load_serial;
use tket_json_rs::circuit_json::SerialCircuit;
use tket_json_rs::optype::OpType;

create_exception!(pyrs, PyValidateError, PyException);

#[pyfunction]
fn check_soundness(c: Py<PyAny>) -> PyResult<()> {
    let ser_c = SerialCircuit::_from_tket1(c);
    let hugr: Hugr = load_serial(ser_c);
    println!("{}", hugr.dot_string());
    hugr.validate()
        .map_err(|e| PyValidateError::new_err(e.to_string()))
}

/// A Python module implemented in Rust.
#[pymodule]
fn pyrs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(check_soundness, m)?)?;

    m.add("ValidateError", _py.get_type::<PyValidateError>())?;
    Ok(())
}
