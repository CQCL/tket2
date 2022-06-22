use pyo3::prelude::*;
use tket_rs::circuit::circuit::{Circuit, CircuitRewrite};
use tket_rs::circuit::operation::WireType;
use tket_rs::circuit::py_circuit::{PyOp, PyOpenCircuit, PySubgraph};

fn _wrap_tket_conversion<F: FnOnce(Circuit) -> Circuit>(
    f: F,
) -> impl FnOnce(Py<PyAny>) -> PyResult<Py<PyAny>> {
    |c: Py<PyAny>| (f)(Circuit::_from_tket1_circ(c)).to_tket1_circ()
}

#[pyfunction]
fn remove_redundancies(c: Py<PyAny>) -> PyResult<Py<PyAny>> {
    _wrap_tket_conversion(|circ| tket_rs::passes::squash::cx_cancel_pass(circ).0)(c)
}

/// A Python module implemented in Rust.
#[pymodule]
fn pyrs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(remove_redundancies, m)?)?;
    m.add_class::<Circuit>()?;
    m.add_class::<PyOp>()?;
    m.add_class::<WireType>()?;
    m.add_class::<PyOpenCircuit>()?;
    m.add_class::<PySubgraph>()?;
    m.add_class::<CircuitRewrite>()?;
    Ok(())
}
