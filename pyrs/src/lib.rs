use hugr::{hugr::views::SiblingGraph, ops::handle::DfgID, Hugr, HugrView};
use pyo3::{exceptions::PyTypeError, prelude::*};
use tket2::{
    circuit::HierarchyView,
    json::TKETDecode,
    passes::apply_greedy_commutation,
    portmatching::{CircuitPattern, PatternMatcher},
};
use tket_json_rs::circuit_json::SerialCircuit;

use tket2::extension::REGISTRY;
use tket2::portmatching::pyo3::PyValidateError;

#[pyfunction]
fn check_soundness(c: Py<PyAny>) -> PyResult<()> {
    let ser_c = SerialCircuit::_from_tket1(c);
    let hugr: hugr::Hugr = ser_c.decode().unwrap();
    hugr.validate(&REGISTRY)
        .map_err(|e| PyValidateError::new_err(e.to_string()))
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

fn pyerr_string<T: std::fmt::Debug>(e: T) -> PyErr {
    PyErr::new::<PyTypeError, _>(format!("{:?}", e))
}
#[pyfunction]
fn greedy_depth_reduce(py_c: PyObject) -> PyResult<(PyObject, u32)> {
    let s_c = SerialCircuit::_from_tket1(py_c.clone());
    let mut h: Hugr = s_c.decode().map_err(pyerr_string)?;
    let n_moves = apply_greedy_commutation(&mut h).map_err(pyerr_string)?;
    let circ: SiblingGraph<'_, DfgID> = SiblingGraph::new(&h, h.root());

    let s_c = SerialCircuit::encode(&circ).map_err(pyerr_string)?;
    Ok((s_c.to_tket1()?, n_moves))
}

/// The Python bindings to TKET2.
#[pymodule]
fn pyrs(py: Python, m: &PyModule) -> PyResult<()> {
    add_patterns_module(py, m)?;
    add_pass_module(py, m)?;
    m.add_function(wrap_pyfunction!(to_hugr_dot, m)?)?;
    m.add_function(wrap_pyfunction!(to_hugr, m)?)?;

    m.add("ValidateError", py.get_type::<PyValidateError>())?;
    m.add_function(wrap_pyfunction!(check_soundness, m)?)?;
    Ok(())
}

fn add_patterns_module(py: Python, parent: &PyModule) -> PyResult<()> {
    let m = PyModule::new(py, "patterns")?;
    m.add_class::<CircuitPattern>()?;
    m.add_class::<PatternMatcher>()?;
    parent.add_submodule(m)?;
    Ok(())
}

fn add_pass_module(py: Python, parent: &PyModule) -> PyResult<()> {
    let m = PyModule::new(py, "passes")?;
    m.add_function(wrap_pyfunction!(greedy_depth_reduce, m)?)?;
    m.add_class::<tket2::T2Op>()?;
    parent.add_submodule(m)?;
    Ok(())
}
