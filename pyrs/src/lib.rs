use hugr::{hugr::views::SiblingGraph, ops::handle::DfgID, Hugr, HugrView};
use pyo3::{exceptions::PyTypeError, prelude::*};
use tket2::{
    circuit::HierarchyView,
    json::TKETDecode,
    passes::apply_greedy_commutation,
    portmatching::{CircuitMatcher, CircuitPattern},
};
use tket_json_rs::circuit_json::SerialCircuit;

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
    Ok(())
}

fn add_patterns_module(py: Python, parent: &PyModule) -> PyResult<()> {
    let m = PyModule::new(py, "patterns")?;
    m.add_class::<CircuitPattern>()?;
    m.add_class::<CircuitMatcher>()?;
    parent.add_submodule(m)?;
    Ok(())
}

fn add_pass_module(py: Python, parent: &PyModule) -> PyResult<()> {
    let m = PyModule::new(py, "passes")?;
    m.add_function(wrap_pyfunction!(greedy_depth_reduce, m)?)?;
    parent.add_submodule(m)?;
    Ok(())
}
