//! Python bindings for TKET2.
#![warn(missing_docs)]
use circuit::{add_circuit_module, try_with_hugr};
use pyo3::prelude::*;
use tket2::{json::TKETDecode, optimiser::taso, passes::apply_greedy_commutation};
use tket_json_rs::circuit_json::SerialCircuit;

mod circuit;

#[pyfunction]
fn greedy_depth_reduce(py_c: PyObject) -> PyResult<(PyObject, u32)> {
    try_with_hugr(py_c, |mut h| {
        let n_moves = apply_greedy_commutation(&mut h)?;
        let py_c = SerialCircuit::encode(&h)?.to_tket1()?;
        PyResult::Ok((py_c, n_moves))
    })
}

/// The Python bindings to TKET2.
#[pymodule]
fn pyrs(py: Python, m: &PyModule) -> PyResult<()> {
    add_circuit_module(py, m)?;
    add_pattern_module(py, m)?;
    add_pass_module(py, m)?;
    add_optimiser_module(py, m)?;
    Ok(())
}

/// portmatching module
fn add_pattern_module(py: Python, parent: &PyModule) -> PyResult<()> {
    let m = PyModule::new(py, "pattern")?;
    m.add_class::<tket2::portmatching::CircuitPattern>()?;
    m.add_class::<tket2::portmatching::PatternMatcher>()?;

    m.add(
        "InvalidPatternError",
        py.get_type::<tket2::portmatching::pattern::PyInvalidPatternError>(),
    )?;
    m.add(
        "InvalidReplacementError",
        py.get_type::<hugr::hugr::views::sibling_subgraph::PyInvalidReplacementError>(),
    )?;

    parent.add_submodule(m)
}

fn add_pass_module(py: Python, parent: &PyModule) -> PyResult<()> {
    let m = PyModule::new(py, "passes")?;
    m.add_function(wrap_pyfunction!(greedy_depth_reduce, m)?)?;
    m.add_class::<tket2::T2Op>()?;
    m.add(
        "PullForwardError",
        py.get_type::<tket2::passes::PyPullForwardError>(),
    )?;
    parent.add_submodule(m)?;
    Ok(())
}

/// circuit optimisation module
fn add_optimiser_module(py: Python, parent: &PyModule) -> PyResult<()> {
    let m = PyModule::new(py, "optimiser")?;
    m.add_class::<taso::pyo3::PyDefaultTasoOptimiser>()?;

    parent.add_submodule(m)
}
