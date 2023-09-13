//! Python bindings for TKET2.
#![warn(missing_docs)]

mod circuit;

use pyo3::prelude::*;

/// The Python bindings to TKET2.
#[pymodule]
fn pyrs(py: Python, m: &PyModule) -> PyResult<()> {
    add_circuit_module(py, m)?;
    add_pattern_module(py, m)?;
    Ok(())
}

/// circuit module
fn add_circuit_module(py: Python, parent: &PyModule) -> PyResult<()> {
    let m = PyModule::new(py, "circuit")?;
    m.add_class::<tket2::T2Op>()?;
    m.add_class::<tket2::Pauli>()?;
    m.add(
        "ValidationError",
        py.get_type::<hugr::hugr::validate::PyValidationError>(),
    )?;
    m.add(
        "OpConvertError",
        py.get_type::<tket2::json::PyOpConvertError>(),
    )?;
    parent.add_submodule(m)
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
