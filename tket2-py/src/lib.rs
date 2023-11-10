//! Python bindings for TKET2.
#![warn(missing_docs)]

pub mod circuit;
pub mod optimiser;
pub mod passes;
pub mod pattern;

use pyo3::prelude::*;

/// The Python bindings to TKET2.
#[pymodule]
#[pyo3(name = "tket2")]
fn tket2_py(py: Python, m: &PyModule) -> PyResult<()> {
    add_submodule(py, m, circuit::module(py)?)?;
    add_submodule(py, m, optimiser::module(py)?)?;
    add_submodule(py, m, passes::module(py)?)?;
    add_submodule(py, m, pattern::module(py)?)?;
    Ok(())
}

fn add_submodule(py: Python, parent: &PyModule, submodule: &PyModule) -> PyResult<()> {
    parent.add_submodule(submodule)?;

    // Add submodule to sys.modules.
    // This is required to be able to do `from parent.submodule import ...`.
    //
    // See [https://github.com/PyO3/pyo3/issues/759]
    let parent_name = parent.name()?;
    let submodule_name = submodule.name()?;
    let modules = py.import("sys")?.getattr("modules")?;
    modules.set_item(format!("{parent_name}.{submodule_name}"), submodule)?;
    Ok(())
}
