//! PyO3 wrapper for the Badger circuit optimiser.

use pyo3::prelude::*;

mod badger;
pub use badger::PyBadgerOptimiser;

mod cost_function;
pub use cost_function::BadgerCostFunction;
use cost_function::PyBadgerStrategy;

/// The module definition
pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "optimiser")?;
    m.add_class::<PyBadgerOptimiser>()?;
    Ok(m)
}
