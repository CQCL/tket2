//! Circuit-related functionality and utilities.
#![allow(unused)]

pub mod convert;
pub mod cost;

use derive_more::{From, Into};
use pyo3::prelude::*;
use std::fmt;

use hugr::{Hugr, HugrView};
use tket2::extension::REGISTRY;
use tket2::json::TKETDecode;
use tket2::rewrite::CircuitRewrite;
use tket_json_rs::circuit_json::SerialCircuit;

use crate::utils::create_py_exception;

pub use self::convert::{try_update_hugr, try_with_hugr, update_hugr, with_hugr, Tk2Circuit};
pub use self::cost::PyCircuitCost;
pub use tket2::{Pauli, Tk2Op};

/// The module definition
pub fn module(py: Python) -> PyResult<&PyModule> {
    let m = PyModule::new(py, "_circuit")?;
    m.add_class::<Tk2Circuit>()?;
    m.add_class::<PyNode>()?;
    m.add_class::<PyCircuitCost>()?;
    m.add_class::<Tk2Op>()?;
    m.add_class::<Pauli>()?;

    m.add_function(wrap_pyfunction!(validate_hugr, m)?)?;
    m.add_function(wrap_pyfunction!(to_hugr_dot, m)?)?;

    m.add("HugrError", py.get_type::<PyHugrError>())?;
    m.add("BuildError", py.get_type::<PyBuildError>())?;
    m.add("ValidationError", py.get_type::<PyValidationError>())?;
    m.add(
        "HUGRSerializationError",
        py.get_type::<PyHUGRSerializationError>(),
    )?;
    m.add("OpConvertError", py.get_type::<PyOpConvertError>())?;

    Ok(m)
}

create_py_exception!(
    hugr::hugr::HugrError,
    PyHugrError,
    "Errors that can occur while manipulating a HUGR."
);

create_py_exception!(
    hugr::builder::BuildError,
    PyBuildError,
    "Error while building the HUGR."
);

create_py_exception!(
    hugr::hugr::validate::ValidationError,
    PyValidationError,
    "Errors that can occur while validating a Hugr."
);

create_py_exception!(
    hugr::hugr::serialize::HUGRSerializationError,
    PyHUGRSerializationError,
    "Errors that can occur while serializing a HUGR."
);

create_py_exception!(
    tket2::json::OpConvertError,
    PyOpConvertError,
    "Error type for the conversion between tket2 and tket1 operations."
);

/// Run the validation checks on a circuit.
#[pyfunction]
pub fn validate_hugr(c: &PyAny) -> PyResult<()> {
    try_with_hugr(c, |hugr, _| hugr.validate(&REGISTRY))
}

/// Return a Graphviz DOT string representation of the circuit.
#[pyfunction]
pub fn to_hugr_dot(c: &PyAny) -> PyResult<String> {
    with_hugr(c, |hugr, _| hugr.dot_string())
}

/// A [`hugr::Node`] wrapper for Python.
#[pyclass]
#[pyo3(name = "Node")]
#[repr(transparent)]
#[derive(From, Into, PartialEq, Eq, Hash, Clone, Copy)]
pub struct PyNode {
    /// Rust representation of the node
    pub node: hugr::Node,
}

impl fmt::Display for PyNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.node.fmt(f)
    }
}

impl fmt::Debug for PyNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.node.fmt(f)
    }
}

#[pymethods]
impl PyNode {
    /// A string representation of the pattern.
    pub fn __repr__(&self) -> String {
        format!("{:?}", self)
    }
}
