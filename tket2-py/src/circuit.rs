//! Circuit-related functionality and utilities.
#![allow(unused)]

mod convert;
mod cost;
mod tk2circuit;

use derive_more::{From, Into};
use hugr::extension::prelude::{BOOL_T, QB_T};
use hugr::hugr::IdentList;
use hugr::ops::custom::{ExtensionOp, OpaqueOp};
use hugr::ops::{NamedOp, OpName, OpType};
use hugr::types::{CustomType, Type, TypeBound};
use pyo3::prelude::*;
use std::fmt;

use hugr::{type_row, Hugr, HugrView, PortIndex};
use tket2::extension::REGISTRY;
use tket2::rewrite::CircuitRewrite;
use tket2::serialize::TKETDecode;
use tket_json_rs::circuit_json::SerialCircuit;

use crate::utils::create_py_exception;
use crate::utils::ConvertPyErr;

pub use self::convert::{try_update_circ, try_with_circ, update_circ, with_circ, CircuitType};
pub use self::cost::PyCircuitCost;
pub use self::tk2circuit::Tk2Circuit;
pub use tket2::{Pauli, Tk2Op};

/// The module definition
pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "circuit")?;
    m.add_class::<Tk2Circuit>()?;
    m.add_class::<PyNode>()?;
    m.add_class::<PyWire>()?;
    m.add_class::<PyCircuitCost>()?;

    m.add_function(wrap_pyfunction!(validate_circuit, &m)?)?;
    m.add_function(wrap_pyfunction!(render_circuit_dot, &m)?)?;
    m.add_function(wrap_pyfunction!(render_circuit_mermaid, &m)?)?;

    m.add("HugrError", py.get_type::<PyHugrError>())?;
    m.add("BuildError", py.get_type::<PyBuildError>())?;
    m.add("ValidationError", py.get_type::<PyValidationError>())?;
    m.add(
        "HUGRSerializationError",
        py.get_type::<PyHUGRSerializationError>(),
    )?;
    m.add("TK1ConvertError", py.get_type::<PyTK1ConvertError>())?;

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
    tket2::serialize::pytket::TK1ConvertError,
    PyTK1ConvertError,
    "Error type for the conversion between tket2 and tket1 operations."
);

/// Run the validation checks on a circuit.
#[pyfunction]
pub fn validate_circuit(c: &Bound<PyAny>) -> PyResult<()> {
    try_with_circ(c, |circ, _| circ.hugr().validate(&REGISTRY))
}

/// Return a Graphviz DOT string representation of the circuit.
#[pyfunction]
pub fn render_circuit_dot(c: &Bound<PyAny>) -> PyResult<String> {
    with_circ(c, |hugr, _| hugr.dot_string())
}

/// Return a Mermaid diagram representation of the circuit.
#[pyfunction]
pub fn render_circuit_mermaid(c: &Bound<PyAny>) -> PyResult<String> {
    with_circ(c, |hugr, _| hugr.mermaid_string())
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
    #[new]
    fn new(index: usize) -> Self {
        Self {
            node: serde_json::from_value(serde_json::Value::Number(index.into())).unwrap(),
        }
    }
    /// A string representation of the pattern.
    pub fn __repr__(&self) -> String {
        format!("{:?}", self)
    }
}

/// A [`hugr::Node`] wrapper for Python.
#[pyclass]
#[pyo3(name = "Wire")]
#[repr(transparent)]
#[derive(From, Into, PartialEq, Eq, Hash, Clone, Copy)]
pub struct PyWire {
    /// Rust representation of the node
    pub wire: hugr::Wire,
}

impl fmt::Display for PyWire {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.wire.fmt(f)
    }
}

impl fmt::Debug for PyWire {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.wire.fmt(f)
    }
}

#[pymethods]
impl PyWire {
    /// A string representation of the pattern.
    pub fn __repr__(&self) -> String {
        format!("{:?}", self)
    }

    fn node(&self) -> PyNode {
        self.wire.node().into()
    }

    fn port(&self) -> usize {
        self.wire.source().index()
    }
}
