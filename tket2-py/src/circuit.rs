//! Circuit-related functionality and utilities.
#![allow(unused)]

mod convert;
mod cost;
mod tk2circuit;

use derive_more::{From, Into};
use hugr::extension::prelude::{BOOL_T, QB_T};
use hugr::hugr::IdentList;
use hugr::ops::custom::{ExtensionOp, OpaqueOp};
use hugr::ops::{CustomOp, NamedOp, OpName, OpType};
use hugr::types::{CustomType, FunctionType, Type, TypeBound};
use pyo3::prelude::*;
use std::fmt;

use hugr::{type_row, Hugr, HugrView, PortIndex};
use tket2::extension::{LINEAR_BIT, REGISTRY};
use tket2::rewrite::CircuitRewrite;
use tket2::serialize::TKETDecode;
use tket_json_rs::circuit_json::SerialCircuit;

use crate::utils::create_py_exception;
use crate::utils::ConvertPyErr;

pub use self::convert::{try_update_hugr, try_with_hugr, update_hugr, with_hugr, CircuitType};
pub use self::cost::PyCircuitCost;
pub use self::tk2circuit::Tk2Circuit;
use self::tk2circuit::{into_vec, Dfg};
pub use tket2::{Pauli, Tk2Op};

/// The module definition
pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new_bound(py, "circuit")?;
    m.add_class::<Tk2Circuit>()?;
    m.add_class::<Dfg>()?;
    m.add_class::<PyNode>()?;
    m.add_class::<PyWire>()?;
    m.add_class::<WireIter>()?;
    m.add_class::<PyCircuitCost>()?;
    m.add_class::<PyCustom>()?;
    m.add_class::<PyHugrType>()?;
    m.add_class::<PyTypeBound>()?;

    m.add_function(wrap_pyfunction!(validate_hugr, &m)?)?;
    m.add_function(wrap_pyfunction!(to_hugr_dot, &m)?)?;
    m.add_function(wrap_pyfunction!(to_hugr_mermaid, &m)?)?;

    m.add("HugrError", py.get_type_bound::<PyHugrError>())?;
    m.add("BuildError", py.get_type_bound::<PyBuildError>())?;
    m.add("ValidationError", py.get_type_bound::<PyValidationError>())?;
    m.add(
        "HUGRSerializationError",
        py.get_type_bound::<PyHUGRSerializationError>(),
    )?;
    m.add("OpConvertError", py.get_type_bound::<PyOpConvertError>())?;

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
    tket2::serialize::pytket::OpConvertError,
    PyOpConvertError,
    "Error type for the conversion between tket2 and tket1 operations."
);

/// Run the validation checks on a circuit.
#[pyfunction]
pub fn validate_hugr(c: &Bound<PyAny>) -> PyResult<()> {
    try_with_hugr(c, |hugr, _| hugr.validate(&REGISTRY))
}

/// Return a Graphviz DOT string representation of the circuit.
#[pyfunction]
pub fn to_hugr_dot(c: &Bound<PyAny>) -> PyResult<String> {
    with_hugr(c, |hugr, _| hugr.dot_string())
}

/// Return a Mermaid diagram representation of the circuit.
#[pyfunction]
pub fn to_hugr_mermaid(c: &Bound<PyAny>) -> PyResult<String> {
    with_hugr(c, |hugr, _| hugr.mermaid_string())
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

#[pyclass]
/// An iterator over the wires of a node.
pub struct WireIter {
    node: PyNode,
    current: usize,
}

#[pymethods]
impl WireIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<PyWire> {
        slf.current += 1;
        Some(slf.node.__getitem__(slf.current - 1).unwrap())
    }
}

#[pymethods]
impl PyNode {
    /// A string representation of the pattern.
    pub fn __repr__(&self) -> String {
        format!("{:?}", self)
    }

    fn __getitem__(&self, idx: usize) -> PyResult<PyWire> {
        Ok(hugr::Wire::new(self.node, idx).into())
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<WireIter>> {
        let iter = WireIter {
            current: 0,
            node: *slf,
        };
        Py::new(slf.py(), iter)
    }

    fn outs(&self, n: usize) -> Vec<PyWire> {
        (0..n)
            .map(|i| hugr::Wire::new(self.node, i).into())
            .collect()
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

#[pyclass]
#[pyo3(name = "CustomOp")]
#[repr(transparent)]
#[derive(From, Into, PartialEq, Clone)]
struct PyCustom(CustomOp);

impl fmt::Debug for PyCustom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl From<PyCustom> for OpType {
    fn from(op: PyCustom) -> Self {
        op.0.into()
    }
}

#[pymethods]
impl PyCustom {
    #[new]
    fn new(
        extension: &str,
        op_name: &str,
        input_types: Vec<PyHugrType>,
        output_types: Vec<PyHugrType>,
    ) -> PyResult<Self> {
        Ok(CustomOp::new_opaque(OpaqueOp::new(
            IdentList::new(extension).unwrap(),
            op_name,
            Default::default(),
            [],
            FunctionType::new(into_vec(input_types), into_vec(output_types)),
        ))
        .into())
    }

    fn to_custom(&self) -> Self {
        self.clone()
    }
    pub fn __repr__(&self) -> String {
        format!("{:?}", self)
    }

    fn name(&self) -> String {
        self.0.name().to_string()
    }
}
#[pyclass]
#[pyo3(name = "TypeBound")]
#[derive(PartialEq, Clone, Debug)]
enum PyTypeBound {
    Any,
    Copyable,
    Eq,
}

impl From<PyTypeBound> for TypeBound {
    fn from(bound: PyTypeBound) -> Self {
        match bound {
            PyTypeBound::Any => TypeBound::Any,
            PyTypeBound::Copyable => TypeBound::Copyable,
            PyTypeBound::Eq => TypeBound::Eq,
        }
    }
}

impl From<TypeBound> for PyTypeBound {
    fn from(bound: TypeBound) -> Self {
        match bound {
            TypeBound::Any => PyTypeBound::Any,
            TypeBound::Copyable => PyTypeBound::Copyable,
            TypeBound::Eq => PyTypeBound::Eq,
        }
    }
}

#[pyclass]
#[pyo3(name = "HugrType")]
#[repr(transparent)]
#[derive(From, Into, PartialEq, Clone)]
struct PyHugrType(Type);

impl fmt::Debug for PyHugrType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[pymethods]
impl PyHugrType {
    #[new]
    fn new(extension: &str, type_name: &str, bound: PyTypeBound) -> Self {
        Self(Type::new_extension(CustomType::new_simple(
            type_name.into(),
            IdentList::new(extension).unwrap(),
            bound.into(),
        )))
    }
    #[staticmethod]
    fn qubit() -> Self {
        Self(QB_T)
    }

    #[staticmethod]
    fn linear_bit() -> Self {
        Self(LINEAR_BIT.to_owned())
    }

    #[staticmethod]
    fn bool() -> Self {
        Self(BOOL_T)
    }

    pub fn __repr__(&self) -> String {
        format!("{:?}", self)
    }
}
