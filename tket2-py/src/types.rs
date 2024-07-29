//! Hugr types

use derive_more::{From, Into};
use hugr::extension::prelude::{BOOL_T, QB_T};
use hugr::hugr::IdentList;
use hugr::types::{CustomType, Type, TypeBound};
use pyo3::prelude::*;
use std::fmt;

/// The module definition
pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new_bound(py, "types")?;
    m.add_class::<PyHugrType>()?;
    m.add_class::<PyTypeBound>()?;

    Ok(m)
}

/// Bounds on the valid operations on a type in a HUGR program.
#[pyclass]
#[pyo3(name = "TypeBound")]
#[derive(PartialEq, Clone, Debug)]
pub enum PyTypeBound {
    /// No bound on the type.
    Any,
    /// The type can be copied in the program.
    Copyable,
}

impl From<PyTypeBound> for TypeBound {
    fn from(bound: PyTypeBound) -> Self {
        match bound {
            PyTypeBound::Any => TypeBound::Any,
            PyTypeBound::Copyable => TypeBound::Copyable,
        }
    }
}

impl From<TypeBound> for PyTypeBound {
    fn from(bound: TypeBound) -> Self {
        match bound {
            TypeBound::Any => PyTypeBound::Any,
            TypeBound::Copyable => PyTypeBound::Copyable,
        }
    }
}

/// A HUGR type
#[pyclass]
#[pyo3(name = "HugrType")]
#[repr(transparent)]
#[derive(From, Into, PartialEq, Clone)]
pub struct PyHugrType(Type);

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
    fn bool() -> Self {
        Self(BOOL_T)
    }

    /// A string representation of the type.
    pub fn __repr__(&self) -> String {
        format!("{:?}", self)
    }
}
