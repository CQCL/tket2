//! Python wrapper for OpArg.

use derive_more::derive::{From, Into};
use portgraph::algorithms::convex::LineIndex;
use pyo3::prelude::*;
// Note: These imports may need to be adjusted based on the actual tket crate structure
use tket::rewrite::matcher::OpArg;

/// A wrapper for OpArg, not itself exposed to Python!
///
/// Each variant has its own Python wrapper type:
///  - converting a PyOpArg to a Python object will create the appropriate variant wrapper
///  - extracting a PyOpArg from a Python object will succeed if the object is one
///    of the variant wrappers.
#[derive(Debug, Clone, From, Into)]
pub struct PyOpArg(OpArg);

/// Python wrapper for qubit operation arguments.
#[pyclass(name = "QubitOpArg")]
#[derive(Debug, Clone)]
pub struct PyQubitOpArg(usize);

#[pymethods]
impl PyQubitOpArg {
    #[new]
    fn new(index: usize) -> Self {
        Self(index)
    }

    #[getter]
    fn index(&self) -> usize {
        self.0
    }

    fn __repr__(&self) -> String {
        format!("Qubit({})", self.0)
    }
}

/// Python wrapper for qubit operation arguments positioned before operations.
#[pyclass(name = "QubitOpBeforeArg")]
#[derive(Debug, Clone)]
pub struct PyQubitOpBeforeArg(usize);

#[pymethods]
impl PyQubitOpBeforeArg {
    #[new]
    fn new(index: usize) -> Self {
        Self(index)
    }

    #[getter]
    fn index(&self) -> usize {
        self.0
    }

    fn __repr__(&self) -> String {
        format!("QubitOpBeforeArg({})", self.0)
    }
}

/// Python wrapper for qubit operation arguments positioned after operations.
#[pyclass(name = "QubitOpAfterArg")]
#[derive(Debug, Clone)]
pub struct PyQubitOpAfterArg(usize);

#[pymethods]
impl PyQubitOpAfterArg {
    #[new]
    fn new(index: usize) -> Self {
        Self(index)
    }

    #[getter]
    fn index(&self) -> usize {
        self.0
    }

    fn __repr__(&self) -> String {
        format!("QubitOpAfterArg({})", self.0)
    }
}

/// Python wrapper for constant floating point arguments.
#[pyclass(name = "ConstF64Arg")]
#[derive(Debug, Clone)]
pub struct PyConstF64Arg(f64);

#[pymethods]
impl PyConstF64Arg {
    #[new]
    fn new(value: f64) -> Self {
        Self(value)
    }

    #[getter]
    fn value(&self) -> f64 {
        self.0
    }

    fn __repr__(&self) -> String {
        format!("ConstF64Arg({})", self.0)
    }
}

impl<'py> IntoPyObject<'py> for PyOpArg {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(match self.0 {
            OpArg::Qubit(index) => PyQubitOpArg(index.0 as usize).into_pyobject(py)?.into_any(),
            OpArg::QubitOpBefore(index) => PyQubitOpBeforeArg(index.0 as usize)
                .into_pyobject(py)?
                .into_any(),
            OpArg::QubitOpAfter(index) => PyQubitOpAfterArg(index.0 as usize)
                .into_pyobject(py)?
                .into_any(),
            OpArg::ConstF64(value) => PyConstF64Arg(value).into_pyobject(py)?.into_any(),
        })
    }
}

impl<'py> FromPyObject<'py> for PyOpArg {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(qubit) = ob.extract::<PyQubitOpArg>() {
            Ok(Self(OpArg::Qubit(LineIndex(qubit.0 as u32))))
        } else if let Ok(qubit_op_before) = ob.extract::<PyQubitOpBeforeArg>() {
            Ok(Self(OpArg::QubitOpBefore(LineIndex(
                qubit_op_before.0 as u32,
            ))))
        } else if let Ok(qubit_op_after) = ob.extract::<PyQubitOpAfterArg>() {
            Ok(Self(OpArg::QubitOpAfter(LineIndex(
                qubit_op_after.0 as u32,
            ))))
        } else if let Ok(const_f64) = ob.extract::<PyConstF64Arg>() {
            Ok(Self(OpArg::ConstF64(const_f64.0)))
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Invalid OpArg type",
            ))
        }
    }
}
