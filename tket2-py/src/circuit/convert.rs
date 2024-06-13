//! Utilities for calling Hugr functions on generic python objects.

use std::borrow::Borrow;

use hugr::builder::{CircuitBuilder, DFGBuilder, Dataflow, DataflowHugr};
use hugr::extension::prelude::QB_T;
use hugr::ops::handle::NodeHandle;
use hugr::ops::{CustomOp, OpType};
use hugr::types::{FunctionType, Type};
use itertools::Itertools;
use pyo3::exceptions::{PyAttributeError, PyValueError};
use pyo3::types::{PyAnyMethods, PyModule, PyString, PyTypeMethods};
use pyo3::{
    pyclass, pymethods, Bound, FromPyObject, PyAny, PyErr, PyObject, PyRefMut, PyResult,
    PyTypeInfo, Python, ToPyObject,
};

use derive_more::From;
use hugr::{Hugr, HugrView, Wire};
use serde::Serialize;
use tket2::circuit::CircuitHash;
use tket2::extension::REGISTRY;
use tket2::passes::CircuitChunks;
use tket2::serialize::TKETDecode;
use tket2::{Circuit, Tk2Op};
use tket_json_rs::circuit_json::SerialCircuit;

use crate::rewrite::PyCircuitRewrite;
use crate::utils::ConvertPyErr;

use super::{cost, PyCircuitCost, PyCustom, PyHugrType, PyNode, PyWire, Tk2Circuit};

/// A flag to indicate the encoding of a circuit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum CircuitType {
    /// A `pytket` `Circuit`.
    Tket1,
    /// A tket2 `Tk2Circuit`, represented as a HUGR.
    Tket2,
}

impl CircuitType {
    /// Converts a `Hugr` into the format indicated by the flag.
    pub fn convert(self, py: Python, hugr: Hugr) -> PyResult<Bound<PyAny>> {
        match self {
            CircuitType::Tket1 => SerialCircuit::encode(&hugr.into())
                .convert_pyerrs()?
                .to_tket1(py),
            CircuitType::Tket2 => Ok(Bound::new(py, Tk2Circuit { circ: hugr.into() })?.into_any()),
        }
    }
}

/// Apply a fallible function expecting a hugr on a python circuit.
///
/// This method supports both `pytket.Circuit` and `Tk2Circuit` python objects.
pub fn try_with_hugr<T, E, F>(circ: &Bound<PyAny>, f: F) -> PyResult<T>
where
    E: ConvertPyErr<Output = PyErr>,
    F: FnOnce(Hugr, CircuitType) -> Result<T, E>,
{
    let (circ, typ) = match Tk2Circuit::extract_bound(circ) {
        // hugr circuit
        Ok(t2circ) => (t2circ.circ, CircuitType::Tket2),
        // tket1 circuit
        Err(_) => (
            SerialCircuit::from_tket1(circ)?.decode().convert_pyerrs()?,
            CircuitType::Tket1,
        ),
    };
    (f)(circ.into_hugr(), typ).map_err(|e| e.convert_pyerrs())
}

/// Apply a function expecting a hugr on a python circuit.
///
/// This method supports both `pytket.Circuit` and `Tk2Circuit` python objects.
pub fn with_hugr<T, F>(circ: &Bound<PyAny>, f: F) -> PyResult<T>
where
    F: FnOnce(Hugr, CircuitType) -> T,
{
    try_with_hugr(circ, |hugr, typ| Ok::<T, PyErr>((f)(hugr, typ)))
}

/// Apply a fallible hugr-to-hugr function on a python circuit, and return the modified circuit.
///
/// This method supports both `pytket.Circuit` and `Tk2Circuit` python objects.
/// The returned Hugr is converted to the matching python object.
pub fn try_update_hugr<'py, E, F>(circ: &Bound<'py, PyAny>, f: F) -> PyResult<Bound<'py, PyAny>>
where
    E: ConvertPyErr<Output = PyErr>,
    F: FnOnce(Hugr, CircuitType) -> Result<Hugr, E>,
{
    let py = circ.py();
    try_with_hugr(circ, |hugr, typ| {
        let hugr = f(hugr, typ).map_err(|e| e.convert_pyerrs())?;
        typ.convert(py, hugr)
    })
}

/// Apply a hugr-to-hugr function on a python circuit, and return the modified circuit.
///
/// This method supports both `pytket.Circuit` and `Tk2Circuit` python objects.
/// The returned Hugr is converted to the matching python object.
pub fn update_hugr<'py, F>(circ: &Bound<'py, PyAny>, f: F) -> PyResult<Bound<'py, PyAny>>
where
    F: FnOnce(Hugr, CircuitType) -> Hugr,
{
    let py = circ.py();
    try_with_hugr(circ, |hugr, typ| {
        let hugr = f(hugr, typ);
        typ.convert(py, hugr)
    })
}
