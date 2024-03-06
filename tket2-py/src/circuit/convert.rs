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
use tket2::json::TKETDecode;
use tket2::passes::CircuitChunks;
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
    /// A `guppylang.module.GuppyModule` or `guppylang.hugr.Hugr`.
    ///
    /// This a pure python structure, and requires a conversion to a
    /// `Tk2Circuit` before using it in the Rust backend.
    ///
    /// Re-encoding is not supported, modifying a `Guppy` object will return it
    /// as a `Tk2Circuit`.
    Guppy,
    // TODO: Support hugr-py's `SerialHugr`.
}

impl CircuitType {
    /// Converts a `Hugr` into the format indicated by the flag.
    ///
    /// If the flag is `Guppy`, the `Hugr` is converted to a `Tk2Circuit`.
    pub fn convert(self, py: Python, hugr: Hugr) -> PyResult<Bound<PyAny>> {
        match self {
            CircuitType::Tket1 => SerialCircuit::encode(&hugr).convert_pyerrs()?.to_tket1(py),
            CircuitType::Tket2 => Ok(Bound::new(py, Tk2Circuit { hugr })?.into_any()),
            CircuitType::Guppy => CircuitType::Tket2.convert(py, hugr),
        }
    }
}

/// Extract a `Hugr` from a python object, and return it along the original encoding tag.
fn extract_hugr(circ: &Bound<PyAny>) -> PyResult<(Hugr, CircuitType)> {
    if let Ok(t2circ) = Tk2Circuit::extract_bound(circ) {
        return Ok((t2circ.hugr, CircuitType::Tket2));
    }

    if let Some(hugr) = try_extract_guppy(circ)? {
        return Ok((hugr, CircuitType::Guppy));
    }

    if let Some(hugr) = try_extract_tk1(circ)? {
        return Ok((hugr, CircuitType::Tket1));
    }

    let input_type = circ.get_type();
    let msg = format!(
        "Unsupported circuit type. Expected a `Tk2Circuit`, a `pytket.Circuit`, or a guppylang definition, got {}",
        input_type.name()?
    );
    Err(PyErr::new::<PyValueError, _>(msg))
}

/// Try extracting a guppy-defined circuit into a `Hugr`.
/// If the input is not a guppy circuit or if `guppylang` is not installed, returns `None`.
fn try_extract_guppy(circ: &Bound<PyAny>) -> PyResult<Option<Hugr>> {
    let py = circ.py();
    if py.import_bound("guppylang").is_err() {
        /// The `guppylang` library is not installed.
        return Ok(None);
    };

    let guppy_fn =
        PyModule::import_bound(py, "guppylang.definition.function")?.getattr("RawFunctionDef")?;
    let guppy_module = py
        .import_bound("guppylang.module")?
        .getattr("GuppyModule")?;

    let is_guppy_module = circ.is_instance(&guppy_module)?;
    let is_guppy_fn = circ.is_instance(&guppy_fn)?;

    // Extract the compiled module, if possible.
    let circ = match (is_guppy_module, is_guppy_fn) {
        // Module
        (true, _) => circ.call_method0("compile")?,
        // Function
        //
        // TODO: This compiles the whole module, not just the function.
        // We should take a subhugr here.
        (_, true) => circ
            .getattr("id")?
            .getattr("module")?
            .call_method0("compile")?,
        // Not a guppy object
        (false, false) => return Ok(None),
    };

    // Convert the guppy Hugr to a Tk2Circuit.
    let json = circ.call_method0("serialize")?.extract::<&PyString>()?;
    let hugr = serde_json::from_str(&json.to_string_lossy()).map_err(|e| {
        PyErr::new::<PyAttributeError, _>(format!(
            "Failed to convert the Guppy circuit into a Tk2Circuit. {e}"
        ))
    })?;

    // TODO: The compiled Guppy is a Module root, which is not a valid Circuit.
    // We need guppy to output DFGs instead.
    Ok(Some(hugr))
}

/// Try extracting a tket1 circuit into a `Hugr`.
/// If the input is not a tket1 circuit or if `pytket` is not installed, returns `None`.
fn try_extract_tk1(circ: &Bound<PyAny>) -> PyResult<Option<Hugr>> {
    let py = circ.py();
    if py.import_bound("pytket").is_err() {
        /// The `guppylang` library is not installed.
        return Ok(None);
    };
    let pytket_circ = PyModule::import_bound(py, "pytket.circuit")?.getattr("Circuit")?;

    if !circ.is_instance(&pytket_circ)? {
        return Ok(None);
    }

    let serial_circ = SerialCircuit::from_tket1(circ)?;
    Ok(Some(serial_circ.decode().convert_pyerrs()?))
}

/// Apply a fallible function expecting a hugr on a python circuit.
///
/// This method supports both `pytket.Circuit` and `Tk2Circuit` python objects.
pub fn try_with_hugr<T, E, F>(circ: &Bound<PyAny>, f: F) -> PyResult<T>
where
    E: ConvertPyErr<Output = PyErr>,
    F: FnOnce(Hugr, CircuitType) -> Result<T, E>,
{
    let (hugr, typ) = extract_hugr(circ)?;
    (f)(hugr, typ).map_err(|e| e.convert_pyerrs())
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
