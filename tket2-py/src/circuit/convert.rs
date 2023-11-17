//! Utilities for calling Hugr functions on generic python objects.

use pyo3::exceptions::PyAttributeError;
use pyo3::{prelude::*, PyTypeInfo};

use derive_more::From;
use hugr::{Hugr, HugrView};
use serde::Serialize;
use tket2::circuit::CircuitHash;
use tket2::extension::REGISTRY;
use tket2::json::TKETDecode;
use tket2::passes::CircuitChunks;
use tket_json_rs::circuit_json::SerialCircuit;

use crate::pattern::rewrite::PyCircuitRewrite;

/// A manager for tket 2 operations on a tket 1 Circuit.
#[pyclass]
#[derive(Clone, Debug, PartialEq, From)]
pub struct Tk2Circuit {
    /// Rust representation of the circuit.
    pub hugr: Hugr,
}

#[pymethods]
impl Tk2Circuit {
    /// Convert a tket1 circuit to a [`Tk2Circuit`].
    #[new]
    pub fn from_tket1(circ: &PyAny) -> PyResult<Self> {
        Ok(Self {
            hugr: with_hugr(circ, |hugr, _| hugr)?,
        })
    }

    /// Convert the [`Tk2Circuit`] to a tket1 circuit.
    pub fn to_tket1<'py>(&self, py: Python<'py>) -> PyResult<&'py PyAny> {
        SerialCircuit::encode(&self.hugr)?.to_tket1(py)
    }

    /// Apply a rewrite on the circuit.
    pub fn apply_match(&mut self, rw: PyCircuitRewrite) {
        rw.rewrite.apply(&mut self.hugr).expect("Apply error.");
    }

    /// Encode the circuit as a HUGR json string.
    //
    // TODO: Bind a messagepack encoder/decoder too.
    pub fn to_hugr_json(&self) -> PyResult<String> {
        Ok(serde_json::to_string(&self.hugr).unwrap())
    }

    /// Decode a HUGR json string to a circuit.
    #[staticmethod]
    pub fn from_hugr_json(json: &str) -> PyResult<Self> {
        let hugr = serde_json::from_str(json)
            .map_err(|e| PyErr::new::<PyAttributeError, _>(format!("Invalid encoded HUGR: {e}")))?;
        Ok(Tk2Circuit { hugr })
    }

    /// Encode the circuit as a tket1 json string.
    ///
    /// FIXME: Currently the encoded circuit cannot be loaded back due to
    /// [https://github.com/CQCL/hugr/issues/683]
    pub fn to_tket1_json(&self) -> PyResult<String> {
        Ok(serde_json::to_string(&SerialCircuit::encode(&self.hugr)?).unwrap())
    }

    /// Decode a tket1 json string to a circuit.
    #[staticmethod]
    pub fn from_tket1_json(json: &str) -> PyResult<Self> {
        let tk1: SerialCircuit = serde_json::from_str(json)
            .map_err(|e| PyErr::new::<PyAttributeError, _>(format!("Invalid encoded HUGR: {e}")))?;
        Ok(Tk2Circuit {
            hugr: tk1.decode()?,
        })
    }

    /// Returns a hash of the circuit.
    pub fn hash(&self) -> u64 {
        self.hugr.circuit_hash()
    }

    /// Hash the circuit
    pub fn __hash__(&self) -> isize {
        self.hash() as isize
    }
}
impl Tk2Circuit {
    /// Tries to extract a Tk2Circuit from a python object.
    ///
    /// Returns an error if the py object is not a Tk2Circuit.
    pub fn try_extract(circ: &PyAny) -> PyResult<Self> {
        circ.extract::<Tk2Circuit>()
    }
}

/// A flag to indicate the encoding of a circuit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CircuitType {
    /// A `pytket` `Circuit`.
    Tket1,
    /// A tket2 `Tk2Circuit`, represented as a HUGR.
    Tket2,
}

impl CircuitType {
    /// Converts a `Hugr` into the format indicated by the flag.
    pub fn convert(self, py: Python, hugr: Hugr) -> PyResult<&PyAny> {
        match self {
            CircuitType::Tket1 => SerialCircuit::encode(&hugr)?.to_tket1(py),
            CircuitType::Tket2 => Ok(Py::new(py, Tk2Circuit { hugr })?.into_ref(py)),
        }
    }
}

/// Apply a fallible function expecting a hugr on a python circuit.
///
/// This method supports both `pytket.Circuit` and `Tk2Circuit` python objects.
pub fn try_with_hugr<T, E, F>(circ: &PyAny, f: F) -> PyResult<T>
where
    E: Into<PyErr>,
    F: FnOnce(Hugr, CircuitType) -> Result<T, E>,
{
    let (hugr, typ) = match Tk2Circuit::extract(circ) {
        // hugr circuit
        Ok(t2circ) => (t2circ.hugr, CircuitType::Tket2),
        // tket1 circuit
        Err(_) => (
            SerialCircuit::from_tket1(circ)?.decode()?,
            CircuitType::Tket1,
        ),
    };
    (f)(hugr, typ).map_err(|e| e.into())
}

/// Apply a function expecting a hugr on a python circuit.
///
/// This method supports both `pytket.Circuit` and `Tk2Circuit` python objects.
pub fn with_hugr<T, F>(circ: &PyAny, f: F) -> PyResult<T>
where
    F: FnOnce(Hugr, CircuitType) -> T,
{
    try_with_hugr(circ, |hugr, typ| Ok::<T, PyErr>((f)(hugr, typ)))
}

/// Apply a fallible hugr-to-hugr function on a python circuit, and return the modified circuit.
///
/// This method supports both `pytket.Circuit` and `Tk2Circuit` python objects.
/// The returned Hugr is converted to the matching python object.
pub fn try_update_hugr<E, F>(circ: &PyAny, f: F) -> PyResult<&PyAny>
where
    E: Into<PyErr>,
    F: FnOnce(Hugr, CircuitType) -> Result<Hugr, E>,
{
    let py = circ.py();
    try_with_hugr(circ, |hugr, typ| {
        let hugr = f(hugr, typ).map_err(|e| e.into())?;
        typ.convert(py, hugr)
    })
}

/// Apply a hugr-to-hugr function on a python circuit, and return the modified circuit.
///
/// This method supports both `pytket.Circuit` and `Tk2Circuit` python objects.
/// The returned Hugr is converted to the matching python object.
pub fn update_hugr<F>(circ: &PyAny, f: F) -> PyResult<&PyAny>
where
    F: FnOnce(Hugr, CircuitType) -> Hugr,
{
    let py = circ.py();
    try_with_hugr(circ, |hugr, typ| {
        let hugr = f(hugr, typ);
        typ.convert(py, hugr)
    })
}
