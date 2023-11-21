//! Utilities for calling Hugr functions on generic python objects.

use hugr::ops::OpType;
use pyo3::exceptions::{PyAttributeError, PyValueError};
use pyo3::{prelude::*, PyTypeInfo};

use derive_more::From;
use hugr::{Hugr, HugrView};
use serde::Serialize;
use tket2::circuit::CircuitHash;
use tket2::extension::REGISTRY;
use tket2::json::TKETDecode;
use tket2::passes::CircuitChunks;
use tket2::{Circuit, Tk2Op};
use tket_json_rs::circuit_json::SerialCircuit;

use crate::rewrite::PyCircuitRewrite;

use super::{cost, PyCircuitCost};

/// A circuit in tket2 format.
///
/// This can be freely converted to and from a `pytket.Circuit`. Prefer using
/// this class when applying multiple tket2 operations on a circuit, as it
/// avoids the overhead of converting to and from a `pytket.Circuit` each time.
///
/// Node indices returned by this class are not stable across conversion to and
/// from a `pytket.Circuit`.
///
/// # Examples
///
/// Convert between `pytket.Circuit`s and `Tk2Circuit`s:
/// ```python
/// from pytket import Circuit
/// c = Circuit(2).H(0).CX(0, 1)
/// # Convert to a Tk2Circuit
/// t2c = Tk2Circuit(c)
/// # Convert back to a pytket.Circuit
/// c2 = t2c.to_tket1()
/// ```
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
    pub fn apply_rewrite(&mut self, rw: PyCircuitRewrite) {
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

    /// Compute the cost of the circuit based on a per-operation cost function.
    ///
    /// :param cost_fn: A function that takes a `Tk2Op` and returns an arbitrary cost.
    ///     The cost must implement `__add__`.
    /// :returns: The sum of all operation costs.
    pub fn circuit_cost<'py>(&self, cost_fn: &'py PyAny) -> PyResult<&'py PyAny> {
        let py = cost_fn.py();
        let cost_fn = |op: &OpType| -> PyResult<PyCircuitCost> {
            let tk2_op: Tk2Op = op.try_into().map_err(|e| {
                PyErr::new::<PyValueError, _>(format!(
                    "Could not convert circuit operation to a `Tk2Op`: {e}"
                ))
            })?;
            let cost = cost_fn.call1((tk2_op,))?;
            Ok(PyCircuitCost {
                cost: cost.to_object(py),
            })
        };
        let circ_cost = self.hugr.circuit_cost(cost_fn)?;
        Ok(circ_cost.cost.into_ref(py))
    }

    /// Returns a hash of the circuit.
    pub fn hash(&self) -> u64 {
        self.hugr.circuit_hash().unwrap()
    }

    /// Hash the circuit
    pub fn __hash__(&self) -> isize {
        self.hash() as isize
    }

    /// Copy the circuit.
    pub fn __copy__(&self) -> PyResult<Self> {
        Ok(self.clone())
    }

    /// Copy the circuit.
    pub fn __deepcopy__(&self, _memo: Py<PyAny>) -> PyResult<Self> {
        Ok(self.clone())
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
