//! Utilities for calling Hugr functions on generic python objects.
//!
//! Supports both `pytket.Circuit` and `Tk2Circuit` python objects.

use hugr::ops::OpType;
use pyo3::exceptions::{PyAttributeError, PyValueError};
use pyo3::types::PyString;
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
use crate::utils::ConvertPyErr;

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
    /// Initialize a `Tk2Circuit` from a `pytket.Circuit` or `guppy.Hugr`.
    ///
    /// Converts the input circuit to a `Hugr` if required via its serialisation
    /// interface.
    #[new]
    pub fn new(circ: &PyAny) -> PyResult<Self> {
        Ok(Self {
            hugr: with_hugr(circ, |hugr, _| hugr)?,
        })
    }

    /// Convert the `Tk2Circuit` to a tket1 circuit.
    pub fn to_tket1<'py>(&self, py: Python<'py>) -> PyResult<&'py PyAny> {
        SerialCircuit::encode(&self.hugr)
            .convert_pyerrs()?
            .to_tket1(py)
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
    pub fn to_tket1_json(&self) -> PyResult<String> {
        Ok(serde_json::to_string(&SerialCircuit::encode(&self.hugr).convert_pyerrs()?).unwrap())
    }

    /// Decode a tket1 json string to a circuit.
    #[staticmethod]
    pub fn from_tket1_json(json: &str) -> PyResult<Self> {
        let tk1: SerialCircuit = serde_json::from_str(json)
            .map_err(|e| PyErr::new::<PyAttributeError, _>(format!("Invalid encoded HUGR: {e}")))?;
        Ok(Tk2Circuit {
            hugr: tk1.decode().convert_pyerrs()?,
        })
    }

    /// Compute the cost of the circuit based on a per-operation cost function.
    ///
    /// :param cost_fn: A function that takes a `Tk2Op` and returns an arbitrary cost.
    ///     The cost must implement `__add__`, `__sub__`, `__lt__`,
    ///     `__eq__`, `__int__`, and integer `__div__`.
    ///
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
}

impl CircuitType {
    /// Converts a `Hugr` into the format indicated by the flag.
    ///
    /// If the flag is `Guppy`, the `Hugr` is converted to a `Tk2Circuit`.
    pub fn convert(self, py: Python, hugr: Hugr) -> PyResult<&PyAny> {
        match self {
            CircuitType::Tket1 => SerialCircuit::encode(&hugr).convert_pyerrs()?.to_tket1(py),
            CircuitType::Tket2 => Ok(Py::new(py, Tk2Circuit { hugr })?.into_ref(py)),
            CircuitType::Guppy => CircuitType::Tket2.convert(py, hugr),
        }
    }
}

/// Extract a `Hugr` from a python object, and return it along the original encoding tag.
fn extract_hugr(circ: &PyAny) -> PyResult<(Hugr, CircuitType)> {
    if let Ok(t2circ) = Tk2Circuit::extract(circ) {
        return Ok((t2circ.hugr, CircuitType::Tket2));
    }

    let py = circ.py();
    let guppy_hugr = py.import("guppylang.hugr.Hugr")?;
    let guppy_module = py.import("guppylang.module.GuppyModule")?;

    let is_guppy_module = circ.is_instance(guppy_module)?;
    let is_guppy_hugr = circ.is_instance(guppy_hugr)?;

    if !is_guppy_module && !is_guppy_hugr {
        // Decode it as a tket1 circuit.
        return Ok((
            SerialCircuit::from_tket1(circ)?.decode().convert_pyerrs()?,
            CircuitType::Tket1,
        ));
    }

    // Ensure we are working with the compiled circuit.
    let circ = match is_guppy_module {
        true => circ.call_method0("compile")?,
        false => circ,
    };

    // Convert the guppy Hugr to a Tk2Circuit.
    let json = guppy_hugr
        .call_method0("serialize")?
        .extract::<&PyString>()?;
    let hugr = serde_json::from_str(&json.to_string_lossy()).map_err(|e| {
        PyErr::new::<PyAttributeError, _>(format!(
            "Failed to convert the Guppy circuit into a Tk2Circuit. {e}"
        ))
    })?;

    // TODO: The compiled Guppy is a Module root, which is not a valid Circuit.
    // We need guppy to output DFGs instead.

    Ok((hugr, CircuitType::Guppy))
}

/// Apply a fallible function expecting a hugr on a python circuit.
///
/// This method supports both `pytket.Circuit` and `Tk2Circuit` python objects.
pub fn try_with_hugr<T, E, F>(circ: &PyAny, f: F) -> PyResult<T>
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
