//! Rust-backed representation of circuits

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

use crate::ops::PyTk2Op;
use crate::rewrite::PyCircuitRewrite;
use crate::utils::ConvertPyErr;

use super::{cost, with_hugr, PyCircuitCost, PyCustom, PyHugrType, PyNode, PyWire};

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
    pub fn new(circ: &Bound<PyAny>) -> PyResult<Self> {
        Ok(Self {
            hugr: with_hugr(circ, |hugr, _| hugr)?,
        })
    }

    /// Convert the [`Tk2Circuit`] to a tket1 circuit.
    pub fn to_tket1<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
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
    pub fn circuit_cost<'py>(&self, cost_fn: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let py = cost_fn.py();
        let cost_fn = |op: &OpType| -> PyResult<PyCircuitCost> {
            let tk2_op: Tk2Op = op.try_into().map_err(|e| {
                PyErr::new::<PyValueError, _>(format!(
                    "Could not convert circuit operation to a `Tk2Op`: {e}"
                ))
            })?;
            let tk2_py_op = PyTk2Op::from(tk2_op);
            let cost = cost_fn.call1((tk2_py_op,))?;
            Ok(PyCircuitCost {
                cost: cost.to_object(py),
            })
        };
        let circ_cost = self.hugr.circuit_cost(cost_fn)?;
        Ok(circ_cost.cost.into_bound(py))
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
    pub fn __deepcopy__(&self, _memo: Bound<PyAny>) -> PyResult<Self> {
        Ok(self.clone())
    }

    fn node_op(&self, node: PyNode) -> PyResult<PyCustom> {
        let custom: CustomOp = self
            .hugr
            .get_optype(node.node)
            .clone()
            .try_into()
            .map_err(|e| {
                PyErr::new::<PyValueError, _>(format!(
                    "Could not convert circuit operation to a `CustomOp`: {e}"
                ))
            })?;

        Ok(custom.into())
    }

    fn node_inputs(&self, node: PyNode) -> Vec<PyWire> {
        self.hugr
            .all_linked_outputs(node.node)
            .map(|(n, p)| Wire::new(n, p).into())
            .collect()
    }

    fn node_outputs(&self, node: PyNode) -> Vec<PyWire> {
        self.hugr
            .node_outputs(node.node)
            .map(|p| Wire::new(node.node, p).into())
            .collect()
    }

    fn input_node(&self) -> PyNode {
        self.hugr.input().into()
    }

    fn output_node(&self) -> PyNode {
        self.hugr.output().into()
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

#[pyclass]
#[derive(Clone, Debug, PartialEq, From)]
pub(super) struct Dfg {
    /// Rust representation of the circuit.
    builder: DFGBuilder<Hugr>,
}
#[pymethods]
impl Dfg {
    #[new]
    fn new(input_types: Vec<PyHugrType>, output_types: Vec<PyHugrType>) -> PyResult<Self> {
        let builder = DFGBuilder::new(FunctionType::new(
            into_vec(input_types),
            into_vec(output_types),
        ))
        .convert_pyerrs()?;
        Ok(Self { builder })
    }

    fn inputs(&self) -> Vec<PyWire> {
        self.builder.input_wires().map_into().collect()
    }

    fn add_op(&mut self, op: PyCustom, inputs: Vec<PyWire>) -> PyResult<PyNode> {
        let custom: CustomOp = op.into();
        self.builder
            .add_dataflow_op(custom, inputs.into_iter().map_into())
            .convert_pyerrs()
            .map(|d| d.node().into())
    }

    fn finish(&mut self, outputs: Vec<PyWire>) -> PyResult<Tk2Circuit> {
        Ok(Tk2Circuit {
            hugr: self
                .builder
                .clone()
                .finish_hugr_with_outputs(outputs.into_iter().map_into(), &REGISTRY)
                .convert_pyerrs()?,
        })
    }
}

pub(super) fn into_vec<T, S: From<T>>(v: impl IntoIterator<Item = T>) -> Vec<S> {
    v.into_iter().map_into().collect()
}