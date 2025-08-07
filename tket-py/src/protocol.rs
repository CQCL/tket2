//! Python protocol implementations for CircuitMatcher and CircuitReplacer traits.

use derive_more::derive::Into;
use hugr::{hugr::views::SiblingSubgraph, HugrView};
use pyo3::{
    prelude::*,
    types::{PyDict, PyNone},
    BoundObject,
};
use tket::{
    rewrite::{
        matcher::{CircuitMatcher, MatchContext, MatchOutcome, OpArg},
        replacer::CircuitReplacer,
    },
    Circuit, TketOp,
};

use crate::{
    circuit::Tk2Circuit,
    matcher::{PyMatchOutcome, PyOpArg},
    ops::PyTketOp,
};

/// Rust wrapper for Python objects implementing the CircuitMatcher protocol.
#[derive(Debug, Clone, Into)]
pub struct PyCircuitMatcherImpl {
    py_obj: PyObject,
}

impl<'py> FromPyObject<'py> for PyCircuitMatcherImpl {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        Self::new(ob.to_owned().unbind())
    }
}

impl PyCircuitMatcherImpl {
    /// Create a new wrapper, validating that the Python object implements the required methods.
    pub fn new(py_obj: PyObject) -> PyResult<Self> {
        Python::with_gil(|py| {
            let obj = py_obj.bind(py);

            // Check that the object implements the required method
            if !obj.hasattr("match_tket_op")? {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "Object must implement match_tket_op method (CircuitMatcher protocol)",
                ));
            }

            // Optionally, we could check the method signature here
            // For now, we'll trust that it matches the protocol

            Ok(Self { py_obj })
        })
    }

    /// Call the Python object's match_tket_op method.
    ///
    /// Panics if the method is not available (should be checked at construction time).
    pub fn py_match_tket_op(
        &self,
        op: TketOp,
        op_args: &[OpArg],
        match_info: &Option<PyObject>,
    ) -> PyResult<PyMatchOutcome> {
        let py_op: PyTketOp = op.into();
        let py_args: Vec<PyOpArg> = op_args.iter().map(|&arg| arg.into()).collect();
        Python::with_gil(|py| {
            // Call the Python matcher's match_tket_op method
            let context = PyDict::new(py);
            context.set_item("match_info", match_info)?;
            // context.set_item("op_node", None::<PyNode>).unwrap();
            self.py_obj
                .bind(py)
                .call_method("match_tket_op", (py_op, py_args, context), None)?
                .extract()
        })
    }
}

impl CircuitMatcher for PyCircuitMatcherImpl {
    type PartialMatchInfo = Option<PyObject>;
    type MatchInfo = PyObject;

    fn match_tket_op(
        &self,
        op: TketOp,
        op_args: &[OpArg],
        match_context: MatchContext<Self::PartialMatchInfo, impl hugr::HugrView>,
    ) -> MatchOutcome<Self::PartialMatchInfo, Self::MatchInfo> {
        self.py_match_tket_op(op, op_args, &match_context.match_info)
            // TODO: handle python errors gracefully?
            .expect("could not call match_tket_op")
            .into()
    }
}

/// Rust wrapper for Python objects implementing the CircuitReplacer protocol.
#[derive(Debug, Clone)]
pub struct PyImplCircuitReplacer {
    py_obj: PyObject,
}

impl<'py> FromPyObject<'py> for PyImplCircuitReplacer {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        Self::new(ob.to_owned().unbind())
    }
}

impl PyImplCircuitReplacer {
    /// Create a new wrapper, validating that the Python object implements the required methods.
    pub fn new(py_obj: PyObject) -> PyResult<Self> {
        Python::with_gil(|py| {
            let obj = py_obj.bind(py);

            // Check that the object implements the required method
            if !obj.hasattr("replace_match")? {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "Object must implement replace_match method (CircuitReplacer protocol)",
                ));
            }

            Ok(Self { py_obj })
        })
    }

    /// Call the Python object's replace_match method.
    ///
    /// Panics if the method is not available (should be checked at construction time).
    pub fn py_replace_match<H: HugrView>(
        &self,
        subgraph: &SiblingSubgraph<H::Node>,
        hugr: &H,
        match_info: &PyObject,
    ) -> PyResult<Vec<Tk2Circuit>> {
        let subgraph_circuit = Circuit::new(subgraph.extract_subgraph(hugr, "subgraph"));
        Python::with_gil(|py| {
            // Call the Python replacement's replace_match method
            let result = self.py_obj.bind(py).call_method(
                "replace_match",
                (Tk2Circuit::from(subgraph_circuit), match_info),
                None,
            )?;
            // Extract Vec<Tk2Circuit>
            result.extract::<Vec<Tk2Circuit>>()
        })
    }
}

impl CircuitReplacer<PyObject> for PyImplCircuitReplacer {
    fn replace_match<H: hugr::HugrView>(
        &self,
        subgraph: &hugr::hugr::views::SiblingSubgraph<H::Node>,
        hugr: H,
        match_info: PyObject,
    ) -> Vec<Circuit> {
        self.py_replace_match(&subgraph, &hugr, &match_info)
            // TODO: handle python errors gracefully?
            .expect("could not call replace_match")
            .into_iter()
            .map(|tc| tc.circ)
            .collect()
    }
}

impl CircuitReplacer<()> for PyImplCircuitReplacer {
    fn replace_match<H: HugrView>(
        &self,
        subgraph: &SiblingSubgraph<H::Node>,
        hugr: H,
        (): (),
    ) -> Vec<Circuit> {
        let match_info = Python::with_gil(|py| PyNone::get(py).into_any().into());
        self.py_replace_match(&subgraph, &hugr, &match_info)
            // TODO: handle python errors gracefully?
            .expect("could not call replace_match")
            .into_iter()
            .map(|tc| tc.circ)
            .collect()
    }
}
