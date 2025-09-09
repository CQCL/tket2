//! Circuit matcher protocol implementation.

use derive_more::derive::Into;
use hugr::{hugr::views::sibling_subgraph::InvalidSubgraph, HugrView};
// use itertools::Itertools;
use pyo3::{exceptions::PyValueError, prelude::*};
use tket::{
    rewrite::matcher::{CircuitMatcher, MatchContext, MatchOutcome, Update},
    TketOp,
};

use crate::{
    matcher::{PyCircuitUnit, PyMatchContext, PyMatchOutcome},
    ops::{PyExtensionOp, PyTketOp},
};

/// Python protocol for [`CircuitMatcher`] trait.
pub trait CircuitMatcherPyProtocol {
    /// Wrapper for the CircuitMatcher protocol's `match_tket_op` method
    fn py_match_tket_op(
        &self,
        op: PyTketOp,
        op_args: Vec<PyCircuitUnit>,
        context: PyMatchContext,
    ) -> PyResult<PyMatchOutcome>;

    /// Wrapper for the CircuitMatcher protocol's `match_extension_op` method
    ///
    /// By default, this method causes any unknown extension operations to be
    /// skipped.
    fn py_match_extension_op(
        &self,
        _op: PyExtensionOp,
        _inputs: Vec<PyCircuitUnit>,
        _outputs: Vec<PyCircuitUnit>,
        _context: PyMatchContext,
    ) -> PyResult<PyMatchOutcome> {
        Ok(MatchOutcome::default().skip(Update::Unchanged).into())
    }
}

/// Implement CircuitMatcher for a type that implements PyCircuitMatcher.
#[macro_export]
macro_rules! impl_circuit_matcher {
    ($M:ty) => {
        use itertools::Itertools;
        use tket::resource::CircuitUnit;

        impl CircuitMatcher for $M {
            type PartialMatchInfo = Option<PyObject>;

            type MatchInfo = PyObject;

            fn match_tket_op<H: HugrView>(
                &self,
                op: TketOp,
                args: &[CircuitUnit<H::Node>],
                match_context: MatchContext<Self::PartialMatchInfo, H>,
            ) -> MatchOutcome<Self::PartialMatchInfo, Self::MatchInfo> {
                match try_match_tket_op(self, op, args, match_context) {
                    Ok(outcome) => outcome,
                    Err(err) => {
                        eprintln!("Warning: could not match op {op:?}: {err}");
                        MatchOutcome::default().skip(Update::Unchanged)
                    }
                }
            }
        }
    };
}

fn try_match_tket_op<H: HugrView>(
    matcher: &impl CircuitMatcherPyProtocol,
    op: TketOp,
    args: &[CircuitUnit<H::Node>],
    match_context: MatchContext<Option<PyObject>, H>,
) -> Result<MatchOutcome<Option<PyObject>, PyObject>, PyErr> {
    let args = args
        .iter()
        .map(|arg| PyCircuitUnit::with_context(arg.clone(), &match_context))
        .collect_vec();
    let context_res =
        Python::with_gil(|py| PyMatchContext::try_from_match_context(match_context, py));
    let context = match context_res {
        Ok(context) => context,
        Err(InvalidSubgraph::NotConvex) => {
            // Our match is non-convex, discard
            return Ok(MatchOutcome::default().skip(Update::<Option<PyObject>>::Unchanged));
        }
        Err(err) => {
            return Err(PyValueError::new_err(format!(
                "could not create match context: {err}"
            )))
        }
    };
    matcher
        .py_match_tket_op(op.into(), args, context)
        .map(Into::into)
}

/// Rust wrapper for Python objects implementing the CircuitMatcher protocol.
#[derive(Debug, Clone, Into)]
pub struct PyImplCircuitMatcher {
    py_obj: PyObject,
}

impl CircuitMatcherPyProtocol for PyImplCircuitMatcher {
    /// Call the Python object's match_tket_op method.
    ///
    /// Panics if the method is not available (should be checked at construction
    /// time).
    fn py_match_tket_op(
        &self,
        op: PyTketOp,
        args: Vec<PyCircuitUnit>,
        context: PyMatchContext,
    ) -> PyResult<PyMatchOutcome> {
        let py_op: PyTketOp = op.into();
        Python::with_gil(|py| {
            self.py_obj
                .bind(py)
                .call_method("match_tket_op", (py_op, args, context), None)?
                .extract()
        })
    }

    fn py_match_extension_op(
        &self,
        op: PyExtensionOp,
        inputs: Vec<PyCircuitUnit>,
        outputs: Vec<PyCircuitUnit>,
        context: PyMatchContext,
    ) -> PyResult<PyMatchOutcome> {
        match Python::with_gil(|py| {
            self.py_obj
                .bind(py)
                .call_method("match_extension_op", (op, inputs, outputs, context), None)
                .and_then(|v| v.extract())
        }) {
            Ok(py_match_outcome) => Ok(py_match_outcome),
            Err(_) => Ok(MatchOutcome::default().skip(Update::Unchanged).into()),
        }
    }
}

impl_circuit_matcher!(PyImplCircuitMatcher);

impl<'py> FromPyObject<'py> for PyImplCircuitMatcher {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        Self::new(ob.to_owned().unbind())
    }
}

impl PyImplCircuitMatcher {
    /// Create a new wrapper, validating that the Python object implements the
    /// required methods.
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
}
