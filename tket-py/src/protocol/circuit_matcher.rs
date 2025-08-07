//! Circuit matcher protocol implementation.

use derive_more::derive::Into;
use hugr::HugrView;
// use itertools::Itertools;
use pyo3::prelude::*;
use tket::{
    rewrite::matcher::{CircuitMatcher, MatchContext, MatchOutcome},
    TketOp,
};

use crate::{
    matcher::{PyCircuitUnit, PyMatchContext, PyMatchOutcome},
    ops::PyTketOp,
};

/// Python protocol for [`CircuitMatcher`] trait.
pub trait CircuitMatcherPyProtocol {
    /// Wrapper for the CircuitMatcher protocol's `match_tket_op` method
    fn py_match_tket_op(
        &self,
        op: TketOp,
        op_args: Vec<PyCircuitUnit>,
        context: PyMatchContext,
    ) -> PyResult<PyMatchOutcome>;
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
                let args = args
                    .iter()
                    .map(|arg| PyCircuitUnit::with_context(arg.clone(), &match_context))
                    .collect_vec();
                let context =
                    Python::with_gil(|py| PyMatchContext::from_match_context(match_context, py))
                        .map_err(|e| panic!("A python error occurred:\n{}", e))
                        .unwrap();
                match <$M as CircuitMatcherPyProtocol>::py_match_tket_op(self, op, args, context) {
                    Ok(outcome) => outcome.into(),
                    Err(err) => panic!("A python error occurred:\n{}", err),
                }
            }
        }
    };
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
        op: TketOp,
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
