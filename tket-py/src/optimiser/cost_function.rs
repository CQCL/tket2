use std::sync::Arc;

use hugr::extension::simple_op::MakeExtensionOp;
use hugr::ops::OpType;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use tket::rewrite::strategy::{ExhaustiveGreedyStrategy, LexicographicCostFunction};
use tket::TketOp;

use crate::ops::{PyExtensionOp, PyTketOp};

/// The cost function to use for the Badger optimiser.
#[derive(Clone, Default)]
pub enum BadgerCostFunction {
    /// Minimise CX count.
    #[default]
    CXCount,
    /// Minimise Rz count.
    RzCount,
    /// Custom cost function.
    Custom(Arc<dyn Fn(&OpType) -> usize + Send + Sync>),
}

// type BoxedBadgerStrategy =
//     ExhaustiveGreedyStrategy<LexicographicCostFunction<Box<dyn Fn(&OpType) ->
// usize>, 2>>;

pub(super) type PyBadgerStrategy = ExhaustiveGreedyStrategy<
    LexicographicCostFunction<Arc<dyn Fn(&OpType) -> usize + Send + Sync>, 2>,
>;

impl BadgerCostFunction {
    pub(super) fn into_strategy(self) -> PyBadgerStrategy {
        let cost_fn = match self {
            BadgerCostFunction::CXCount => LexicographicCostFunction::cx_count().into(),
            BadgerCostFunction::RzCount => LexicographicCostFunction::rz_count().into(),
            BadgerCostFunction::Custom(cost_fn) => LexicographicCostFunction::from_cost_fn(cost_fn),
        };
        cost_fn.into_greedy_strategy()
    }
}

impl<'py> FromPyObject<'py> for BadgerCostFunction {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(str) = ob.extract::<&str>() {
            return match str {
                "cx" => Ok(BadgerCostFunction::CXCount),
                "rz" => Ok(BadgerCostFunction::RzCount),
                _ => Err(PyErr::new::<PyValueError, _>(format!(
                    "Invalid cost function: {str}. Expected 'cx' or 'rz'."
                ))),
            };
        }
        // not a string, interpret it as a callable
        let cost_fn = ob.to_owned().unbind();
        Ok(BadgerCostFunction::Custom(Arc::new(move |op| {
            let Some(ext_op) = op.as_extension_op() else {
                return 0;
            };
            if let Ok(tket_op) = TketOp::from_extension_op(ext_op) {
                Python::with_gil(|py| {
                    cost_fn
                        .call1(py, (PyTketOp::from(tket_op),))
                        .and_then(|v| v.extract(py))
                })
                .unwrap()
            } else {
                Python::with_gil(|py| {
                    cost_fn
                        .call1(py, (PyExtensionOp::from(ext_op.clone()),))
                        .and_then(|v| v.extract(py))
                })
                .unwrap()
            }
        })))
    }
}
