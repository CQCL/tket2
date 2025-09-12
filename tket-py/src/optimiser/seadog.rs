use std::path::PathBuf;

use derive_more::derive::From;
use pyo3::prelude::*;
use tket::optimiser::seadog::{SeadogOptimiser, SeadogOptions};
use tket::Circuit;

use super::{BadgerCostFunction, PyBadgerStrategy};
use crate::circuit::update_circ;
use crate::rewrite::PyRewriter;

/// Wrapped [`DefaultBadgerOptimiser`].
///
/// Currently only exposes loading from an ECC file using the constructor
/// and optimising using default logging settings.
#[pyclass(name = "SeadogOptimiser")]
#[derive(From, Clone)]
pub struct PySeadogOptimiser(SeadogOptimiser<PyRewriter, PyBadgerStrategy>);

#[pymethods]
impl PySeadogOptimiser {
    /// Create a new [`PySeadogOptimiser`] from a rewriter and cost function.
    #[new]
    #[pyo3(signature = (rewriter, cost_fn=None))]
    pub fn new(rewriter: PyRewriter, cost_fn: Option<BadgerCostFunction>) -> Self {
        SeadogOptimiser::new(rewriter, cost_fn.unwrap_or_default().into_strategy()).into()
    }

    /// Run the optimiser on a circuit.
    ///
    /// Returns an optimised circuit and optionally log the progress to a CSV
    /// file.
    ///
    /// # Parameters
    ///
    /// * `circ`: The circuit to optimise.
    ///
    /// * `timeout`: The maximum time (in seconds) to run the optimiser.
    ///
    ///     If `None` the optimiser will run indefinitely, or until
    ///     `progress_timeout` is reached.
    ///
    /// * `progress_timeout`: The maximum time (in seconds) to search for new
    ///   improvements to the circuit. If no progress is made in this time, the
    ///   optimiser will stop.
    ///
    ///     If `None` the optimiser will run indefinitely, or until `timeout` is
    ///     reached.
    ///
    /// * `max_circuit_count`: The maximum number of circuits to process before
    ///   stopping the optimisation.
    ///
    ///
    ///     For data parallel multi-threading, (split_circuit=true), applies on
    ///     a per-thread basis, otherwise applies globally.
    ///
    ///     If `None` the optimiser will run indefinitely, or until `timeout` is
    ///     reached.
    ///
    /// * `queue_size`: The maximum size of the circuit candidates priority
    ///   queue. Defaults to `20`.
    #[pyo3(name = "optimise")]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (circ, timeout=None, progress_timeout=None, max_circuit_count=None, queue_size=None, save_rewrite_space=None))]
    pub fn py_optimise<'py>(
        &self,
        circ: &Bound<'py, PyAny>,
        timeout: Option<u64>,
        progress_timeout: Option<u64>,
        max_circuit_count: Option<usize>,
        queue_size: Option<usize>,
        save_rewrite_space: Option<PathBuf>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let options = SeadogOptions {
            timeout,
            progress_timeout,
            max_circuit_count,
            queue_size: queue_size.unwrap_or(100),
            save_rewrite_space,
            ..SeadogOptions::default()
        };
        update_circ(circ, |circ, _| self.optimise(circ, options))
    }
}

impl PySeadogOptimiser {
    /// The Python optimise method, but on Hugrs.
    pub(crate) fn optimise(&self, circ: Circuit, options: SeadogOptions) -> Circuit {
        self.0.optimise(&circ, options)
    }
}
