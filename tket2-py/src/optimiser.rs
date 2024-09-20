//! PyO3 wrapper for the Badger circuit optimiser.

use std::collections::BTreeSet;
use std::io::BufWriter;
use std::{fs, num::NonZeroUsize, path::PathBuf};

use itertools::Itertools;
use portdiff::port_diff::{EdgeData, PortDiffData};
use portdiff::{PortDiff, PortDiffGraph};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use relrc::graph_view::RelRcGraphSerializer;
use relrc::{RelRc, RelRcGraph};
use tket2::optimiser::badger::BadgerOptions;
use tket2::optimiser::{BadgerLogger, DiffBadgerOptimiser};
use tket2::static_circ::StaticSizeCircuit;
use tket2::Circuit;

use crate::circuit::{try_with_circ, Tk2Circuit};

/// The module definition
pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new_bound(py, "optimiser")?;
    m.add_class::<PyBadgerOptimiser>()?;
    m.add_class::<PyPortDiffGraph>()?;
    Ok(m)
}

/// Wrapped [`DefaultBadgerOptimiser`].
///
/// Currently only exposes loading from an ECC file using the constructor
/// and optimising using default logging settings.
#[pyclass(name = "BadgerOptimiser")]
pub struct PyBadgerOptimiser(DiffBadgerOptimiser);

#[pymethods]
impl PyBadgerOptimiser {
    /// Create a new [`PyDefaultBadgerOptimiser`] from a precompiled rewriter.
    #[staticmethod]
    pub fn load_precompiled(path: PathBuf) -> Self {
        Self(DiffBadgerOptimiser::diff_with_rewriter_binary(path).unwrap())
    }

    /// Create a new [`PyDefaultBadgerOptimiser`] from ECC sets.
    ///
    /// This will compile the rewriter from the provided ECC JSON file.
    #[staticmethod]
    pub fn compile_eccs(path: &str) -> Self {
        Self(DiffBadgerOptimiser::diff_with_eccs_json_file(path).unwrap())
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
    ///     improvements to the circuit. If no progress is made in this time,
    ///     the optimiser will stop.
    ///
    ///     If `None` the optimiser will run indefinitely, or until `timeout` is
    ///     reached.
    ///
    /// * `max_circuit_count`: The maximum number of circuits to process before
    ///     stopping the optimisation.
    ///
    ///
    ///     For data parallel multi-threading, (split_circuit=true), applies on
    ///     a per-thread basis, otherwise applies globally.
    ///
    ///     If `None` the optimiser will run indefinitely, or until `timeout` is
    ///     reached.
    ///
    /// * `n_threads`: The number of threads to use. Defaults to `1`.
    ///
    /// * `split_circ`: Whether to split the circuit into chunks and process
    ///     each in a separate thread.
    ///
    ///     If this option is set to `true`, the optimiser will split the
    ///     circuit into `n_threads` chunks.
    ///
    ///     If this option is set to `false`, the optimiser will run `n_threads`
    ///     parallel searches on the whole circuit (default).
    ///
    /// * `queue_size`: The maximum size of the circuit candidates priority
    ///     queue. Defaults to `20`.
    ///
    /// * `log_progress`: The path to a CSV file to log progress to.
    ///
    #[allow(clippy::too_many_arguments)]
    pub fn run_portdiff(
        &self,
        circ: &Bound<'_, PyAny>,
        timeout: Option<u64>,
        progress_timeout: Option<u64>,
        max_circuit_count: Option<usize>,
        n_threads: Option<NonZeroUsize>,
        split_circ: Option<bool>,
        queue_size: Option<usize>,
        log_progress: Option<PathBuf>,
    ) -> PyResult<PyPortDiffGraph> {
        let options = BadgerOptions {
            timeout,
            progress_timeout,
            max_circuit_count,
            n_threads: n_threads.unwrap_or(NonZeroUsize::new(1).unwrap()),
            split_circuit: split_circ.unwrap_or(false),
            queue_size: queue_size.unwrap_or(100),
        };
        try_with_circ(circ, |circ, _| {
            let diffs = self.optimise(circ, log_progress, options);
            let py_diffs: Result<PyPortDiffGraph, _> = diffs.try_into();
            py_diffs.map_err(|e| PyErr::new::<PyValueError, _>(e.to_string()))
        })
    }
}

impl PyBadgerOptimiser {
    /// The Python optimise method, but on Hugrs.
    pub(super) fn optimise(
        &self,
        circ: Circuit,
        log_progress: Option<PathBuf>,
        options: BadgerOptions,
    ) -> PortDiffGraph<StaticSizeCircuit> {
        let circ: StaticSizeCircuit = (&circ).try_into().unwrap();
        let badger_logger = log_progress
            .map(|file_name| {
                let log_file = fs::File::create(file_name).unwrap();
                let log_file = BufWriter::new(log_file);
                BadgerLogger::new(log_file)
            })
            .unwrap_or_default();
        self.0.optimise_with_log(&circ, badger_logger, options)
    }
}

/// Wrapped [`DefaultBadgerOptimiser`].
///
/// Currently only exposes loading from an ECC file using the constructor
/// and optimising using default logging settings.
#[pyclass(name = "PortDiffGraph")]
#[derive(Clone)]
pub struct PyPortDiffGraph(
    RelRcGraphSerializer<PortDiffData<StaticSizeCircuit>, EdgeData<StaticSizeCircuit>>,
);

impl PyPortDiffGraph {
    /// Parse the graph
    #[allow(dead_code)]
    fn diffs(&self) -> PyResult<PortDiffGraph<StaticSizeCircuit>> {
        self.clone()
            .try_into()
            .map_err(|_| PyErr::new::<PyValueError, _>(format!("Invalid encoded circuit")))
    }
}

#[pymethods]
impl PyPortDiffGraph {
    /// Extract circuit from a portdiff graph rewrite history.
    fn extract_circuit(&self, nodes: Vec<usize>) -> PyResult<Tk2Circuit> {
        let all_diffs = self.0.get_diffs().unwrap();
        let diffs = nodes
            .into_iter()
            .map(|idx| all_diffs[idx].clone().into())
            .collect();
        let graph = PortDiff::extract_graph(diffs)
            .map_err(|e| PyErr::new::<PyValueError, _>(format!("Error extracting graph: {e:?}")))?;
        if !graph.is_acyclic() {
            return Err(PyErr::new::<PyValueError, _>(format!("Not a DAG")));
        } else {
            let circ: Circuit = graph.into();
            Ok(circ.into())
        }
    }

    fn all_edges(&self) -> PyResult<Vec<(usize, usize, BTreeSet<usize>)>> {
        let all_diffs = self.0.get_diffs().unwrap();
        let find_ind = |n: &RelRc<_, _>| {
            all_diffs
                .iter()
                .position(|v| RelRc::as_ptr(v) == RelRc::as_ptr(n))
                .unwrap()
        };
        let res = all_diffs
            .iter()
            .flat_map(|d| {
                d.all_outgoing()
                    .iter()
                    .map(|e| {
                        let nodes = e
                            .value()
                            .subgraph()
                            .nodes()
                            .iter()
                            .map(|n| (*n).into())
                            .collect();
                        (find_ind(e.source()), find_ind(e.target()), nodes)
                    })
                    .collect_vec()
            })
            .collect();
        Ok(res)
    }

    fn value(&self, idx: usize) -> Option<usize> {
        let all_diffs = self.0.get_diffs().unwrap();
        all_diffs.get(idx).and_then(|d| d.value().value())
    }

    fn n_diffs(&self) -> usize {
        self.0.get_diffs().unwrap().len()
    }

    fn json(&self) -> String {
        serde_json::to_string(&self.0).unwrap()
    }
}

impl From<PortDiffGraph<StaticSizeCircuit>> for PyPortDiffGraph {
    fn from(value: PortDiffGraph<StaticSizeCircuit>) -> Self {
        let g = value.inner().into();
        Self(g)
    }
}

impl TryFrom<PyPortDiffGraph> for PortDiffGraph<StaticSizeCircuit> {
    type Error = (); //GraphDeserializationError;

    fn try_from(value: PyPortDiffGraph) -> Result<Self, Self::Error> {
        let g: RelRcGraph<PortDiffData<StaticSizeCircuit>, EdgeData<StaticSizeCircuit>> =
            value.0.try_into().map_err(|_| ())?;
        Ok(g.into())
    }
}
