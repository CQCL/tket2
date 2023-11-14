//! PyO3 wrapper for the Badger circuit optimiser.

use std::io::BufWriter;
use std::{fs, num::NonZeroUsize, path::PathBuf};

use hugr::Hugr;
use pyo3::prelude::*;
use tket2::optimiser::{BadgerLogger, DefaultBadgerOptimiser};

use crate::circuit::update_hugr;

/// The module definition
pub fn module(py: Python) -> PyResult<&PyModule> {
    let m = PyModule::new(py, "_optimiser")?;
    m.add_class::<PyBadgerOptimiser>()?;
    Ok(m)
}

/// Wrapped [`DefaultBadgerOptimiser`].
///
/// Currently only exposes loading from an ECC file using the constructor
/// and optimising using default logging settings.
#[pyclass(name = "BadgerOptimiser")]
pub struct PyBadgerOptimiser(DefaultBadgerOptimiser);

#[pymethods]
impl PyBadgerOptimiser {
    /// Create a new [`PyDefaultBadgerOptimiser`] from a precompiled rewriter.
    #[staticmethod]
    pub fn load_precompiled(path: PathBuf) -> Self {
        Self(DefaultBadgerOptimiser::default_with_rewriter_binary(path).unwrap())
    }

    /// Create a new [`PyDefaultBadgerOptimiser`] from ECC sets.
    ///
    /// This will compile the rewriter from the provided ECC JSON file.
    #[staticmethod]
    pub fn compile_eccs(path: &str) -> Self {
        Self(DefaultBadgerOptimiser::default_with_eccs_json_file(path).unwrap())
    }

    /// Run the optimiser on a circuit.
    ///
    /// Returns an optimised circuit and optionally log the progress to a CSV
    /// file.
    ///
    /// # Parameters
    ///
    /// * `circ`: The circuit to optimise.
    /// * `timeout`: The timeout in seconds.
    /// * `n_threads`: The number of threads to use.
    /// * `split_circ`: Whether to split the circuit into chunks before
    ///   processing.
    ///
    ///   If this option is set, the optimise will divide the circuit into
    ///   `n_threads` chunks and optimise each on a separate thread.
    /// * `log_progress`: The path to a CSV file to log progress to.
    ///
    #[pyo3(name = "optimise")]
    pub fn py_optimise<'py>(
        &self,
        circ: &'py PyAny,
        timeout: Option<u64>,
        n_threads: Option<NonZeroUsize>,
        split_circ: Option<bool>,
        log_progress: Option<PathBuf>,
        queue_size: Option<usize>,
    ) -> PyResult<&'py PyAny> {
        update_hugr(circ, |circ, _| {
            self.optimise(
                circ,
                timeout,
                n_threads,
                split_circ,
                log_progress,
                queue_size,
            )
        })
    }
}

impl PyBadgerOptimiser {
    /// The Python optimise method, but on Hugrs.
    pub(super) fn optimise(
        &self,
        circ: Hugr,
        timeout: Option<u64>,
        n_threads: Option<NonZeroUsize>,
        split_circ: Option<bool>,
        log_progress: Option<PathBuf>,
        queue_size: Option<usize>,
    ) -> Hugr {
        let badger_logger = log_progress
            .map(|file_name| {
                let log_file = fs::File::create(file_name).unwrap();
                let log_file = BufWriter::new(log_file);
                BadgerLogger::new(log_file)
            })
            .unwrap_or_default();
        self.0.optimise_with_log(
            &circ,
            badger_logger,
            timeout,
            n_threads.unwrap_or(NonZeroUsize::new(1).unwrap()),
            split_circ.unwrap_or(false),
            queue_size.unwrap_or(100),
        )
    }
}
