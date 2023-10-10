//! PyO3 wrapper for the TASO circuit optimiser.

use std::io::BufWriter;
use std::{fs, num::NonZeroUsize, path::PathBuf};

use hugr::Hugr;
use pyo3::prelude::*;
use tket2::optimiser::{DefaultTasoOptimiser, TasoLogger};

use crate::circuit::update_hugr;

/// The circuit optimisation module.
pub fn add_optimiser_module(py: Python, parent: &PyModule) -> PyResult<()> {
    let m = PyModule::new(py, "optimiser")?;
    m.add_class::<PyTasoOptimiser>()?;

    parent.add_submodule(m)
}

/// Wrapped [`DefaultTasoOptimiser`].
///
/// Currently only exposes loading from an ECC file using the constructor
/// and optimising using default logging settings.
#[pyclass(name = "TasoOptimiser")]
pub struct PyTasoOptimiser(DefaultTasoOptimiser);

#[pymethods]
impl PyTasoOptimiser {
    /// Create a new [`PyDefaultTasoOptimiser`] from a precompiled rewriter.
    #[staticmethod]
    pub fn load_precompiled(path: PathBuf) -> Self {
        Self(DefaultTasoOptimiser::default_with_rewriter_binary(path).unwrap())
    }

    /// Create a new [`PyDefaultTasoOptimiser`] from ECC sets.
    ///
    /// This will compile the rewriter from the provided ECC JSON file.
    #[staticmethod]
    pub fn compile_eccs(path: &str) -> Self {
        Self(DefaultTasoOptimiser::default_with_eccs_json_file(path).unwrap())
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
    pub fn py_optimise(
        &self,
        circ: PyObject,
        timeout: Option<u64>,
        n_threads: Option<NonZeroUsize>,
        split_circ: Option<bool>,
        log_progress: Option<PathBuf>,
        queue_size: Option<usize>,
    ) -> PyResult<PyObject> {
        update_hugr(circ, |circ| {
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

impl PyTasoOptimiser {
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
        let taso_logger = log_progress
            .map(|file_name| {
                let log_file = fs::File::create(file_name).unwrap();
                let log_file = BufWriter::new(log_file);
                TasoLogger::new(log_file)
            })
            .unwrap_or_default();
        self.0.optimise_with_log(
            &circ,
            taso_logger,
            timeout,
            n_threads.unwrap_or(NonZeroUsize::new(1).unwrap()),
            split_circ.unwrap_or(false),
            queue_size.unwrap_or(100),
        )
    }
}
