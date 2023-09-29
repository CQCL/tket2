//! PyO3 wrapper for the TASO circuit optimiser.

use std::{fs, num::NonZeroUsize, path::PathBuf};

use pyo3::prelude::*;
use tket2::optimiser::{DefaultTasoOptimiser, TasoLogger};

use crate::circuit::update_hugr;

/// The circuit optimisation module.
pub fn add_optimiser_module(py: Python, parent: &PyModule) -> PyResult<()> {
    let m = PyModule::new(py, "optimiser")?;
    m.add_class::<PyDefaultTasoOptimiser>()?;

    parent.add_submodule(m)
}

/// Wrapped [`DefaultTasoOptimiser`].
///
/// Currently only exposes loading from an ECC file using the constructor
/// and optimising using default logging settings.
#[pyclass(name = "TasoOptimiser")]
pub struct PyDefaultTasoOptimiser(DefaultTasoOptimiser);

#[pymethods]
impl PyDefaultTasoOptimiser {
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
    pub fn optimise(
        &self,
        circ: PyObject,
        timeout: Option<u64>,
        n_threads: Option<NonZeroUsize>,
        log_progress: Option<PathBuf>,
    ) -> PyResult<PyObject> {
        let taso_logger = log_progress
            .map(|file_name| {
                let log_file = fs::File::create(file_name).unwrap();
                TasoLogger::new(log_file)
            })
            .unwrap_or_default();
        update_hugr(circ, |circ| {
            self.0.optimise_with_log(
                &circ,
                taso_logger,
                timeout,
                n_threads.unwrap_or(NonZeroUsize::new(1).unwrap()),
                false,
            )
        })
    }
}
