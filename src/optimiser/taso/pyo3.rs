//! PyO3 wrapper for the TASO optimiser.

use std::{fs, num::NonZeroUsize};

use pyo3::{exceptions::PyTypeError, prelude::*};
use tket_json_rs::circuit_json::SerialCircuit;

use crate::{json::TKETDecode, utils::pyobj_as_hugr};

use super::{log::TasoLogger, DefaultTasoOptimiser};

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
    pub fn load_precompiled(path: &str) -> Self {
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
    /// Returns an optimised circuit and log the progress to a CSV
    /// file called "best_circs.csv".
    pub fn optimise(
        &self,
        circ: PyObject,
        timeout: Option<u64>,
        n_threads: Option<NonZeroUsize>,
    ) -> PyResult<PyObject> {
        let circ = pyobj_as_hugr(circ)?;
        let circ_candidates_csv = fs::File::create("best_circs.csv").unwrap();

        let taso_logger = TasoLogger::new(circ_candidates_csv);
        let opt_circ = self.0.optimise_with_log(
            &circ,
            taso_logger,
            timeout,
            n_threads.unwrap_or(NonZeroUsize::new(1).unwrap()),
        );
        let ser_circ =
            SerialCircuit::encode(&opt_circ).map_err(|e| PyTypeError::new_err(e.to_string()))?;
        let tk1_circ = ser_circ
            .to_tket1()
            .map_err(|e| PyTypeError::new_err(e.to_string()))?;
        Ok(tk1_circ)
    }
}
