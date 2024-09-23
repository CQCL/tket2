//! Module for extracting circuits from PortDiffGraphs.

use itertools::Itertools;
use portdiff::{PortDiff, PortDiffGraph};
use pyo3::{exceptions::PyValueError, prelude::*};
use tket2::{serialize::save_tk1_json_file, static_circ::StaticSizeCircuit, Circuit};

/// Try to extract a circuit from a PortDiffGraph and save it to a file.
///
/// Returns true if the extraction is successful and the circuit is acyclic.
#[pyfunction]
fn extract_circ_to_file<'py>(
    all_diffs: &str,
    chosen_diffs: Vec<usize>,
    file_name: &str,
) -> PyResult<bool> {
    let diffs: PortDiffGraph<StaticSizeCircuit> = serde_json::from_str(all_diffs).unwrap();
    let all_diffs = diffs.all_nodes().collect_vec();
    let chosen_diffs = chosen_diffs
        .into_iter()
        .map(|i| diffs.get_diff(all_diffs[i]).clone())
        .collect_vec();

    let extracted = PortDiff::extract_graph(chosen_diffs)
        .map_err(|err| PyValueError::new_err(format!("{:?}", err)))?;
    if !extracted.is_acyclic() {
        Ok(false)
    } else {
        let circ: Circuit = extracted.into();
        save_tk1_json_file(&circ, file_name)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(true)
    }
}

/// Nascent Portdiff module.
pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new_bound(py, "portdiff")?;
    m.add_function(wrap_pyfunction!(extract_circ_to_file, &m)?)?;
    Ok(m)
}
