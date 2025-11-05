//! Passes that call to tket1-passes using the tket-c-api.

use rayon::iter::ParallelIterator;
use std::sync::Arc;

use pyo3::prelude::*;
use tket::serialize::pytket::{EncodeOptions, EncodedCircuit};
use tket::Circuit;
use tket_qsystem::pytket::{qsystem_decoder_config, qsystem_encoder_config};

use crate::circuit::try_with_circ;
use crate::utils::{create_py_exception, ConvertPyErr};

/// An optimisation pass that applies a number of rewrite rules for simplifying
/// Clifford gate sequences, similar to Duncan & Fagan
/// (https://arxiv.org/abs/1901.10114). Produces a circuit comprising TK1 gates
/// and the two-qubit gate specified as the target.
///
/// Parameters:
/// - allow_swaps: whether the rewriting may introduce implicit wire swaps.
/// - traverse_subcircuits: Whether to apply the optimisation to nested
///   subregions in the hugr too, rather than just the top-level region.
//
// TODO: We should also expose `target_gate` here, but the most appropriate
// parameter type [`crate::ops::PyTketOp`] doesn't include `TK2` -.-
#[pyfunction]
#[pyo3(signature = (circ, *, allow_swaps = true, traverse_subcircuits = true))]
pub fn clifford_simp<'py>(
    circ: &Bound<'py, PyAny>,
    allow_swaps: bool,
    traverse_subcircuits: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let py = circ.py();

    try_with_circ(circ, |circ, typ| {
        let circ = run_tket1_pass(circ, traverse_subcircuits, |tk1_circ| {
            tk1_circ.clifford_simp(tket_json_rs::OpType::CX, allow_swaps)
        })?;

        let circ = typ.convert(py, circ)?;
        PyResult::Ok(circ)
    })
}

/// Squash single qubit gates into PhasedX and Rz gates. Also remove identity
/// gates. Commute Rz gates to the back if possible.
///
/// Parameters:
/// - traverse_subcircuits: Whether to apply the optimisation to nested
///   subregions in the hugr too, rather than just the top-level region.
#[pyfunction]
#[pyo3(signature = (circ, *, traverse_subcircuits = true))]
pub fn squash_phasedx_rz<'py>(
    circ: &Bound<'py, PyAny>,
    traverse_subcircuits: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let py = circ.py();

    try_with_circ(circ, |circ, typ| {
        let circ = run_tket1_pass(circ, traverse_subcircuits, |tk1_circ| {
            tk1_circ.squash_phasedx_rz()
        })?;

        let circ = typ.convert(py, circ)?;
        PyResult::Ok(circ)
    })
}

fn run_tket1_pass<F>(mut circ: Circuit, traverse_subcircuits: bool, pass: F) -> PyResult<Circuit>
where
    F: Fn(&mut tket1_passes::Tket1Circuit) -> Result<(), tket1_passes::PassError> + Send + Sync,
{
    let mut encoded_circ = EncodedCircuit::new(
        &circ,
        EncodeOptions::new()
            .with_config(qsystem_encoder_config())
            .with_subcircuits(traverse_subcircuits),
    )
    .convert_pyerrs()?;

    encoded_circ
        .par_iter_mut()
        .try_for_each(|(_, circ)| -> Result<(), tket1_passes::PassError> {
            let mut tk1_circ = tket1_passes::Tket1Circuit::from_serial_circuit(circ)?;
            pass(&mut tk1_circ)?;
            *circ = tk1_circ.to_serial_circuit()?;
            Ok(())
        })
        .convert_pyerrs()?;

    encoded_circ
        .reassemble_inplace(circ.hugr_mut(), Some(Arc::new(qsystem_decoder_config())))
        .convert_pyerrs()?;

    Ok(circ)
}

create_py_exception!(
    tket1_passes::PassError,
    PytketPassError,
    "Error from a call to tket-c-api"
);
