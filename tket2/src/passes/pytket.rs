//! This module contains routines needed for normalizing a circuit
//! into a form that can be encoded as a pytket legacy circuit.
//!
//! This is a best-effort attempt, and may not always succeed.

use itertools::Itertools;

use crate::serialize::pytket::OpConvertError;
use crate::Circuit;

use super::find_tuple_unpack_rewrites;

/// Try to lower a circuit to a form that can be encoded as a pytket legacy circuit.
pub fn lower_to_pytket(circ: &Circuit) -> Result<Circuit, PytketLoweringError> {
    let mut circ = circ
        .extract_dfg()
        .map_err(|_| PytketLoweringError::NonLocalOperations)?;

    // Remove sequences of tuple pack-unpack operations,
    // typically generated by guppy.
    let rewrites = find_tuple_unpack_rewrites(&circ).collect_vec();
    for rewrite in rewrites {
        rewrite.apply(&mut circ).unwrap();
    }

    Ok(circ)
}

/// Errors that can occur during the lowering process.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum PytketLoweringError {
    /// An error occurred during the conversion of an operation.
    #[error("operation conversion error: {0}")]
    OpConversionError(#[from] OpConvertError),
    /// The circuit is not fully-contained in a region.
    /// Function calls are not supported.
    #[error("Non-local operations found. Function calls are not supported.")]
    NonLocalOperations,
}