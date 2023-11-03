//! Rust struct for PHIR and conversion from HUGR.

mod convert;
mod model;

pub use convert::circuit_to_phir;
pub use model::PHIRModel;
