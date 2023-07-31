//! Pattern matching for circuits

pub mod matcher;
mod optype;
#[cfg(feature = "pyo3")]
mod pyo3;

pub use matcher::{CircuitMatcher, CircuitPattern};
use optype::MatchOp;
