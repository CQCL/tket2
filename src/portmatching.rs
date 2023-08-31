//! Pattern matching for circuits

pub mod matcher;
pub mod pattern;
#[cfg(feature = "pyo3")]
mod pyo3;

pub use matcher::{CircuitMatch, CircuitMatcher, CircuitRewrite};
pub use pattern::CircuitPattern;

use hugr::Port;

use crate::T2Op;

type PEdge = (Port, Port);
type PNode = T2Op;
