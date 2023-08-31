//! Pattern matching for circuits

pub mod matcher;
pub mod pattern;
#[cfg(feature = "pyo3")]
mod pyo3;

pub use matcher::{CircuitMatch, CircuitMatcher, CircuitRewrite, MatchOp};
pub use pattern::CircuitPattern;

use hugr::Port;

type PEdge = (Port, Port);
type PNode = MatchOp;
