//! Pattern matching for circuits

pub mod matcher;
mod optype;
pub mod pattern;
#[cfg(feature = "pyo3")]
mod pyo3;

pub use matcher::{CircuitMatch, CircuitMatcher, CircuitRewrite};
pub use pattern::CircuitPattern;

use hugr::Port;
use optype::MatchOp;

type PEdge = (Port, Port);
type PNode = MatchOp;
