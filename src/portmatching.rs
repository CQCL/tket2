//! Pattern matching for circuits

pub mod matcher;
pub mod pattern;
#[cfg(feature = "pyo3")]
pub mod pyo3;

pub use matcher::{PatternMatch, PatternMatcher};
pub use pattern::CircuitPattern;

use hugr::Port;
use matcher::MatchOp;

type PEdge = (Port, Port);
type PNode = MatchOp;
