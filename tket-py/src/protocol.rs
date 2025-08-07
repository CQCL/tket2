//! Python protocol implementations for CircuitMatcher and CircuitReplacer
//! traits.

mod circuit_matcher;
pub use circuit_matcher::{CircuitMatcherPyProtocol, PyImplCircuitMatcher};

mod circuit_replacer;
pub use circuit_replacer::{CircuitReplacerPyProtocol, PyImplCircuitReplacer};
