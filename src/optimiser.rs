//! Optimisers for circuit rewriting.
//!
//! Currently, the only optimiser is TASO

pub mod taso;

#[cfg(feature = "portmatching")]
pub use taso::DefaultTasoOptimiser;
pub use taso::TasoOptimiser;
