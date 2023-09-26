//! Optimisers for circuit rewriting.
//!
//! Currently, the only optimiser is TASO

pub mod taso;
pub use taso::{DefaultTasoOptimiser, TasoOptimiser};
