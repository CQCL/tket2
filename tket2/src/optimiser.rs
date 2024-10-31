//! Optimisers for circuit rewriting.
//!
//! Currently, the only optimiser is Badger

pub mod badger;

#[cfg(feature = "portmatching")]
pub use badger::DefaultBadgerOptimiser;
pub use badger::{BadgerLogger, BadgerOptimiser};
