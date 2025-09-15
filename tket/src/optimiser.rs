//! Optimisers for circuit rewriting.
//!
//! Currently, the only optimiser is Badger

pub mod badger;

#[cfg(feature = "portmatching")]
pub use badger::ECCBadgerOptimiser;
pub use badger::{BadgerLogger, BadgerOptimiser};
