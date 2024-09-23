//! Optimisers for circuit rewriting.
//!
//! Currently, the only optimiser is Badger

#[cfg(feature = "portmatching")]
pub mod badger;
#[cfg(feature = "portmatching")]
pub use badger::{BadgerLogger, BadgerOptimiser, DefaultBadgerOptimiser, DiffBadgerOptimiser};
