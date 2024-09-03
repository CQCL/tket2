//! Optimisers for circuit rewriting.
//!
//! Currently, the only optimiser is Badger

pub mod badger;

pub use badger::{BadgerLogger, BadgerOptimiser};
#[cfg(feature = "portmatching")]
pub use badger::{DefaultBadgerOptimiser, DiffBadgerOptimiser};
