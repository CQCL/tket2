//! Module for richer circuit representation and operations.
//! This module provides three extensions: modifiers, global phase, and safe drop.
//!
//! ## Modifiers
//! Modifiers are functions that takes circuits and return modified circuits
//! by applying modifiers: control, dagger, or power.
//!
//! ## Global Phase
//! Global phase is an operation that applies some global phase to a circuit.
//! It is implemented as a side-effect that takes a rotation angle as an input.
pub mod control;
pub mod dagger;
pub mod power;
pub mod qubit_types_utils;
