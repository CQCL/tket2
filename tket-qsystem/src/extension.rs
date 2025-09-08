//! This module defines the Hugr extensions used by tket-qsystem.
mod classical_compute;
pub use classical_compute::wasm;
pub mod futures;
pub mod qsystem;
pub mod random;
pub mod result;
pub mod utils;
