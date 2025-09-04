//! This module defines the Hugr extensions used by tket-qsystem.
mod compute;
pub use compute::gpu;
pub use compute::wasm;
pub mod futures;
pub mod qsystem;
pub mod random;
pub mod result;
pub mod utils;
