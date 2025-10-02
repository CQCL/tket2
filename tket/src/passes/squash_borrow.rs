//! Reorder and squash pairs of borrow and return nodes where possible.

pub mod analysis;
pub use analysis::BorrowAnalysis;
