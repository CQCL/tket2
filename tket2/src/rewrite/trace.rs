//! Utilities for tracing the rewrites applied to a circuit.
//!
//! This is only tracked if the `rewrite-tracing` feature is enabled.

use hugr::hugr::hugrmut::HugrMut;
use hugr::hugr::NodeMetadata;
use itertools::Itertools;

use crate::Circuit;

use super::CircuitRewrite;

/// Metadata key for the circuit rewrite trace.
pub const METADATA_REWRITES: &str = "TKET2.rewrites";

/// Global read-only flag for enabling rewrite tracing.
/// Enable it by setting the `rewrite-tracing` feature.
///
/// Note that circuits must be explicitly enabled for rewrite tracing by calling
/// [`RewriteTracer::enable_rewrite_tracing`].
pub const REWRITE_TRACING_ENABLED: bool = cfg!(feature = "rewrite-tracing");

/// The trace of a rewrite applied to a circuit.
///
/// Traces are only enabled if the `rewrite-tracing` feature is enabled and
/// [`RewriteTracer::enable_rewrite_tracing`] is called on the circuit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct RewriteTrace {
    /// A count of the number of individual patterns matched for this rewrite step.
    ///
    /// This is relevant when using a greedy rewrite strategy.
    individual_matches: u16,
}

impl From<&CircuitRewrite> for RewriteTrace {
    #[inline]
    fn from(_rewrite: &CircuitRewrite) -> Self {
        // NOTE: We don't currently track any actual information about the rewrite.
        Self {
            individual_matches: 1,
        }
    }
}

impl RewriteTrace {
    /// Create a new trace.
    #[inline]
    pub fn new(individual_matches: u16) -> Self {
        Self { individual_matches }
    }
}

impl From<&serde_json::Value> for RewriteTrace {
    #[inline]
    fn from(value: &serde_json::Value) -> Self {
        Self {
            individual_matches: value.as_u64().unwrap() as u16,
        }
    }
}

impl From<RewriteTrace> for serde_json::Value {
    #[inline]
    fn from(trace: RewriteTrace) -> Self {
        serde_json::Value::from(trace.individual_matches)
    }
}

/// Extension trait for circuits that can trace rewrites applied to them.
///
/// This is only tracked if the `rewrite-tracing` feature is enabled and
/// `enable_rewrite_tracing` is called on the circuit before.
pub trait RewriteTracer: Circuit + HugrMut + Sized {
    /// Enable rewrite tracing for the circuit.
    #[inline]
    fn enable_rewrite_tracing(&mut self) {
        if !REWRITE_TRACING_ENABLED {
            return;
        }
        let meta = self
            .get_metadata_mut(self.root(), METADATA_REWRITES)
            .unwrap();
        if *meta == NodeMetadata::Null {
            *meta = NodeMetadata::Array(vec![]);
        }
    }

    /// Register a rewrite applied to the circuit.
    ///
    /// Returns `true` if the rewrite was successfully registered, or `false` if it was ignored.
    #[inline]
    fn add_rewrite_trace(&mut self, rewrite: impl Into<RewriteTrace>) -> bool {
        if !REWRITE_TRACING_ENABLED {
            return false;
        }
        match self
            .get_metadata_mut(self.root(), METADATA_REWRITES)
            .ok()
            .and_then(|m| m.as_array_mut())
        {
            Some(meta) => {
                let rewrite = rewrite.into();
                meta.push(rewrite.into());
                true
            }
            // Tracing was not enable for this circuit.
            None => false,
        }
    }

    /// Returns the traces of rewrites applied to the circuit.
    ///
    /// Returns `None` if rewrite tracing is not enabled for this circuit.
    //
    // TODO return an `impl Iterator` once rust 1.75 lands.
    #[inline]
    fn rewrite_trace(&self) -> Option<Vec<RewriteTrace>> {
        if !REWRITE_TRACING_ENABLED {
            return None;
        }
        let meta = self.get_metadata(self.root(), METADATA_REWRITES)?;
        let rewrites = meta.as_array()?;
        Some(rewrites.iter().map_into().collect_vec())
    }
}

impl<T: Circuit + HugrMut> RewriteTracer for T {}
