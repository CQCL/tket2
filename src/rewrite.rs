//! Transform circuits using rewrite rules.

#[cfg(feature = "portmatching")]
pub mod ecc_rewriter;
#[cfg(feature = "portmatching")]
pub use ecc_rewriter::ECCRewriter;

use delegate::delegate;
use derive_more::{From, Into};
use hugr::hugr::views::sibling_subgraph::InvalidReplacement;
use hugr::{
    hugr::{hugrmut::HugrMut, views::SiblingSubgraph, Rewrite, SimpleReplacementError},
    Hugr, SimpleReplacement,
};

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

use crate::circuit::Circuit;

/// A subcircuit of a circuit.
#[derive(Debug, Clone, From, Into)]
pub struct Subcircuit {
    pub(crate) subgraph: SiblingSubgraph,
}

/// A rewrite rule for circuits.
#[cfg_attr(feature = "pyo3", pyclass)]
#[derive(Debug, Clone, From, Into)]
pub struct CircuitRewrite(SimpleReplacement);

impl CircuitRewrite {
    /// Create a new rewrite rule.
    pub fn try_new(
        source_position: &Subcircuit,
        source: &Hugr,
        target: Hugr,
    ) -> Result<Self, InvalidReplacement> {
        source_position
            .subgraph
            .create_simple_replacement(source, target)
            .map(Self)
    }

    delegate! {
        to self.0 {
            /// Apply the rewrite rule to a circuit.
            pub fn apply(self, circ: &mut impl HugrMut) -> Result<(), SimpleReplacementError>;
        }
    }
}

/// Generate rewrite rules for circuits.
pub trait Rewriter {
    /// Get the rewrite rules for a circuit.
    fn get_rewrites<C: Circuit + Clone>(&self, circ: &C) -> Vec<CircuitRewrite>;
}
