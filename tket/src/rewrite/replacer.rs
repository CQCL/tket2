//! Providing replacements for pattern matches.

use hugr::HugrView;

use crate::{resource::ResourceScope, Circuit, Subcircuit};

/// Provide possible replacements for a pattern match.
pub trait CircuitReplacer<MatchInfo> {
    /// Get the possible replacements for a pattern match.
    ///
    /// The order (and signature) of the inputs and outputs on the returned
    /// circuits must match the order of the boundary ports in `subgraph`.
    fn replace_match<H: HugrView>(
        &self,
        subcircuit: &Subcircuit<H::Node>,
        circuit: &ResourceScope<H>,
        match_info: MatchInfo,
    ) -> Vec<Circuit>;
}
