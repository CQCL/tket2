//! Providing replacements for pattern matches.

use hugr::{hugr::views::SiblingSubgraph, HugrView};

use crate::Circuit;

/// Provide possible replacements for a pattern match.
pub trait CircuitReplacer<MatchInfo> {
    /// Get the possible replacements for a pattern match.
    ///
    /// The order (and signature) of the inputs and outputs on the returned circuits must match
    /// the order of the boundary ports in `subgraph`.
    fn replace_match<H: HugrView>(
        &self,
        subgraph: &SiblingSubgraph<H::Node>,
        hugr: H,
        match_info: MatchInfo,
    ) -> Vec<Circuit>;
}
