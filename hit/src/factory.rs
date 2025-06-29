//! Commit factories for rewrite space exploration.

use hugr::persistent::{PatchNode, PersistentWire};

use itertools::Either;
pub use tket2::rewrite_space::CliffordSimpFactory;
pub use tket2::rewrite_space::CommuteCZFactory;
use tket2::rewrite_space::IterMatched;
use tket2::rewrite_space::{CliffordSubcircuit, CommitFactory, CommuteCZ};
// add further factories here

#[derive(Debug, Clone, Copy, strum::Display, strum::EnumString)]
pub enum SupportedFactory {
    Clifford,
    CommuteCZ,
}

impl From<CliffordSimpFactory> for SupportedFactory {
    fn from(_factory: CliffordSimpFactory) -> Self {
        SupportedFactory::Clifford
    }
}

impl From<CommuteCZFactory> for SupportedFactory {
    fn from(_factory: CommuteCZFactory) -> Self {
        SupportedFactory::CommuteCZ
    }
}

pub enum PatternMatch {
    Clifford(CliffordSubcircuit),
    CommuteCZ(CommuteCZ),
}

impl PatternMatch {
    fn as_clifford(&self) -> Option<&CliffordSubcircuit> {
        match self {
            PatternMatch::Clifford(m) => Some(m),
            PatternMatch::CommuteCZ(_) => None,
        }
    }

    fn as_commute_cz(&self) -> Option<&CommuteCZ> {
        match self {
            PatternMatch::Clifford(_) => None,
            PatternMatch::CommuteCZ(m) => Some(m),
        }
    }
}

impl IterMatched for PatternMatch {
    fn matched_wires(&self) -> impl Iterator<Item = &PersistentWire> + '_ {
        match self {
            PatternMatch::Clifford(m) => Either::Left(m.matched_wires()),
            PatternMatch::CommuteCZ(m) => Either::Right(m.matched_wires()),
        }
    }

    fn matched_isolated_nodes(&self) -> impl Iterator<Item = PatchNode> + '_ {
        match self {
            PatternMatch::Clifford(m) => Either::Left(m.matched_isolated_nodes()),
            PatternMatch::CommuteCZ(m) => Either::Right(m.matched_isolated_nodes()),
        }
    }
}

impl CommitFactory for SupportedFactory {
    type PatternMatch = PatternMatch;

    type Cost = usize;

    const PATTERN_RADIUS: usize = 4;

    fn get_replacement(
        &self,
        pattern_match: &Self::PatternMatch,
        matched_subgraph: &hugr::hugr::views::SiblingSubgraph<PatchNode>,
        host: &tket2::rewrite_space::PersistentHugr,
    ) -> Option<hugr::Hugr> {
        match self {
            SupportedFactory::Clifford => CliffordSimpFactory::get_replacement(
                &CliffordSimpFactory,
                pattern_match.as_clifford().unwrap(),
                matched_subgraph,
                host,
            ),
            SupportedFactory::CommuteCZ => CommuteCZFactory::get_replacement(
                &CommuteCZFactory,
                pattern_match.as_commute_cz().unwrap(),
                matched_subgraph,
                host,
            ),
        }
    }

    fn find_pattern_matches<'w: 'a, 'a>(
        &'a self,
        pattern_root: PatchNode,
        walker: tket2::rewrite_space::Walker<'w>,
    ) -> impl Iterator<Item = (Self::PatternMatch, tket2::rewrite_space::Walker<'w>)> + 'a {
        match self {
            SupportedFactory::Clifford => Either::Left(
                CliffordSimpFactory::find_pattern_matches(
                    &CliffordSimpFactory,
                    pattern_root,
                    walker,
                )
                .map(|(m, w)| (PatternMatch::Clifford(m), w)),
            ),
            SupportedFactory::CommuteCZ => Either::Right(
                CommuteCZFactory::find_pattern_matches(&CommuteCZFactory, pattern_root, walker)
                    .map(|(m, w)| (PatternMatch::CommuteCZ(m), w)),
            ),
        }
    }

    fn op_cost(&self, op: &hugr::ops::OpType) -> Option<Self::Cost> {
        let cost = CliffordSimpFactory::op_cost(&CliffordSimpFactory, op);
        debug_assert_eq!(
            cost,
            CommuteCZFactory::op_cost(&CommuteCZFactory, op),
            "mismatched cost functions"
        );
        cost
    }

    fn get_name(
        &self,
        pattern_match: &Self::PatternMatch,
        host: &tket2::rewrite_space::PersistentHugr,
    ) -> String {
        match self {
            SupportedFactory::Clifford => CliffordSimpFactory::get_name(
                &CliffordSimpFactory,
                pattern_match.as_clifford().unwrap(),
                host,
            ),
            SupportedFactory::CommuteCZ => CommuteCZFactory::get_name(
                &CommuteCZFactory,
                pattern_match.as_commute_cz().unwrap(),
                host,
            ),
        }
    }
}
