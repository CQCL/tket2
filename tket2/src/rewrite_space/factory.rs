use hugr::hugr::views::sibling_subgraph::{IncomingPorts, OutgoingPorts};
use hugr::hugr::views::SiblingSubgraph;
use hugr::persistent::{
    subgraph::InvalidPinnedSubgraph, PatchNode, PersistentWire, PinnedSubgraph,
};
use hugr::{Hugr, Port};
use hugr::{IncomingPort, OutgoingPort};
use itertools::Either;

use crate::circuit::cost::CircuitCost;

use super::{PersistentHugr, Walker};

mod boxed;
mod commute_cz;
mod explore;
pub use boxed::{BoxedCommitFactory, FindCommits};
pub use commute_cz::{CommuteCZ, CommuteCZFactory};
pub use explore::ExploreOptions;

/// A trait for producing commits to be used to explore the rewrite space.
///
/// This trait captures two main capabilities of a commit factory:
///  - find patterns of interest in the space (see
///    [CommitFactory::find_pattern_matches])
///  - construct new rewrite candidates from pattern matches, (see
///    [CommitFactory::get_replacement] and [CommitFactory::map_boundary])
///
/// Given these capabilities, the [CommitFactory::explore] method is provided to
/// iteratively explore and expand the space.
pub trait CommitFactory {
    /// A record of a pattern match during exploration.
    type PatternMatch: IterMatched;
    /// The cost type attributed to a rewrite.
    type Cost: CircuitCost;

    /// Maximum distance between a newly inserted node and a new root node
    /// candidate.
    ///
    /// When new nodes are added to the space, all nodes within this radius will
    /// be considered as potential new roots for the next round of
    /// exploration.
    ///
    /// When patterns are of bounded size, this should typically be set to the
    /// maximum distance between the pattern root and a node of the pattern.
    /// Otherwise, pick a reasonable value that reflects the expected size of
    /// the typical patterns matched (a larger radius makes exploration slower
    /// but more complete).
    const PATTERN_RADIUS: usize;

    /// Get the replacement HUGR for a pattern match.
    fn get_replacement(
        &self,
        pattern_match: &Self::PatternMatch,
        matched_subgraph: &SiblingSubgraph<PatchNode>,
        host: &PersistentHugr,
    ) -> Option<Hugr>;

    /// Map a boundary port of a pattern match to a port of the replacement.
    ///
    /// Given the subgraph defined by the wires of `pattern_match` and its
    /// associated boundary edges, `map_boundary` must provide a map from
    /// the boundary ports of the subgraph to the inputs/output ports in
    /// `repl`. The returned port must be of the opposite direction as the
    /// port passed as argument:
    ///  - an incoming subgraph port must be mapped to an outgoing port of the
    ///    input node of `repl`
    /// - an outgoing subgraph port must be mapped to an incoming port of the
    ///   output node of `repl`
    ///
    /// By default, this will map the boundary ports according to the inputs and
    /// outputs ordering of the subgraph returned by [IterMatched::to_subgraph],
    /// i.e. the n-th input/ output port of the matched subgraph is mapped to
    /// the n-th input/output port of the replacement.
    fn map_boundary(
        &self,
        node: PatchNode,
        port: Port,
        pattern_match: &Self::PatternMatch,
        host: &PersistentHugr,
    ) -> Port {
        let (inputs, outputs) = pattern_match.io_ports(host);

        match port.as_directed() {
            Either::Left(incoming) => IncomingPort::from(
                inputs
                    .iter()
                    .position(|vec| vec.iter().any(|&(n, p)| n == node && p == incoming))
                    .unwrap(),
            )
            .into(),
            Either::Right(outgoing) => OutgoingPort::from(
                outputs
                    .iter()
                    .position(|&(n, p)| n == node && p == outgoing)
                    .unwrap(),
            )
            .into(),
        }
    }

    /// Find all pattern matches in the space with `pattern_root` as their root.
    fn find_pattern_matches<'w: 'a, 'a>(
        &'a self,
        pattern_root: PatchNode,
        walker: Walker<'w>,
    ) -> impl Iterator<Item = (Self::PatternMatch, Walker<'w>)> + 'a;
}

/// A trait for iterating over the nodes and wires that make up a pattern match.
pub trait IterMatched {
    /// Iterate over the wires that make up a pattern match.
    fn matched_wires(&self) -> impl Iterator<Item = &PersistentWire> + '_;

    /// Iterate over the nodes not adjacent to any matched wire in the pattern
    /// match.
    ///
    /// Defaults to an empty iterator. Implementations should override this
    /// method if they support matching isolated nodes.
    fn matched_isolated_nodes(&self) -> impl Iterator<Item = PatchNode> + '_ {
        std::iter::empty()
    }

    /// Convert the pattern match to a pinned subgraph.
    ///
    /// This will fail if in `walker` any node of the match is not pinned or
    /// any wire is not complete.
    fn to_subgraph(&self, walker: &Walker) -> Result<PinnedSubgraph, InvalidPinnedSubgraph> {
        let nodes = self.matched_isolated_nodes();
        let wires = self.matched_wires().cloned();
        PinnedSubgraph::try_from_pinned(nodes, wires, walker)
    }

    /// Get the input and output ports of the pattern match.
    ///
    /// The returned ports are the boundary ports of the subgraph returned by
    /// [IterMatched::to_subgraph].
    fn io_ports(
        &self,
        host: &PersistentHugr,
    ) -> (IncomingPorts<PatchNode>, OutgoingPorts<PatchNode>) {
        let (inputs, outputs, _) = PinnedSubgraph::compute_io_ports(
            self.matched_isolated_nodes(),
            self.matched_wires().cloned(),
            host,
        );
        (inputs, outputs)
    }
}

impl IterMatched for Vec<PersistentWire> {
    fn matched_wires(&self) -> impl Iterator<Item = &PersistentWire> + '_ {
        self.iter()
    }
}

#[cfg(test)]
mod tests {
    use explore::new_nodes_within_radius;
    use hugr::persistent::serial::SerialPersistentHugr;
    use itertools::Itertools;
    use rstest::{fixture, rstest};

    use super::*;
    use crate::{
        rewrite_space::{PersistentHugr, Walker},
        serialize::HugrWithExts,
    };

    #[fixture]
    fn perhugr1() -> PersistentHugr {
        const FILE: &str = "../test_files/perhugr1.json";
        let f = std::fs::File::open(FILE).unwrap();
        let ser_perhugr: SerialPersistentHugr<HugrWithExts, _> =
            serde_json::from_reader(f).unwrap();
        PersistentHugr::from_serial(ser_perhugr)
    }

    #[rstest]
    fn test_cz_commit_factory(perhugr1: PersistentHugr) {
        let non_base_commit = perhugr1
            .all_commit_ids()
            .filter(|&cm| cm != perhugr1.base())
            .exactly_one()
            .ok()
            .unwrap();

        let walkers = perhugr1.inserted_nodes(non_base_commit).map(|n| {
            (
                n,
                Walker::from_pinned_node(n, perhugr1.as_state_space().clone()),
            )
        });

        let res = new_nodes_within_radius(walkers, 1);

        let within_base = res
            .into_iter()
            .filter(|(n, _)| n.owner() == perhugr1.base())
            .collect_vec();
        assert_eq!(within_base.len(), 4);

        let cz_factory = CommuteCZFactory;

        let all_rewrites = within_base
            .into_iter()
            .flat_map(|(pattern_root, walker)| {
                cz_factory.find_pattern_matches(pattern_root, walker)
            })
            .collect_vec();

        assert!(all_rewrites
            .into_iter()
            .any(|(rw, _)| matches!(rw, CommuteCZ::Cancel(_))))
    }
}
