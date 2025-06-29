use std::collections::{BTreeSet, VecDeque};

use hugr::hugr::views::sibling_subgraph::{IncomingPorts, OutgoingPorts};
use hugr::hugr::views::SiblingSubgraph;
use hugr::persistent::{
    subgraph::InvalidPinnedSubgraph, PatchNode, PersistentWire, PinnedSubgraph,
};
use hugr::{ops::OpType, Hugr, HugrView, Port, SimpleReplacement};
use hugr::{Direction, IncomingPort, OutgoingPort};
use itertools::{Either, Itertools};

use crate::circuit::cost::CircuitCost;

use super::{PersistentHugr, RewriteSpace, Walker};

mod clifford;
mod commute_cz;
pub use clifford::{CliffordSimpFactory, CliffordSubcircuit};
pub use commute_cz::{CommuteCZ, CommuteCZFactory};

/// Options to control the exploration process within commit factories.
#[derive(Debug, Clone, Default)]
pub struct ExploreOptions {
    /// The maximum number of rewrites to explore.
    pub max_rewrites: Option<usize>,
}

/// A trait for producing commits to be used to explore the rewrite space.
///
/// This trait captures two main capabilities of a commit factory:
///  - find patterns of interest in the space (see
///    [CommitFactory::find_pattern_matches])
///  - construct new rewrite candidates from pattern matches, (see
///    [CommitFactory::get_replacement], [CommitFactory::map_boundary] and
///    [CommitFactory::op_cost])
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
            Either::Left(incoming) => OutgoingPort::from(
                inputs
                    .iter()
                    .position(|vec| vec.iter().any(|&(n, p)| n == node && p == incoming))
                    .unwrap(),
            )
            .into(),
            Either::Right(outgoing) => IncomingPort::from(
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

    /// Get the cost of an operation.
    fn op_cost(&self, op: &OpType) -> Option<Self::Cost>;

    /// Get the name for a pattern match.
    fn get_name(&self, pattern_match: &Self::PatternMatch, host: &PersistentHugr) -> String;

    /// Explore and expand the rewrite space.
    ///
    /// This method will iteratively explore the space, finding pattern matches
    /// and constructing new rewrite candidates.
    fn explore(
        &self,
        space: &mut RewriteSpace<<Self::Cost as CircuitCost>::CostDelta>,
        options: &ExploreOptions,
    ) {
        let mut queue = VecDeque::new();
        let mut explored_rewrites = 0;

        // enqueue all modes of all commits
        for cm in space.all_commit_ids() {
            for node in space.inserted_nodes(cm) {
                let walker = Walker::from_pinned_node(node, space.state_space.clone());
                queue.push_back((node, walker));
            }
        }

        while let Some((node, walker)) = queue.pop_front() {
            for (pattern_match, new_walker) in self.find_pattern_matches(node, walker) {
                let host = new_walker.as_hugr_view();
                let subgraph = pattern_match.to_subgraph(&new_walker).expect("valid match");
                let Ok(sibling_subgraph) = subgraph.to_sibling_subgraph(host) else {
                    // not convex
                    continue;
                };
                let Some(replacement) =
                    self.get_replacement(&pattern_match, &sibling_subgraph, host)
                else {
                    continue;
                };
                let Ok(new_commit) =
                    new_walker.try_create_commit(subgraph, replacement, |node, port| {
                        self.map_boundary(node, port, &pattern_match, host)
                    })
                else {
                    continue;
                };
                let new_replacement = new_commit.replacement().expect("not a base commit");
                let cost = repl_cost(new_replacement, host, |op| self.op_cost(op));
                let Some(new_commit_id) = space
                    .add_rewrite(new_commit, cost, self.get_name(&pattern_match, host))
                    .expect("constructed invalid commit")
                else {
                    continue;
                };
                explored_rewrites += 1;

                if options
                    .max_rewrites
                    .is_some_and(|max| explored_rewrites >= max)
                {
                    return;
                }

                // Explore up to PATTERN_RADIUS hops from the new nodes. Creating
                // new walkers from the new nodes should be as easy as using
                // Walker::from_pinned_node. However, there is a special
                // case when there are no new nodes. In that case, we need to make
                // PATTERN_RADIUS hops starting from the nodes of the subgraph
                // boundary.
                let mut new_nodes = space.inserted_nodes(new_commit_id).peekable();
                let nodes_walker_pairs = if new_nodes.peek().is_some() {
                    Either::Left(
                        new_nodes
                            .map(|n| (n, Walker::from_pinned_node(n, space.state_space.clone()))),
                    )
                } else {
                    // special case when replacing with no new nodes: make PATTERN_RADIUS hops
                    // starting from the nodes of the subgraph boundary.
                    let new_commit = space.get_commit(new_commit_id);
                    let new_replacement = new_commit.replacement().expect("not a base commit");
                    let subg_nodes = new_replacement.subgraph().nodes().iter().copied();
                    // nodes one hop away from the deleted subgraph
                    let neighbours = new_nodes_within_radius(
                        subg_nodes
                            .map(|n| (n, Walker::from_pinned_node(n, space.state_space.clone()))),
                        1,
                    )
                    .into_iter()
                    .map(|(n, _)| n);
                    // Create new walkers with the new (empty) commit pre-selected.
                    let nodes_walker_pairs = neighbours.filter_map(|n| {
                        let mut walker = Walker::from_pinned_node(n, space.state_space.clone());
                        walker.try_select_commit(new_commit.clone()).ok()?;
                        Some((n, walker))
                    });
                    let nodes_walker_pairs = nodes_walker_pairs.collect_vec();
                    Either::Right(nodes_walker_pairs.into_iter())
                };

                // enqueue new potential roots
                queue.extend(new_nodes_within_radius(
                    nodes_walker_pairs,
                    Self::PATTERN_RADIUS,
                ));
            }
        }
    }
}

/// Return all nodes within `pattern_radius` of `node` in the state space, to
/// serve as new potential roots in explore's walker queue
fn new_nodes_within_radius<'a>(
    nodes_walker: impl IntoIterator<Item = (PatchNode, Walker<'a>)>,
    pattern_radius: usize,
) -> Vec<(PatchNode, Walker<'a>)> {
    let mut new_walker_queue =
        VecDeque::from_iter(nodes_walker.into_iter().map(|(n, walker)| (n, walker, 0)));
    let mut all_new_walkers = vec![];
    let mut seen = BTreeSet::new();

    while let Some((node, walker, depth)) = new_walker_queue.pop_front() {
        if !seen.insert(node) {
            continue;
        }

        all_new_walkers.push((node, walker.clone()));
        if depth >= pattern_radius {
            continue;
        }

        for port in value_ports(walker.as_hugr_view(), node, None) {
            let wire = walker.get_wire(node, port);
            let already_pinned: BTreeSet<_> = walker.wire_pinned_ports(&wire, None).collect();
            for new_walker in walker.expand(&wire, None) {
                let new_wire = new_walker.get_wire(node, port);
                let Some((new_node, _)) = new_walker
                    .wire_pinned_ports(&new_wire, None)
                    .find(|np| !already_pinned.contains(np))
                else {
                    continue;
                };
                new_walker_queue.push_back((new_node, new_walker, depth + 1));
            }
        }
    }

    all_new_walkers
}

fn is_value_port<H: HugrView>(h: &H, node: H::Node, port: impl Into<Port>) -> bool {
    h.get_optype(node)
        .port_kind(port)
        .is_some_and(|v| v.is_value())
}

fn value_ports<H: HugrView>(
    host: &H,
    node: H::Node,
    dir: impl Into<Option<Direction>>,
) -> impl Iterator<Item = Port> + '_ {
    let ports = match dir.into() {
        Some(dir) => Either::Left(host.node_ports(node, dir)),
        None => Either::Right(host.all_node_ports(node)),
    };
    ports.filter(move |&port| is_value_port(host, node, port))
}

fn repl_cost<C: CircuitCost, H: HugrView>(
    replacement: &SimpleReplacement<H::Node>,
    host: &H,
    op_cost: impl Fn(&OpType) -> Option<C>,
) -> C::CostDelta {
    let repl = replacement.replacement();
    let subg = replacement.subgraph();

    let repl_cost: C = repl
        .nodes()
        .filter_map(|n| op_cost(repl.get_optype(n)))
        .sum();
    let subg_cost: C = subg
        .nodes()
        .iter()
        .filter_map(|&n| op_cost(host.get_optype(n)))
        .sum();

    repl_cost.sub_cost(&subg_cost)
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
