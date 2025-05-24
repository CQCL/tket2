use std::collections::{BTreeSet, VecDeque};

use hugr::{ops::OpType, Hugr, HugrView, Port, SimpleReplacement};
use hugr_core::hugr::persistent::{CommitStateSpace, PatchNode, PinnedWire, Walker};
use itertools::{Either, Itertools};

use crate::circuit::cost::CircuitCost;

use super::RewriteSpace;

/// Options for the exploration process.
#[derive(Debug, Clone, Default)]
pub struct ExploreOptions {
    /// The maximum number of rewrites to explore.
    pub max_rewrites: Option<usize>,
}

/// A trait for exploring the rewrite space.
///
/// This trait captures two main capabilities of an space explorer object:
///  - find patterns of interest in the space (see
///    [Explore::find_pattern_matches])
///  - construct new rewrite candidates from pattern matches, (see
///    [Explore::get_replacement], [Explore::map_boundary] and
///    [Explore::op_cost])
///
/// Given these capabilities, the [Explore::explore] method is provided to
/// iteratively explore and expand the space.
pub trait Explore {
    /// A record of a pattern match during exploration.
    type PatternMatch: IterMatchedWires;
    /// The cost type attributed to a rewrite.
    type Cost: CircuitCost;

    /// Maximum distance between a newly inserted node and a new root node
    /// candidate.
    ///
    /// When new nodes are added to the space, all nodes within this radius will
    /// be considered as potential new roots for the next round of
    /// exploration.
    ///
    /// This should typically be set to the maximum distance between the pattern
    /// root and a node of the pattern.
    const PATTERN_RADIUS: usize;

    /// Get the replacement HUGR for a pattern match.
    fn get_replacement(&self, pattern_match: &Self::PatternMatch) -> Option<Hugr>;

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
    fn map_boundary(&self, node: PatchNode, port: Port, pattern_match: &Self::PatternMatch)
        -> Port;

    /// Find all pattern matches in the space with `pattern_root` as their root.
    fn find_pattern_matches<'w: 'a, 'a>(
        &'a self,
        pattern_root: PatchNode,
        walker: Walker<'w>,
    ) -> impl Iterator<Item = (Self::PatternMatch, Walker<'w>)> + 'a;

    /// Get the cost of an operation.
    fn op_cost(&self, op: &OpType) -> Option<Self::Cost>;

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
            for (pattern_match, subwalker) in self.find_pattern_matches(node, walker) {
                if let Some(replacement) = self.get_replacement(&pattern_match) {
                    let Ok(new_commit) = subwalker.try_create_commit(
                        pattern_match.matched_wires().cloned(),
                        replacement,
                        |node, port| self.map_boundary(node, port, &pattern_match),
                    ) else {
                        continue;
                    };
                    let new_replacement = new_commit.replacement().expect("not a base commit");
                    let cost = repl_cost(new_replacement, subwalker.as_hugr_view(), |op| {
                        self.op_cost(op)
                    });
                    let new_commit_id = space
                        .add_rewrite(new_commit, cost)
                        .expect("constructed invalid commit");
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
                            new_nodes.map(|n| {
                                (n, Walker::from_pinned_node(n, space.state_space.clone()))
                            }),
                        )
                    } else {
                        // special case when replacing with no new nodes: make PATTERN_RADIUS hops
                        // starting from the nodes of the subgraph boundary.
                        let new_commit = space.get_commit(new_commit_id);
                        let new_replacement = new_commit.replacement().expect("not a base commit");
                        let subg_nodes = new_replacement.subgraph().nodes().iter().copied();
                        // nodes one hop away from the deleted subgraph
                        let neighbours = new_nodes_within_radius(
                            subg_nodes.map(|n| {
                                (n, Walker::from_pinned_node(n, space.state_space.clone()))
                            }),
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
                        // dbg!(&nodes_walker_pairs);
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

        for port in walker.as_hugr_view().all_node_ports(node) {
            if !is_value_port(walker.as_hugr_view(), node, port) {
                continue;
            }

            let wire = walker.get_wire(node, port);
            let already_pinned: BTreeSet<_> = wire.all_pinned_ports().collect();
            for new_walker in walker.expand(&wire, None) {
                let new_wire = new_walker.get_wire(node, port);
                if let Some((new_node, _)) = new_wire
                    .all_pinned_ports()
                    .find(|np| !already_pinned.contains(np))
                {
                    new_walker_queue.push_back((new_node, new_walker, depth + 1));
                };
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

fn repl_cost<C: CircuitCost, H: HugrView>(
    replacement: &SimpleReplacement<H::Node>,
    host: &H,
    op_cost: impl Fn(&OpType) -> Option<C>,
) -> C::CostDelta {
    let repl = replacement.replacement();
    let subg = replacement.subgraph();

    let repl_cost: C = repl
        .nodes()
        .filter_map(|n| op_cost(&repl.get_optype(n)))
        .sum();
    let subg_cost: C = subg
        .nodes()
        .iter()
        .filter_map(|&n| op_cost(&host.get_optype(n)))
        .sum();

    repl_cost.sub_cost(&subg_cost)
}

/// A trait for iterating over the wires that make up a pattern match.
pub trait IterMatchedWires {
    /// Iterate over the wires that make up a pattern match.
    fn matched_wires(&self) -> impl Iterator<Item = &PinnedWire> + '_;
}

impl IterMatchedWires for Vec<PinnedWire> {
    fn matched_wires(&self) -> impl Iterator<Item = &PinnedWire> + '_ {
        self.iter()
    }
}
