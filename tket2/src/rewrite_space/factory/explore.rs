use std::collections::{BTreeSet, VecDeque};

use hugr::{
    ops::OpType,
    persistent::{Commit, PatchNode},
    Direction, HugrView, Port,
};
use itertools::{Either, Itertools};

use crate::{
    circuit::cost::CircuitCost,
    rewrite::strategy::{LexicographicCostFunction, StrategyCost},
    rewrite_space::{CommitStateSpace, RewriteSpace, Walker},
};

use super::FindCommits;

type DefaultCostFn = LexicographicCostFunction<fn(&OpType) -> usize, 2>;

/// Options to control the exploration process within commit factories.
#[derive(Debug, Clone)]
pub struct ExploreOptions<S = DefaultCostFn> {
    /// The maximum number of rewrites to explore.
    pub max_rewrites: Option<usize>,
    /// The maximum number of hops to explore from a new node.
    pub pattern_radius: usize,
    /// The cost function for operations.
    ///
    /// Currently, only support for cx_gate
    pub cost_fn: S,
}

impl Default for ExploreOptions<DefaultCostFn> {
    fn default() -> Self {
        Self {
            max_rewrites: None,
            pattern_radius: 4,
            cost_fn: DefaultCostFn::cx_count(),
        }
    }
}

impl<C> RewriteSpace<C> {
    /// Explore and expand the rewrite space.
    ///
    /// This method will iteratively explore the space, finding pattern matches
    /// and constructing new rewrite candidates.
    pub fn explore<S: StrategyCost>(
        &mut self,
        commit_factory: impl FindCommits,
        options: &ExploreOptions<S>,
    ) where
        S::OpCost: CircuitCost<CostDelta = C>,
    {
        let mut queue = VecDeque::new();
        let mut explored_rewrites = 0;

        // enqueue all modes of all commits
        for cm in self.all_commit_ids() {
            for node in self.inserted_nodes(cm) {
                let walker = Walker::from_pinned_node(node, self.state_space.clone());
                queue.push_back((node, walker));
            }
        }

        while let Some((node, walker)) = queue.pop_front() {
            for new_commit in commit_factory.find_commits(node, walker) {
                let cost = commit_cost(&new_commit, &self.state_space, |op| {
                    options.cost_fn.op_cost(op)
                })
                .expect("not a base commit");
                let Some(new_commit_id) = self
                    .add_rewrite(new_commit, cost)
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

                // Explore up to options.pattern_radius hops from the new nodes. Creating
                // new walkers from the new nodes should be as easy as using
                // Walker::from_pinned_node. However, there is a special
                // case when there are no new nodes. In that case, we need to make
                // options.pattern_radius hops starting from the nodes of the subgraph
                // boundary.
                let mut new_nodes = self.inserted_nodes(new_commit_id).peekable();
                let nodes_walker_pairs = if new_nodes.peek().is_some() {
                    Either::Left(
                        new_nodes
                            .map(|n| (n, Walker::from_pinned_node(n, self.state_space.clone()))),
                    )
                } else {
                    // special case when replacing with no new nodes: make hops
                    // starting from the nodes of the subgraph boundary.
                    let new_commit = self.get_commit(new_commit_id);
                    let new_replacement = new_commit.replacement().expect("not a base commit");
                    let subg_nodes = new_replacement.subgraph().nodes().iter().copied();
                    // nodes one hop away from the deleted subgraph
                    let neighbours = new_nodes_within_radius(
                        subg_nodes
                            .map(|n| (n, Walker::from_pinned_node(n, self.state_space.clone()))),
                        1,
                    )
                    .into_iter()
                    .map(|(n, _)| n);
                    // Create new walkers with the new (empty) commit pre-selected.
                    let nodes_walker_pairs = neighbours.filter_map(|n| {
                        let mut walker = Walker::from_pinned_node(n, self.state_space.clone());
                        walker.try_select_commit(new_commit.clone()).ok()?;
                        Some((n, walker))
                    });
                    let nodes_walker_pairs = nodes_walker_pairs.collect_vec();

                    Either::Right(nodes_walker_pairs.into_iter())
                };

                // enqueue new potential roots
                queue.extend(new_nodes_within_radius(
                    nodes_walker_pairs,
                    options.pattern_radius,
                ));
            }
        }
    }
}

fn commit_cost<C: CircuitCost>(
    commit: &Commit,
    state_space: &CommitStateSpace,
    op_cost: impl Fn(&OpType) -> C,
) -> Option<C::CostDelta> {
    let repl = commit.replacement()?.replacement();
    let deleted_nodes = commit.deleted_nodes();

    let repl_cost: C = repl.nodes().map(|n| op_cost(repl.get_optype(n))).sum();
    let subtract_cost: C = deleted_nodes
        .map(|n| op_cost(state_space.get_optype(n)))
        .sum();

    Some(repl_cost.sub_cost(&subtract_cost))
}

/// Return all nodes within `pattern_radius` of `node` in the state space, to
/// serve as new potential roots in explore's walker queue
pub(super) fn new_nodes_within_radius<'a>(
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

pub(super) fn value_ports<H: HugrView>(
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

fn is_value_port<H: HugrView>(h: &H, node: H::Node, port: impl Into<Port>) -> bool {
    h.get_optype(node)
        .port_kind(port)
        .is_some_and(|v| v.is_value())
}
