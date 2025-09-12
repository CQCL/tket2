//! Explore the space of possible rewrites using commit factories.

use chrono::{DateTime, Utc};
use derive_more::derive::Display;
use derive_more::derive::Error;
use derive_more::derive::From;
use derive_more::derive::Into;
use hugr::hugr::views::NodesIter;
use hugr::persistent as hugr_im;
use itertools::Itertools;
use slotmap_fork_lmondada as slotmap;

use std::cell::Ref;
use std::cell::RefCell;
use std::collections::BTreeSet;
use std::collections::VecDeque;

use crate::circuit::cost::CostDelta;
use crate::resource::ResourceScope;
use crate::rewrite::matcher::all_linear_ports;
use crate::rewrite::CircuitRewrite;
use max_sat::MaxSATSolver;

// mod factory;
mod max_sat;
mod serial;
pub use serial::SerialRewriteSpace;

/// A rewrite in a [`RewriteSpace`].
#[derive(Debug, Clone, From, Into)]
pub struct CommittedRewrite<'a>(hugr_im::Commit<'a>);

/// A space of possible rewrites.
#[derive(Debug, Clone)]
pub struct RewriteSpace<C> {
    /// The state space of all possible rewrites.
    state_space: hugr_im::CommitStateSpace,
    /// The metadata for each commit in the state space.
    metadata: RefCell<slotmap::SecondaryMap<hugr_im::CommitId, CommitMetadata<C>>>,
}

/// Metadata for a commit in a [`RewriteSpace`].
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct CommitMetadata<C> {
    cost: C,
    timestamp: DateTime<Utc>,
}

impl<C> CommitMetadata<C> {
    /// Create new rewrite metadata with the given cost and current timestamp
    fn with_current_time(rewrite_cost: C) -> Self {
        Self {
            cost: rewrite_cost,
            timestamp: Utc::now(),
        }
    }
}

/// An error that can occur when committing a rewrite to a [`RewriteSpace`].
#[derive(Debug, Clone, Display, From, Error)]
#[non_exhaustive]
pub enum InvalidCommit {
    /// The commit is invalid.
    HugrInvalidCommit(#[from] hugr_im::InvalidCommit),
    /// The circuit belongs to a different rewrite space.
    #[display("The rewrite space of circuit does not match self")]
    StateSpaceMismatch,
}

impl<C> RewriteSpace<C> {
    /// Create a new [`RewriteSpace`] from an existing state space.
    pub fn from_state_space(state_space: hugr_im::CommitStateSpace) -> Self {
        Self {
            state_space,
            metadata: RefCell::new(slotmap::SecondaryMap::new()),
        }
    }

    /// Create a new empty [`RewriteSpace`].
    pub fn new() -> Self {
        Self {
            state_space: hugr_im::CommitStateSpace::new(),
            metadata: RefCell::new(slotmap::SecondaryMap::new()),
        }
    }

    /// Set the base hugr of the rewrite space.
    ///
    /// This will only succeed if the rewrite space is currently empty
    /// (otherwise a base hugr already exists).
    pub fn try_set_base(&self, hugr: hugr::Hugr, cost: C) -> Option<CommittedRewrite<'_>> {
        let cm = self.state_space.try_set_base(hugr)?;
        self.metadata
            .borrow_mut()
            .insert(cm.id(), CommitMetadata::with_current_time(cost));
        Some(cm.into())
    }

    /// Add a rewrite to the [`RewriteSpace`].
    ///
    /// The set of parents is inferred from the set of deleted parent nodes of
    /// the rewrite.
    pub fn try_add_rewrite(
        &self,
        rewrite: CircuitRewrite<hugr_im::PatchNode>,
        cost: C,
        circuit: &ResourceScope<&hugr_im::PersistentHugr>,
    ) -> Result<CommittedRewrite<'_>, InvalidCommit> {
        if circuit.hugr().state_space() != &self.state_space {
            return Err(InvalidCommit::StateSpaceMismatch);
        }
        let rewrite = rewrite.to_simple_replacement(circuit);
        let commit = hugr_im::Commit::try_from_replacement(rewrite, &self.state_space)?;
        self.metadata
            .borrow_mut()
            .insert(commit.id(), CommitMetadata::with_current_time(cost));
        Ok(CommittedRewrite(commit))
    }

    /// Add a commit to the [`RewriteSpace`].
    pub fn add_from_commit<'a>(
        &self,
        commit: hugr_im::Commit<'a>,
        cost: C,
    ) -> CommittedRewrite<'a> {
        self.metadata
            .borrow_mut()
            .insert(commit.id(), CommitMetadata::with_current_time(cost));
        CommittedRewrite(commit)
    }

    /// Get the cost of a rewrite.
    pub fn get_cost(&self, commit: &CommittedRewrite) -> Ref<'_, C> {
        let commit_id = commit.0.id();
        debug_assert!(self.metadata.borrow().contains_key(commit_id));

        Ref::map(self.metadata.borrow(), |m| &m.get(commit_id).unwrap().cost)
    }

    /// Get the timestamp of a rewrite.
    pub fn get_timestamp(&self, commit_id: hugr_im::CommitId) -> Option<DateTime<Utc>> {
        self.metadata.borrow().get(commit_id).map(|m| m.timestamp)
    }

    /// Get the state space of the [`RewriteSpace`].
    pub fn state_space(&self) -> &hugr_im::CommitStateSpace {
        &self.state_space
    }
}

impl<C> RewriteSpace<C> {
    /// Find the best sequence of rewrites to apply to the base Hugr.
    pub fn extract_best(&self) -> Option<hugr_im::PersistentHugr>
    where
        C: CostDelta,
    {
        self.extract_best_with_cost(|c| c.clone())
    }

    /// Find the best sequence of rewrites to apply to the base Hugr that
    /// minimises the given cost function.
    pub fn extract_best_with_cost<D: CostDelta>(
        &self,
        cost_fn: impl Fn(&C) -> D,
    ) -> Option<hugr_im::PersistentHugr> {
        let mut opt = MaxSATSolver::new();
        let base_commit_id = self.state_space.base_commit()?.id();
        for (commit_id, commit) in self.state_space.all_commits() {
            // Add child => parent implications
            opt.extend_implications(
                commit
                    .parents()
                    .map(|cm| (fmt_commit_id(commit_id), fmt_commit_id(cm.id()))),
            );

            // Add mutual exclusion between siblings if their invalidation sets overlap
            let children = commit.children(&self.state_space);
            for children_pair in children.combinations(2) {
                let [c1, c2] = children_pair.as_slice() else {
                    panic!("expected two children");
                };
                let del_by_c1: BTreeSet<_> = c1
                    .deleted_parent_nodes()
                    .filter(|n| n.owner() == commit_id)
                    .collect();
                let mut del_by_c2 = c2.deleted_parent_nodes().filter(|n| n.owner() == commit_id);
                if del_by_c2.any(|n| del_by_c1.contains(&n)) {
                    opt.add_mutex(fmt_commit_id(c1.id()), fmt_commit_id(c2.id()));
                }
            }

            // Add cost for each commit
            let mut weight = self
                .metadata
                .borrow()
                .get(commit_id)
                .map(|m| -cost_fn(&m.cost).as_isize())
                .unwrap_or_default();

            // Ensure that the base commit is always selected
            if commit_id == base_commit_id && weight == 0 {
                weight = 1;
            }
            opt.set_weight(fmt_commit_id(commit_id), weight);
        }

        let model = opt.get_model()?;

        let selected_commit_ids = self
            .state_space
            .all_commits()
            .into_iter()
            .map(|(cm, _)| cm)
            .filter(|&cm| model.get(&fmt_commit_id(cm)).copied().unwrap_or_default());

        self.state_space.try_create(selected_commit_ids).ok()
    }

    /// Return all nodes within `pattern_radius` of `node` in the rewrite space.
    ///
    /// For each returned node, a walker that pins the path between the starting
    /// node and the returned node is also returned.
    pub fn nodes_within_radius(
        &self,
        nodes: impl IntoIterator<Item = hugr_im::PatchNode>,
        pattern_radius: usize,
    ) -> Vec<(hugr_im::PatchNode, hugr_im::Walker<'_>)> {
        let mut new_walker_queue = VecDeque::from_iter(nodes.into_iter().map(|n| {
            let walker = hugr_im::Walker::from_pinned_node(n, &self.state_space);
            (n, walker, 0)
        }));
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

            for port in all_linear_ports(walker.as_hugr_view(), node) {
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
}

impl<C> NodesIter for RewriteSpace<C> {
    type Node = hugr_im::PatchNode;

    fn nodes(&self) -> impl Iterator<Item = Self::Node> {
        self.state_space
            .all_commits()
            .into_iter()
            .flat_map(|(id, cm)| {
                cm.inserted_nodes()
                    .map(|n| hugr_im::PatchNode(id, n))
                    .collect_vec()
            })
    }
}

impl<'c, C> NodesIter for &'c RewriteSpace<C> {
    type Node = hugr_im::PatchNode;

    fn nodes(&self) -> impl Iterator<Item = Self::Node> {
        self.state_space
            .all_commits()
            .into_iter()
            .flat_map(|(id, cm)| {
                cm.inserted_nodes()
                    .map(|n| hugr_im::PatchNode(id, n))
                    .collect_vec()
            })
    }
}

fn fmt_commit_id(commit_id: hugr_im::CommitId) -> String {
    format!("{commit_id:?}")
}
