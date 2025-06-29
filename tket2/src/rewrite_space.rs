//! Explore the space of possible rewrites using commit factories.

use chrono::{DateTime, Utc};
use delegate::delegate;
use hugr::persistent::{Commit, CommitId, InvalidCommit, PatchNode, SerdeHashResolver};
use hugr::Hugr;
use itertools::Itertools;
use max_sat::MaxSATSolver;
use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, BTreeSet};

use crate::circuit::cost::CostDelta;
use crate::serialize::HugrWithExts;

mod factory;
mod max_sat;

pub use factory::{CliffordSimpFactory, CliffordSubcircuit, CommuteCZ, CommuteCZFactory};
pub use factory::{CommitFactory, ExploreOptions, IterMatched};

/// A walker over the rewrite space.
pub type Walker<'a> = hugr::persistent::Walker<'a, Resolver>;
/// The resolver used in [`RewriteSpace`].
pub type Resolver = SerdeHashResolver<HugrWithExts>;

/// The state space of all possible rewrites.
pub type CommitStateSpace = hugr::persistent::CommitStateSpace<Resolver>;
/// A persistent Hugr using the default TKET2 resolver.
pub type PersistentHugr = hugr::persistent::PersistentHugr<Resolver>;

/// A space of possible rewrites.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RewriteSpace<C> {
    /// The state space of all possible rewrites.
    #[serde(with = "serial")]
    pub state_space: CommitStateSpace,
    /// The metadata for each commit in the state space.
    metadata: BTreeMap<CommitId, CommitMetadata<C>>,
}

impl<C> From<CommitStateSpace> for RewriteSpace<C> {
    fn from(state_space: CommitStateSpace) -> Self {
        Self {
            state_space,
            metadata: Default::default(),
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct CommitMetadata<C> {
    cost: C,
    timestamp: DateTime<Utc>,
    name: String,
}

impl<C> CommitMetadata<C> {
    /// Create new rewrite metadata with the given cost, name and current
    /// timestamp
    fn with_current_time(rewrite_cost: C, name: String) -> Self {
        Self {
            cost: rewrite_cost,
            timestamp: Utc::now(),
            name,
        }
    }
}

impl<C> RewriteSpace<C> {
    /// Create a new rewrite space with the given base Hugr.
    pub fn with_base(base: Hugr) -> Self {
        let state_space = CommitStateSpace::with_base(base);
        Self {
            state_space,
            metadata: BTreeMap::new(),
        }
    }

    /// Get the cost of a rewrite.
    pub fn get_cost(&self, commit_id: CommitId) -> Option<&C> {
        self.metadata.get(&commit_id).map(|m| &m.cost)
    }

    /// Get the timestamp of a rewrite.
    pub fn get_timestamp(&self, commit_id: CommitId) -> Option<DateTime<Utc>> {
        self.metadata.get(&commit_id).map(|m| m.timestamp)
    }

    /// Get the name of a rewrite.
    pub fn get_name(&self, commit_id: CommitId) -> Option<&str> {
        self.metadata.get(&commit_id).map(|m| m.name.as_str())
    }

    delegate! {
        to self.state_space {
            /// Get all commit IDs in the space.
            pub fn all_commit_ids(&self) -> impl Iterator<Item = CommitId> + '_;
            /// Get all nodes inserted by a commit.
            pub fn inserted_nodes(&self, commit_id: CommitId) -> impl Iterator<Item = PatchNode> + '_;
            /// Get a commit by its ID.
            pub fn get_commit(&self, commit_id: CommitId) -> &Commit;
            /// Get the base commit ID.
            pub fn base(&self) -> CommitId;
        }
    }
}

impl<C: CostDelta> RewriteSpace<C> {
    /// Add a new rewrite to the space.
    ///
    /// If the commit already existed in `self`, `None` is returned.
    pub fn add_rewrite(
        &mut self,
        commit: Commit,
        cost: C,
        name: String,
    ) -> Result<Option<CommitId>, InvalidCommit> {
        let commit_id = self.state_space.try_add_commit(commit)?;
        if let Entry::Vacant(e) = self.metadata.entry(commit_id) {
            e.insert(CommitMetadata::with_current_time(cost, name));
            Ok(Some(commit_id))
        } else {
            Ok(None)
        }
    }

    /// Find the best sequence of rewrites to apply to the base Hugr.
    pub fn extract_best(&self) -> Option<PersistentHugr> {
        let mut opt = MaxSATSolver::new();
        for commit_id in self.state_space.all_commit_ids() {
            // Add child => parent implications
            opt.extend_implications(
                self.state_space
                    .parents(commit_id)
                    .map(|cm| (fmt_commit_id(commit_id), fmt_commit_id(cm))),
            );

            // Add mutual exclusion between siblings if their invalidation sets overlap
            let children = self.state_space.children(commit_id);
            for children_pair in children.combinations(2) {
                let &[c1, c2] = children_pair.as_slice() else {
                    panic!("Expected 2 children, got {}", children_pair.len());
                };
                let del_by_c1: BTreeSet<_> =
                    self.state_space.invalidation_set(c1, commit_id).collect();
                let mut del_by_c2 = self.state_space.invalidation_set(c2, commit_id);
                if del_by_c2.any(|n| del_by_c1.contains(&n)) {
                    opt.add_mutex(fmt_commit_id(c1), fmt_commit_id(c2));
                }
            }

            // Add cost for each commit
            let weight = self
                .metadata
                .get(&commit_id)
                .map(|m| -m.cost.as_isize())
                .unwrap_or(0);
            opt.set_weight(fmt_commit_id(commit_id), weight);
        }

        let model = opt.get_model()?;

        let selected_commits = self
            .state_space
            .all_commit_ids()
            .filter(|cm| model[&fmt_commit_id(*cm)]);

        self.state_space.try_extract_hugr(selected_commits).ok()
    }
}

fn fmt_commit_id(commit_id: CommitId) -> String {
    format!("{commit_id}")
}

mod serial {
    use crate::serialize::HugrWithExts;

    use super::{CommitStateSpace, Resolver};
    use hugr::persistent::state_space::serial::SerialCommitStateSpace;
    use serde::Serialize;

    pub fn deserialize<'de, D>(deserializer: D) -> Result<CommitStateSpace, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value: SerialCommitStateSpace<HugrWithExts, Resolver> =
            serde::Deserialize::deserialize(deserializer)?;
        Ok(CommitStateSpace::from_serial(value))
    }

    pub fn serialize<S>(value: &CommitStateSpace, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let value: SerialCommitStateSpace<HugrWithExts, Resolver> = value.to_serial();
        value.serialize(serializer)
    }
}
