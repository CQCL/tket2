//! Explore the space of possible rewrites.

use chrono::{DateTime, Utc};
use delegate::delegate;
use hugr::{hugr::serialize::ExtensionsSeed, Hugr};
use hugr_core::hugr::persistent::{
    Commit, CommitId, CommitStateSpace, InvalidCommit, PatchNode, PersistentHugr,
};
use itertools::Itertools;
use max_sat::MaxSATSolver;
use std::collections::{BTreeMap, BTreeSet};

use crate::{circuit::cost::CostDelta, extension::REGISTRY};

mod explore;
mod max_sat;

pub use explore::{Explore, ExploreOptions, IterMatchedWires};

/// A space of possible rewrites.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RewriteSpace<C> {
    /// The state space of all rewrites.
    #[serde(deserialize_with = "deserialize_with_exts")]
    pub state_space: CommitStateSpace,
    metadata: BTreeMap<CommitId, RewriteMetadata<C>>,
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
struct RewriteMetadata<C> {
    rewrite_cost: C,
    timestamp: DateTime<Utc>,
}

impl<C> RewriteMetadata<C> {
    /// Create new rewrite metadata with the given cost and current timestamp
    fn with_current_time(rewrite_cost: C) -> Self {
        Self {
            rewrite_cost,
            timestamp: Utc::now(),
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
        self.metadata.get(&commit_id).map(|m| &m.rewrite_cost)
    }

    /// Get the timestamp of a rewrite.
    pub fn get_timestamp(&self, commit_id: CommitId) -> Option<DateTime<Utc>> {
        self.metadata.get(&commit_id).map(|m| m.timestamp)
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
    pub fn add_rewrite(&mut self, commit: Commit, cost: C) -> Result<CommitId, InvalidCommit> {
        let commit_id = self.state_space.try_add_commit(commit)?;
        self.metadata
            .insert(commit_id, RewriteMetadata::with_current_time(cost));
        Ok(commit_id)
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
                .map(|m| -m.rewrite_cost.as_isize())
                .unwrap_or(0);
            opt.set_weight(fmt_commit_id(commit_id), weight);
        }

        let model = opt.get_model()?;

        let selected_commits = self
            .state_space
            .all_commit_ids()
            .filter(|cm| model[&fmt_commit_id(*cm)]);

        match self.state_space.try_extract_hugr(selected_commits) {
            Ok(hugr) => Some(hugr),
            Err(e) => {
                println!("error: {e}");
                None
            }
        }
    }
}

fn fmt_commit_id(commit_id: CommitId) -> String {
    format!("{:?}", commit_id)
}

fn deserialize_with_exts<'de, D>(deserializer: D) -> Result<CommitStateSpace, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::DeserializeSeed;
    let seed = ExtensionsSeed::<CommitStateSpace>::new(&REGISTRY);
    seed.deserialize(deserializer)
}
