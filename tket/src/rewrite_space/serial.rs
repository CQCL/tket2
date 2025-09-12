use std::cell::RefCell;

use hugr::persistent::serial::SerialCommitStateSpace;
use hugr::persistent::{CommitId, PersistentHugr};
use slotmap_fork_lmondada as slotmap;

use crate::rewrite_space::{CommitMetadata, RewriteSpace};
use crate::serialize::HugrWithExts;

/// A serialisable [`RewriteSpace`], along with the HUGRs within it to be
/// serialised.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SerialRewriteSpace<C> {
    /// The state space of all possible rewrites.
    pub state_space: SerialCommitStateSpace<HugrWithExts>,
    /// The metadata for each commit in the state space.
    pub metadata: slotmap::SecondaryMap<CommitId, CommitMetadata<C>>,
}

impl<C: Clone + serde::Serialize> RewriteSpace<C> {
    /// Serialize the rewrite space into a serializable format.
    pub fn serialize(&self, hugrs: Vec<PersistentHugr>) -> SerialRewriteSpace<C> {
        let mut ser_state_space = SerialCommitStateSpace::new(&self.state_space);
        for hugr in hugrs {
            ser_state_space.add_hugr(hugr);
        }
        SerialRewriteSpace {
            state_space: ser_state_space,
            metadata: self.metadata.borrow().clone(),
        }
    }
}

impl<'de, C: Clone + serde::Deserialize<'de>> RewriteSpace<C> {
    /// Replace the current rewrite space with the one obtained from the
    /// serialised format.
    pub fn replace_with(&mut self, serial: SerialRewriteSpace<C>) -> Vec<PersistentHugr> {
        let hugrs = serial.state_space.deserialize_into_hugrs();
        let metadata = RefCell::new(serial.metadata);

        self.state_space = hugrs.first().unwrap().state_space().clone();
        self.metadata = metadata;

        hugrs
    }
}
