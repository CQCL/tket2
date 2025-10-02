//! Subgraphs of unsupported (non-encodable) nodes in the hugr, and their
//! encoding as barrier metadata in pytket circuits.

use std::collections::BTreeMap;

use hugr::core::HugrNode;
use hugr::envelope::EnvelopeConfig;
use hugr::hugr::views::SiblingSubgraph;
use hugr::package::Package;
use hugr::HugrView;

/// Pytket opgroup used to identify opaque barrier operations that encode standalone unsupported HUGR subgraphs.
///
/// See [`UnsupportedSubgraphPayload::Standalone`].
pub const OPGROUP_STANDALONE_UNSUPPORTED_HUGR: &str = "UNSUPPORTED_HUGR";

/// Pytket opgroup used to identify opaque barrier operations that encode external unsupported HUGR subgraphs.
///
/// See [`UnsupportedSubgraphPayload::External`].
pub const OPGROUP_EXTERNAL_UNSUPPORTED_HUGR: &str = "EXTERNAL_UNSUPPORTED_HUGR";

/// The ID of a subgraph in the Hugr.
#[derive(
    Debug,
    derive_more::Display,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    serde::Serialize,
    serde::Deserialize,
)]
#[serde(transparent)]
#[display("Subgraph#{id}", id = self.0)]
pub(super) struct SubgraphId(usize);

/// A set of subgraphs a HUGR that have been marked as _unsupported_ during a
/// pytket encoding.
#[derive(Debug, Clone)]
pub(super) struct UnsupportedSubgraphs<N> {
    /// Sets of subgraphs in the HUGR that have been encoded as opaque barriers
    /// in the pytket circuit.
    ///
    /// Subcircuits are identified in the barrier metadata by their ID. See [`SubgraphId`].
    opaque_subgraphs: BTreeMap<SubgraphId, SiblingSubgraph<N>>,
}

/// Payload for a pytket barrier metadata that indicates the barrier represents
/// an unsupported HUGR subgraph.
///
/// The payload may be standalone, carrying the encoded HUGR subgraph, or be a
/// reference to a subgraph tracked by a [`UnsupportedSubgraphs`] registry.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(super) enum UnsupportedSubgraphPayload {
    /// A standalone payload, carrying the encoded HUGR subgraph.
    Standalone { hugr_envelope: String },
    /// A reference to a subgraph tracked by a [`UnsupportedSubgraphs`] registry.
    External {
        /// The ID of the subgraph in the [`UnsupportedSubgraphs`] registry.
        id: SubgraphId,
    },
}

impl SubgraphId {
    /// Create a new [`SubgraphId`] from a [`SiblingSubgraph`].
    pub fn from_subgraph(subgraph: &SiblingSubgraph<impl HugrNode>) -> Self {
        // The encoder running on a hugr will only generate disjoint subgraphs
        // of unsupported nodes, so it's safe to identify them by their list of
        // nodes ids.
        //
        // It would even be safe to just pick a single node ID for the
        // identifier, but we compute the hash of the node set instead to be
        // more robust.
        Self(fxhash::hash(subgraph.nodes()))
    }
}

impl<N: HugrNode> UnsupportedSubgraphs<N> {
    /// Create a new [`UnsupportedSubgraphs`].
    pub fn new() -> Self {
        Self {
            opaque_subgraphs: BTreeMap::new(),
        }
    }

    /// Register a new unsupported subgraph in the Hugr.
    ///
    /// Returns and ID that can be used to identify the subgraph in the pytket circuit.
    pub fn register_unsupported_subgraph(&mut self, subgraph: SiblingSubgraph<N>) -> SubgraphId {
        let id = SubgraphId::from_subgraph(&subgraph);
        self.opaque_subgraphs.insert(id, subgraph);
        id
    }

    /// Returns the unsupported subgraph with the given ID.
    ///
    /// # Panics
    ///
    /// Panics if the ID is invalid.
    pub fn get_unsupported_subgraph(&self, id: SubgraphId) -> &SiblingSubgraph<N> {
        self.opaque_subgraphs
            .get(&id)
            .unwrap_or_else(|| panic!("Invalid subgraph ID: {id}"))
    }

    /// Returns `true` if the unsupported subgraph with the given ID exists.
    pub fn contains(&self, id: SubgraphId) -> bool {
        self.opaque_subgraphs.contains_key(&id)
    }

    /// Returns an iterator over the IDs of the unsupported subgraphs.
    pub fn ids(&self) -> impl Iterator<Item = SubgraphId> + '_ {
        self.opaque_subgraphs.keys().copied()
    }

    /// Merge another [`UnsupportedSubgraphs`] into this one.
    pub fn merge(&mut self, other: Self) {
        self.opaque_subgraphs.extend(other.opaque_subgraphs);
    }
}

impl UnsupportedSubgraphPayload {
    /// Create a new external payload, referencing a subgraph tracked by a
    /// [`UnsupportedSubgraphs`] registry.
    pub fn external(id: SubgraphId) -> Self {
        Self::External { id }
    }

    /// Create a new standalone payload, by extracting a sibling subgraph from a
    /// HUGR and encoding it.
    ///
    /// Note that this will produce incomplete hugrs when external function
    /// calls or  non-local edges are present.
    //
    // TODO: Detect and deal with non-local edges? What to do there is not
    // clear, we need further discussion.
    //
    // TODO: This should include descendants of the subgraph. Test that.
    pub fn standalone<N: HugrNode>(
        subgraph: &SiblingSubgraph<N>,
        hugr: &impl HugrView<Node = N>,
    ) -> Self {
        let unsupported_hugr = subgraph.extract_subgraph(hugr, "");
        let payload = Package::from_hugr(unsupported_hugr)
            .store_str(EnvelopeConfig::text())
            .unwrap();
        Self::Standalone {
            hugr_envelope: payload,
        }
    }
}
