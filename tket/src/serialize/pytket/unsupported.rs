//! Subgraphs of unsupported (non-encodable) nodes in the hugr, and their
//! encoding as barrier metadata in pytket circuits.

mod payload;

pub use payload::{
    EncodedEdgeID, UnsupportedSubgraphPayload, UnsupportedSubgraphPayloadType, OPGROUP_OPAQUE_HUGR,
};

use std::collections::BTreeMap;
use std::ops::Index;

use crate::serialize::pytket::PytketEncodeError;
use hugr::core::HugrNode;
use hugr::hugr::views::SiblingSubgraph;
use hugr::HugrView;

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
pub struct SubgraphId(usize);

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
    pub fn get(&self, id: SubgraphId) -> Option<&SiblingSubgraph<N>> {
        self.opaque_subgraphs.get(&id)
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

    /// If the pytket command is a barrier operation encoding an opaque subgraph, replace its [`UnsupportedSubgraphPayload::External`] pointer
    /// if present with a [`UnsupportedSubgraphPayload::Standalone`] payload.
    ///
    /// # Errors
    ///
    /// Returns an error if a barrier operation with the [`OPGROUP_OPAQUE_HUGR`] opgroup has an invalid payload.
    pub(super) fn replace_external_with_standalone(
        &self,
        command: &mut tket_json_rs::circuit_json::Command,
        hugr: &impl HugrView<Node = N>,
    ) -> Result<(), PytketEncodeError<N>> {
        if command.op.op_type != tket_json_rs::OpType::Barrier
            || command.opgroup.as_deref() != Some(OPGROUP_OPAQUE_HUGR)
        {
            return Ok(());
        }
        let Some(payload) = command.op.data.take() else {
            return Err(PytketEncodeError::custom(format!(
                "Barrier operation with opgroup {OPGROUP_OPAQUE_HUGR} has no data payload."
            )));
        };

        let Some(mut payload) = parse_external_payload(&payload)? else {
            // Standalone payload, nothing to do.
            return Ok(());
        };
        let UnsupportedSubgraphPayloadType::External { id: subgraph_id } = payload.typ else {
            unreachable!("Checked by `parse_external_payload`");
        };
        if !self.contains(subgraph_id) {
            return Err(PytketEncodeError::custom(format!("Barrier operation with opgroup {OPGROUP_OPAQUE_HUGR} points to an unknown subgraph: {subgraph_id}")));
        }

        payload.typ = UnsupportedSubgraphPayloadType::standalone(&self[subgraph_id], hugr);
        command.op.data = Some(serde_json::to_string(&payload).unwrap());

        Ok(())
    }
}

impl<N: HugrNode> Index<SubgraphId> for UnsupportedSubgraphs<N> {
    type Output = SiblingSubgraph<N>;

    fn index(&self, index: SubgraphId) -> &Self::Output {
        self.get(index)
            .unwrap_or_else(|| panic!("Invalid subgraph ID: {index}"))
    }
}

/// Parse an external payload from a string payload.
///
/// Returns `None` if the payload is a standalone payload. We avoid fully
/// decoding the payload in this case to avoid allocating a new String for the
/// encoded envelope.
///
/// # Errors
///
/// Returns an error if the payload is invalid.
fn parse_external_payload<N>(
    payload: &str,
) -> Result<Option<UnsupportedSubgraphPayload>, PytketEncodeError<N>> {
    let mk_serde_error = |e: serde_json::Error| {
        PytketEncodeError::custom(format!(
            "Barrier operation with opgroup {OPGROUP_OPAQUE_HUGR} has corrupt data payload: {e}"
        ))
    };

    // Check if the payload is a standalone payload, without fully copying it to memory.
    #[derive(serde::Deserialize)]
    struct PartialPayload {
        pub typ: String,
    }
    let partial_payload: PartialPayload = serde_json::from_str(payload).map_err(mk_serde_error)?;
    if partial_payload.typ == "Standalone" {
        return Ok(None);
    }

    let payload: UnsupportedSubgraphPayload =
        serde_json::from_str(payload).map_err(mk_serde_error)?;
    Ok(Some(payload))
}
