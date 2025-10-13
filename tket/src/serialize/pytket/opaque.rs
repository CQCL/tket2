//! Subgraphs of unsupported (non-encodable) nodes in the hugr, and their
//! encoding as barrier metadata in pytket circuits.

mod payload;

pub use payload::{
    EncodedEdgeID, OpaqueSubgraphPayload, OpaqueSubgraphPayloadType, OPGROUP_OPAQUE_HUGR,
};

use std::collections::BTreeMap;
use std::ops::Index;

use crate::serialize::pytket::PytketEncodeError;
use hugr::core::HugrNode;
use hugr::hugr::views::SiblingSubgraph;
use hugr::HugrView;

/// The ID of a subgraph in the Hugr.
#[derive(Debug, derive_more::Display, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[display("{local_id}.{tracker_id}")]
pub struct SubgraphId {
    /// A locally unique ID in the [`OpaqueSubgraphs`] instance.
    local_id: usize,
    /// The unique ID of the [`OpaqueSubgraphs`] instance that generated this ID.
    tracker_id: usize,
}

/// A set of subgraphs a HUGR that have been marked as _unsupported_ during a
/// pytket encoding.
#[derive(Debug, Clone)]
pub(super) struct OpaqueSubgraphs<N> {
    /// A unique ID for this instance of [`OpaqueSubgraphs`].
    ///
    /// New subgraphs encoded by this instance will have a globally unique ID
    /// composed of this index and a sequential ID.
    ///
    /// Note that the IDs in `opaque_subgraphs` may include instances with
    /// different tracker IDs if `merge` has been called.
    id: usize,
    /// Sets of subgraphs in the HUGR that have been encoded as opaque barriers
    /// in the pytket circuit.
    ///
    /// Subcircuits are identified in the barrier metadata by their ID. See [`SubgraphId`].
    opaque_subgraphs: BTreeMap<SubgraphId, SiblingSubgraph<N>>,
}

impl serde::Serialize for SubgraphId {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        (&self.local_id, &self.tracker_id).serialize(s)
    }
}

impl<'de> serde::Deserialize<'de> for SubgraphId {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let (local_id, tracker_id) = serde::Deserialize::deserialize(d)?;
        Ok(Self {
            local_id,
            tracker_id,
        })
    }
}

impl<N: HugrNode> OpaqueSubgraphs<N> {
    /// Create a new [`OpaqueSubgraphs`].
    ///
    /// # Arguments
    ///
    /// - `id`: A unique ID for this instance of [`OpaqueSubgraphs`], used
    ///   to generate new globally unique subgraph IDs.
    pub fn new(id: usize) -> Self {
        Self {
            id,
            opaque_subgraphs: BTreeMap::new(),
        }
    }

    /// Register a new opaque subgraph in the Hugr.
    ///
    /// Returns and ID that can be used to identify the subgraph in the pytket circuit.
    pub fn register_opaque_subgraph(&mut self, subgraph: SiblingSubgraph<N>) -> SubgraphId {
        let id = SubgraphId {
            local_id: self.opaque_subgraphs.len(),
            tracker_id: self.id,
        };
        self.opaque_subgraphs.insert(id, subgraph);
        id
    }

    /// Returns the opaque subgraph with the given ID.
    ///
    /// # Panics
    ///
    /// Panics if the ID is invalid.
    pub fn get(&self, id: SubgraphId) -> Option<&SiblingSubgraph<N>> {
        self.opaque_subgraphs.get(&id)
    }

    /// Returns `true` if the opaque subgraph with the given ID exists.
    pub fn contains(&self, id: SubgraphId) -> bool {
        self.opaque_subgraphs.contains_key(&id)
    }

    /// Returns an iterator over the IDs of the opaque subgraphs.
    pub fn ids(&self) -> impl Iterator<Item = SubgraphId> + '_ {
        self.opaque_subgraphs.keys().copied()
    }

    /// Merge another [`OpaqueSubgraphs`] into this one.
    pub fn merge(&mut self, other: Self) {
        self.opaque_subgraphs.extend(other.opaque_subgraphs);
    }

    /// If the pytket command is a barrier operation encoding an opaque subgraph, replace its [`OpaqueSubgraphPayload::External`] pointer
    /// if present with a [`OpaqueSubgraphPayload::Inline`] payload.
    ///
    /// # Errors
    ///
    /// Returns an error if a barrier operation with the [`OPGROUP_OPAQUE_HUGR`] opgroup has an invalid payload.
    pub(super) fn inline_payload(
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
            // Inline payload, nothing to do.
            return Ok(());
        };
        let OpaqueSubgraphPayloadType::External { id: subgraph_id } = payload.typ else {
            unreachable!("Checked by `parse_external_payload`");
        };
        if !self.contains(subgraph_id) {
            return Err(PytketEncodeError::custom(format!("Barrier operation with opgroup {OPGROUP_OPAQUE_HUGR} points to an unknown subgraph: {subgraph_id}")));
        }

        payload.typ = OpaqueSubgraphPayloadType::inline(&self[subgraph_id], hugr);
        command.op.data = Some(serde_json::to_string(&payload).unwrap());

        Ok(())
    }
}

impl<N: HugrNode> Index<SubgraphId> for OpaqueSubgraphs<N> {
    type Output = SiblingSubgraph<N>;

    fn index(&self, index: SubgraphId) -> &Self::Output {
        self.get(index)
            .unwrap_or_else(|| panic!("Invalid subgraph ID: {index}"))
    }
}

impl<N> Default for OpaqueSubgraphs<N> {
    fn default() -> Self {
        Self {
            id: 0,
            opaque_subgraphs: BTreeMap::new(),
        }
    }
}

/// Parse an external payload from a string payload.
///
/// Returns `None` if the payload is inline. We avoid fully decoding the payload
/// in this case to avoid allocating a new String for the encoded envelope.
///
/// # Errors
///
/// Returns an error if the payload is invalid.
fn parse_external_payload<N>(
    payload: &str,
) -> Result<Option<OpaqueSubgraphPayload>, PytketEncodeError<N>> {
    let mk_serde_error = |e: serde_json::Error| {
        PytketEncodeError::custom(format!(
            "Barrier operation with opgroup {OPGROUP_OPAQUE_HUGR} has corrupt data payload: {e}"
        ))
    };

    // Check if the payload is inline, without fully copying it to memory.
    #[derive(serde::Deserialize)]
    struct PartialPayload {
        pub typ: String,
    }
    let partial_payload: PartialPayload = serde_json::from_str(payload).map_err(mk_serde_error)?;
    if partial_payload.typ == "Inline" {
        return Ok(None);
    }

    let payload: OpaqueSubgraphPayload = serde_json::from_str(payload).map_err(mk_serde_error)?;
    Ok(Some(payload))
}
