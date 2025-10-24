//! Definitions of the payloads for opaque barrier metadata in pytket circuits.

use hugr::core::HugrNode;
use hugr::envelope::{EnvelopeConfig, EnvelopeError};
use hugr::extension::resolution::{resolve_type_extensions, WeakExtensionRegistry};
use hugr::extension::{ExtensionRegistry, ExtensionRegistryLoadError};
use hugr::hugr::views::SiblingSubgraph;
use hugr::package::Package;
use hugr::types::Type;
use hugr::{HugrView, Wire};

use crate::serialize::pytket::{PytketDecodeError, PytketDecodeErrorInner};

use super::SubgraphId;

/// Pytket opgroup used to identify opaque barrier operations that encode opaque HUGR subgraphs.
///
/// See [`OpaqueSubgraphPayload`].
pub const OPGROUP_OPAQUE_HUGR: &str = "OPAQUE_HUGR";

/// Identifier for a wire in the Hugr, encoded as a 64-bit hash that is
/// detached from the node IDs of the in-memory Hugr.
///
/// These are used to identify edges in the [`OpaqueSubgraphPayload`]
/// payloads encoded in opaque barriers on the encoded pytket circuits.
///
/// We require them to reconstruct the edges of the hugr that are not reflected
/// as pytket register/parameter dependencies. This is the case for edges with
/// unsupported types between opaque subgraphs, or between an opaque
/// subgraph and a HUGR input/output node.
#[derive(
    Debug,
    derive_more::Display,
    Default,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    serde::Serialize,
    serde::Deserialize,
)]
#[display("Edge#{id}", id = self.0)]
#[serde(transparent)]
pub struct EncodedEdgeID(u64);

impl EncodedEdgeID {
    /// Create a new subgraph hyper edge ID by hashing a Hugr wire.
    pub fn new<N: HugrNode>(wire: Wire<N>) -> Self {
        let hash = fxhash::hash64(&wire);
        Self(hash)
    }
}

/// Payload for an optional barrier added at the end of the pytket circuit,
/// encoding additional circuit information required to decode the unsupported bits
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ExtraCircuitDataPayload {
    /// Types at the circuit's inputs, with their [`SubgraphHyperEdge`] IDs.
    inputs: Vec<(Type, EncodedEdgeID)>,
    /// Types at the circuit's outputs, with their [`SubgraphHyperEdge`] IDs.
    outputs: Vec<(Type, EncodedEdgeID)>,
    /// An additional unsupported subgraph that connects the input of the hugr
    /// directly to the output, without ever interacting with qubit/bit
    /// registers (and hence, not encoded as a pytket barrier on relevant
    /// registers).
    extra_subgraph: Option<OpaqueSubgraphPayload>,
}

/// Payload for a pytket barrier metadata that indicates the barrier represents
/// an opaque HUGR subgraph.
///
/// The payload may be encoded inline, embedding the HUGR subgraph as an
/// envelope in the operation's date, or be a reference to a subgraph tracked
/// inside a [`EncodedCircuit`][super::super::circuit::EncodedCircuit]
/// structure.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OpaqueSubgraphPayload {
    /// The type of payload.
    ///
    /// Either an inline hugr envelope or a reference to a subgraph tracked
    /// inside a [`OpaqueSubgraphs`][super::OpaqueSubgraphs] structure.
    #[serde(flatten)]
    pub(super) typ: OpaqueSubgraphPayloadType,
    /// Input types of the subgraph.
    ///
    /// Each input is assigned a unique edge identifier, so we can reconstruct
    /// the connections that are not encoded in the pytket circuit.
    ///
    /// The types can also be inferred from the encoded hugr or linked
    /// subcircuit, but we store them here to be robust.
    inputs: Vec<(Type, EncodedEdgeID)>,
    /// Output types of the subgraph.
    ///
    /// Each output is assigned a unique edge identifier, so we can reconstruct
    /// the connections that are not encoded in the pytket circuit.
    ///
    /// The types can also be inferred from the encoded hugr or linked
    /// subcircuit, but we store them here for robustness.
    outputs: Vec<(Type, EncodedEdgeID)>,
}

/// Payload for a pytket barrier metadata that indicates the barrier represents
/// an opaque HUGR subgraph.
///
/// The payload may be encoded inline, embedding the HUGR subgraph as an
/// envelope in the operation's date, or be a reference to a subgraph tracked
/// inside a [`EncodedCircuit`][super::super::circuit::EncodedCircuit]
/// structure.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "typ", content = "subgraph")]
pub enum OpaqueSubgraphPayloadType {
    /// An inline payload, carrying the encoded envelope for the HUGR subgraph.
    Inline {
        /// A string envelope containing the encoded HUGR subgraph.
        hugr_envelope: String,
    },
    /// A reference to a subgraph tracked by an `OpaqueSubgraphs` registry
    /// in an [`EncodedCircuit`][super::super::circuit::EncodedCircuit]
    /// structure.
    External {
        /// The ID of the subgraph in the `OpaqueSubgraphs` registry.
        id: SubgraphId,
    },
}

impl OpaqueSubgraphPayloadType {
    /// Create an inline payload by encoding the subgraph as an envelope.
    //
    // TODO: Detect and deal with non-local edges. Include global fn/const
    // definitions, and reject other non-local edges.
    //
    // TODO: This should include descendants of the subgraph. It doesn't.
    pub(super) fn inline<N: HugrNode>(
        subgraph: &SiblingSubgraph<N>,
        hugr: &impl HugrView<Node = N>,
    ) -> Self {
        let opaque_hugr = subgraph.extract_subgraph(hugr, "");
        let payload = Package::from_hugr(opaque_hugr)
            .store_str(EnvelopeConfig::text())
            .unwrap();
        Self::Inline {
            hugr_envelope: payload,
        }
    }
}

impl OpaqueSubgraphPayload {
    /// Create a new payload for an opaque subgraph in the Hugr.
    pub fn new<N: HugrNode>(
        subgraph: &SiblingSubgraph<N>,
        hugr: &impl HugrView<Node = N>,
        typ: OpaqueSubgraphPayloadType,
    ) -> Self {
        let signature = subgraph.signature(hugr);

        let mut inputs = Vec::with_capacity(subgraph.incoming_ports().iter().map(Vec::len).sum());
        for subgraph_inputs in subgraph.incoming_ports() {
            let Some((inp_node, inp_port0)) = subgraph_inputs.first() else {
                continue;
            };
            let input_wire = Wire::from_connected_port(*inp_node, *inp_port0, hugr);
            let edge_id = EncodedEdgeID::new(input_wire);
            inputs.extend(itertools::repeat_n(edge_id, subgraph_inputs.len()));
        }

        let outputs = subgraph
            .outgoing_ports()
            .iter()
            .map(|(n, p)| EncodedEdgeID::new(Wire::new(*n, *p)));

        Self {
            typ,
            inputs: signature.input().iter().cloned().zip(inputs).collect(),
            outputs: signature.output().iter().cloned().zip(outputs).collect(),
        }
    }

    /// Load a payload encoded in a json string.
    ///
    /// Updates weak extension references inside the definition after loading.
    pub fn load_str(json: &str, extensions: &ExtensionRegistry) -> Result<Self, PytketDecodeError> {
        let mut payload: Self = serde_json::from_str(json).map_err(|e| {
            PytketDecodeErrorInner::UnsupportedSubgraphPayload {
                source: EnvelopeError::SerdeError { source: e },
            }
            .wrap()
        })?;
        let extensions: WeakExtensionRegistry = extensions.into();

        // Resolve the cached input/output types.
        for (ty, _) in payload.inputs.iter_mut().chain(payload.outputs.iter_mut()) {
            resolve_type_extensions(ty, &extensions).map_err(|e| {
                let registry_load_e =
                    ExtensionRegistryLoadError::ExtensionResolutionError(Box::new(e));
                let envelope_e = EnvelopeError::ExtensionLoad {
                    source: registry_load_e,
                };
                PytketDecodeErrorInner::UnsupportedSubgraphPayload { source: envelope_e }.wrap()
            })?;
        }

        Ok(payload)
    }

    /// Returns the inputs types and internal edge IDs of the payload.
    pub fn inputs(&self) -> impl Iterator<Item = (&Type, EncodedEdgeID)> + '_ {
        self.inputs.iter().map(|(t, e)| (t, *e))
    }

    /// Returns the outputs types and internal edge IDs of the payload.
    pub fn outputs(&self) -> impl Iterator<Item = (&Type, EncodedEdgeID)> + '_ {
        self.outputs.iter().map(|(t, e)| (t, *e))
    }

    /// Returns the type of the payload.
    pub fn typ(&self) -> &OpaqueSubgraphPayloadType {
        &self.typ
    }

    /// Returns `true` if the payload is an inline payload.
    pub fn is_inline(&self) -> bool {
        matches!(self.typ, OpaqueSubgraphPayloadType::Inline { .. })
    }

    /// Returns `true` if the payload is an external payload.
    pub fn is_external(&self) -> bool {
        matches!(self.typ, OpaqueSubgraphPayloadType::External { .. })
    }
}
