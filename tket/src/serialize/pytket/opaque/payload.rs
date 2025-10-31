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
/// These are used to identify edges in the [`OpaqueSubgraphPayload::Inline`]
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

/// Payload for a pytket barrier metadata that indicates the barrier represents
/// an opaque HUGR subgraph.
///
/// The payload may be encoded inline, embedding the HUGR subgraph as an
/// envelope in the operation's date, or be a reference to a subgraph tracked
/// inside a [`EncodedCircuit`][super::super::circuit::EncodedCircuit]
/// structure.
///
/// Inline payloads encode their input and output boundaries that cannot be
/// encoded as pytket qubit/bit registers using [`EncodedEdgeID`]s independent
/// from the hugr. A circuit may mix barriers with both inline and external
/// payloads, as long as there are no edges requiring a [`EncodedEdgeID`]
/// between them.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "typ")]
pub enum OpaqueSubgraphPayload {
    /// A reference to a subgraph tracked by an `OpaqueSubgraphs` registry
    /// in an [`EncodedCircuit`][super::super::circuit::EncodedCircuit]
    /// structure.
    External {
        /// The ID of the subgraph in the `OpaqueSubgraphs` registry.
        id: SubgraphId,
    },
    /// An inline payload, carrying the encoded envelope for the HUGR subgraph.
    Inline {
        /// A string envelope containing the encoded HUGR subgraph.
        hugr_envelope: String,
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
    },
}

impl OpaqueSubgraphPayload {
    /// Create an external payload by referencing a subgraph in the
    /// [`OpaqueSubgraphs`][super::OpaqueSubgraphs] registry.
    pub fn new_external(subgraph_id: SubgraphId) -> Self {
        Self::External { id: subgraph_id }
    }

    /// Create a new payload for an opaque subgraph in the Hugr.
    ///
    /// Encodes the subgraph into a hugr envelope.
    pub fn new_inline<N: HugrNode>(
        subgraph: &SiblingSubgraph<N>,
        hugr: &impl HugrView<Node = N>,
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

        // TODO: This should include descendants of the subgraph. It doesn't.
        let opaque_hugr = subgraph.extract_subgraph(hugr, "");
        let hugr_envelope = Package::from_hugr(opaque_hugr)
            .store_str(EnvelopeConfig::text())
            .unwrap();

        Self::Inline {
            hugr_envelope,
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

        // Resolve the extension ops and types in the inline payload.
        if let Self::Inline {
            inputs, outputs, ..
        } = &mut payload
        {
            let extensions: WeakExtensionRegistry = extensions.into();

            // Resolve the cached input/output types.
            for (ty, _) in inputs.iter_mut().chain(outputs.iter_mut()) {
                resolve_type_extensions(ty, &extensions).map_err(|e| {
                    let registry_load_e =
                        ExtensionRegistryLoadError::ExtensionResolutionError(Box::new(e));
                    let envelope_e = EnvelopeError::ExtensionLoad {
                        source: registry_load_e,
                    };
                    PytketDecodeErrorInner::UnsupportedSubgraphPayload { source: envelope_e }.wrap()
                })?;
            }
        }

        Ok(payload)
    }

    /// Returns `true` if the payload is an inline payload.
    pub fn is_inline(&self) -> bool {
        matches!(self, Self::Inline { .. })
    }

    /// Returns `true` if the payload is an external payload.
    pub fn is_external(&self) -> bool {
        matches!(self, Self::External { .. })
    }
}
