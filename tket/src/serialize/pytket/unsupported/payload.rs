//! Definitions of the payloads for opaque barrier metadata in pytket circuits.

use hugr::core::HugrNode;
use hugr::envelope::EnvelopeConfig;
use hugr::hugr::views::SiblingSubgraph;
use hugr::package::Package;
use hugr::types::Type;
use hugr::{HugrView, Wire};

use super::SubgraphId;

/// Pytket opgroup used to identify opaque barrier operations that encode standalone unsupported HUGR subgraphs.
///
/// See [`UnsupportedSubgraphPayloadType::Standalone`].
pub const OPGROUP_STANDALONE_UNSUPPORTED_HUGR: &str = "UNSUPPORTED_HUGR";

/// Pytket opgroup used to identify opaque barrier operations that encode external unsupported HUGR subgraphs.
///
/// See [`UnsupportedSubgraphPayloadType::External`].
pub const OPGROUP_EXTERNAL_UNSUPPORTED_HUGR: &str = "EXTERNAL_UNSUPPORTED_HUGR";

/// Identifier for a hyper edge in the Hugr, encoded as a 64-bit hash that is
/// independent of the original Hugr representation.
///
/// These are used to identify edges in the [`UnsupportedSubgraphPayload`]
/// payloads encoded in opaque barriers on the encoded pytket circuits.
///
/// We require them to reconstruct the edges of the hugr that are not reflected
/// as pytket register/parameter dependencies. This is the case for edges with
/// unsupported types between unsupported subgraphs, or between an unsupported
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
/// an unsupported HUGR subgraph.
///
/// The payload may be standalone, carrying the encoded HUGR subgraph, or be a
/// reference to a subgraph tracked inside a
/// [`EncodedCircuit`][super::super::circuit::EncodedCircuit] structure.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct UnsupportedSubgraphPayload {
    /// The type of payload.
    ///
    /// Either a standalone hugr envelope or a reference to a subgraph tracked
    /// inside a [`UnsupportedSubgraphs`][super::UnsupportedSubgraphs] structure.
    pub(super) typ: UnsupportedSubgraphPayloadType,
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
    /// subcircuit, but we store them here to be robust.
    outputs: Vec<(Type, EncodedEdgeID)>,
}

/// Payload for a pytket barrier metadata that indicates the barrier represents
/// an unsupported HUGR subgraph.
///
/// The payload may be standalone, carrying the encoded HUGR subgraph, or be a
/// reference to a subgraph tracked inside a
/// [`EncodedCircuit`][super::super::circuit::EncodedCircuit] structure.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum UnsupportedSubgraphPayloadType {
    /// A standalone payload, carrying the encoded HUGR subgraph.
    Standalone {
        /// A string envelope containing the encoded HUGR subgraph.
        hugr_envelope: String,
    },
    /// A reference to a subgraph tracked by an `UnsupportedSubgraphs` registry
    /// in an [`EncodedCircuit`][super::super::circuit::EncodedCircuit]
    /// structure.
    External {
        /// The ID of the subgraph in the `UnsupportedSubgraphs` registry.
        id: SubgraphId,
    },
}

impl UnsupportedSubgraphPayloadType {
    /// Create a standalone payload by encoding a subgraph.
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

impl UnsupportedSubgraphPayload {
    /// Create a new payload for an unsupported subgraph in the Hugr.
    pub fn new<N: HugrNode>(
        subgraph: &SiblingSubgraph<N>,
        hugr: &impl HugrView<Node = N>,
        typ: UnsupportedSubgraphPayloadType,
    ) -> Self {
        let signature = subgraph.signature(hugr);

        let mut inputs = Vec::new();
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
        let payload = UnsupportedSubgraphPayloadType::standalone(subgraph, hugr);
        Self::new(subgraph, hugr, payload)
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
    pub fn typ(&self) -> &UnsupportedSubgraphPayloadType {
        &self.typ
    }

    /// Returns `true` if the payload is a standalone payload.
    pub fn is_standalone(&self) -> bool {
        matches!(self.typ, UnsupportedSubgraphPayloadType::Standalone { .. })
    }

    /// Returns `true` if the payload is an external payload.
    pub fn is_external(&self) -> bool {
        matches!(self.typ, UnsupportedSubgraphPayloadType::External { .. })
    }
}
