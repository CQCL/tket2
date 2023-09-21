//! Pattern matching for circuits

pub mod matcher;
pub mod pattern;
#[cfg(feature = "pyo3")]
pub mod pyo3;

use itertools::Itertools;
pub use matcher::{PatternMatch, PatternMatcher};
pub use pattern::CircuitPattern;

use hugr::{
    ops::{OpTag, OpTrait},
    Node, Port,
};
use matcher::MatchOp;
use thiserror::Error;

use crate::{circuit::Circuit, utils::type_is_linear};

type PNode = MatchOp;

/// An edge property in a circuit pattern.
///
/// Edges are
/// Edges are reversible if the edge type is linear.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
enum PEdge {
    /// A "normal" edge between src and dst within a pattern.
    InternalEdge {
        src: Port,
        dst: Port,
        is_reversible: bool,
    },
    /// An edge from a copied input to src.
    ///
    /// Edges from inputs are typically not matched as part of the pattern,
    /// unless a single input is copied multiple times. In this case, an
    /// InputEdge is used to link the source port to the (usually hidden)
    /// copy node.
    ///
    /// Input edges are always irreversible.
    InputEdge { src: Port },
}

#[derive(Debug, Clone, Error)]
enum InvalidEdgeProperty {
    /// The port is linked to multiple edges.
    #[error("port {0:?} is linked to multiple edges")]
    AmbiguousEdge(Port),
    /// The port is not linked to any edge.
    #[error("port {0:?} is not linked to any edge")]
    NoLinkedEdge(Port),
    /// The port does not have a type.
    #[error("port {0:?} does not have a type")]
    UntypedPort(Port),
}

impl PEdge {
    fn try_from_port(
        node: Node,
        port: Port,
        circ: &impl Circuit,
    ) -> Result<Self, InvalidEdgeProperty> {
        let src = port;
        let (dst_node, dst) = circ.linked_ports(node, src).exactly_one().map_err(|e| {
            if e.size_hint().0 > 0 {
                InvalidEdgeProperty::AmbiguousEdge(src)
            } else {
                InvalidEdgeProperty::NoLinkedEdge(src)
            }
        })?;
        if circ.get_optype(dst_node).tag() == OpTag::Input {
            return Ok(Self::InputEdge { src });
        }
        let port_type = circ
            .get_optype(node)
            .signature()
            .get(src)
            .cloned()
            .ok_or(InvalidEdgeProperty::UntypedPort(src))?;
        let is_reversible = type_is_linear(&port_type);
        Ok(Self::InternalEdge {
            src,
            dst,
            is_reversible,
        })
    }
}

impl portmatching::EdgeProperty for PEdge {
    type OffsetID = Port;

    fn reverse(&self) -> Option<Self> {
        match *self {
            Self::InternalEdge {
                src,
                dst,
                is_reversible,
            } => is_reversible.then_some(Self::InternalEdge {
                src: dst,
                dst: src,
                is_reversible,
            }),
            Self::InputEdge { .. } => None,
        }
    }

    fn offset_id(&self) -> Self::OffsetID {
        match *self {
            Self::InternalEdge { src, .. } => src,
            Self::InputEdge { src, .. } => src,
        }
    }
}

/// A node in a pattern.
///
/// A node is either a real node in the HUGR graph or a hidden copy node
/// that is identified by its node and outgoing port.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub(super) enum NodeID {
    HugrNode(Node),
    CopyNode(Node, Port),
}

impl From<Node> for NodeID {
    fn from(node: Node) -> Self {
        Self::HugrNode(node)
    }
}
