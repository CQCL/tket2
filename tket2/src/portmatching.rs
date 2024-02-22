//! Pattern matching for circuits

pub mod matcher;
pub mod pattern;

use hugr::OutgoingPort;
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
        let (dst_node, dst) = circ
            .linked_ports(node, src)
            .exactly_one()
            .map_err(|mut e| {
                if e.next().is_some() {
                    InvalidEdgeProperty::AmbiguousEdge(src)
                } else {
                    InvalidEdgeProperty::NoLinkedEdge(src)
                }
            })?;
        if circ.get_optype(dst_node).tag() == OpTag::Input {
            return Ok(Self::InputEdge { src });
        }
        let port_type = circ
            .signature(node)
            .unwrap()
            .port_type(src)
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
///
/// A NodeID::CopyNode can only be found as a target of a PEdge::InputEdge
/// property. Furthermore, a NodeID::CopyNode never has a node property.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub(super) enum NodeID {
    HugrNode(Node),
    CopyNode(Node, Port),
}

impl NodeID {
    /// Create a new copy NodeID.
    pub fn new_copy(node: Node, port: impl Into<OutgoingPort>) -> Self {
        let port: OutgoingPort = port.into();
        Self::CopyNode(node, port.into())
    }
}

impl From<Node> for NodeID {
    fn from(node: Node) -> Self {
        Self::HugrNode(node)
    }
}

#[cfg(test)]
mod tests {
    use crate::Tk2Op;
    use hugr::{
        builder::{DFGBuilder, Dataflow, DataflowHugr},
        extension::{prelude::QB_T, PRELUDE_REGISTRY},
        types::FunctionType,
        Hugr,
    };
    use rstest::{fixture, rstest};

    use super::{CircuitPattern, PatternMatcher};

    #[fixture]
    fn lhs() -> Hugr {
        let mut h = DFGBuilder::new(FunctionType::new(vec![], vec![QB_T])).unwrap();

        let res = h.add_dataflow_op(Tk2Op::QAlloc, []).unwrap();
        let q = res.out_wire(0);

        h.finish_hugr_with_outputs([q], &PRELUDE_REGISTRY).unwrap()
    }

    #[fixture]
    fn rhs() -> Hugr {
        let mut h = DFGBuilder::new(FunctionType::new(vec![], vec![QB_T])).unwrap();

        let res = h.add_dataflow_op(Tk2Op::QAlloc, []).unwrap();
        let q = res.out_wire(0);
        let res = h.add_dataflow_op(Tk2Op::Reset, [q]).unwrap();
        let q = res.out_wire(0);

        h.finish_hugr_with_outputs([q], &PRELUDE_REGISTRY).unwrap()
    }

    #[fixture]
    pub fn circ() -> Hugr {
        let mut h = DFGBuilder::new(FunctionType::new(vec![QB_T], vec![QB_T])).unwrap();
        let mut inps = h.input_wires();
        let q_in = inps.next().unwrap();

        let res = h.add_dataflow_op(Tk2Op::QAlloc, []).unwrap();
        let q_out = res.out_wire(0);
        let res = h.add_dataflow_op(Tk2Op::CZ, [q_in, q_out]).unwrap();
        let q_in = res.out_wire(0);
        let q_out = res.out_wire(1);
        h.add_dataflow_op(Tk2Op::QFree, [q_in]).unwrap();

        h.finish_hugr_with_outputs([q_out], &PRELUDE_REGISTRY)
            .unwrap()
    }

    #[rstest]
    fn simple_match(circ: Hugr, lhs: Hugr) {
        let p = CircuitPattern::try_from_circuit(&lhs).unwrap();
        let m = PatternMatcher::from_patterns(vec![p]);

        let matches = m.find_matches(&circ);
        assert_eq!(matches.len(), 1);
    }
}
