#![allow(dead_code)]

use super::{
    circuit::UIDRef,
    operation::{OpPtr, WireType},
};
// use daggy::

pub struct VertexProperties {
    pub op: OpPtr,
    pub opgroup: Option<String>,
}

impl VertexProperties {
    pub fn new(op: OpPtr) -> Self {
        Self { op, opgroup: None }
    }
}

#[derive(Clone)]
pub struct EdgeProperties {
    pub edge_type: WireType,
    pub uid_ref: UIDRef,
    // pub ports: (Port, Port),
}

// pub(crate) type DAG = StableDag<VertexProperties, EdgeProperties>;
pub(crate) type DAG = crate::graph::graph::Graph<VertexProperties, EdgeProperties>;
pub(crate) type Vertex = crate::graph::graph::NodeIndex;
pub(crate) type Edge = crate::graph::graph::EdgeIndex;
