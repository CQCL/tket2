#![allow(dead_code)]

use super::{circuit::UidIndex, operation::Op};

pub struct VertexProperties {
    pub op: Op,
    pub opgroup: Option<String>,
}

impl VertexProperties {
    pub fn new(op: Op) -> Self {
        Self { op, opgroup: None }
    }
}

#[derive(Clone)]
pub struct EdgeProperties {
    // pub edge_type: WireType,
    pub uid_ref: UidIndex,
    // pub ports: (Port, Port),
}

// pub(crate) type DAG = StableDag<VertexProperties, EdgeProperties>;
pub(crate) type DAG = crate::graph::graph::Graph<VertexProperties, EdgeProperties>;
pub(crate) type Vertex = crate::graph::graph::NodeIndex;
pub(crate) type Edge = crate::graph::graph::EdgeIndex;
