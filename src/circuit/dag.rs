#![allow(dead_code)]

use super::{circuit::UidIndex, operation::Op};

#[derive(Clone)]
pub struct VertexProperties {
    pub op: Op,
}

impl VertexProperties {
    pub fn new(op: Op) -> Self {
        Self { op }
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
pub(crate) type TopSorter<'a> =
    crate::graph::toposort::TopSortWalker<'a, VertexProperties, EdgeProperties>;
pub(crate) type Vertex = crate::graph::graph::NodeIndex;
pub(crate) type Edge = crate::graph::graph::EdgeIndex;
