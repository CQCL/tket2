use std::fmt::Display;

use super::operation::{Op, WireType};

#[derive(Clone, Default, Debug, PartialEq)]
pub struct VertexProperties {
    pub op: Op,
}

impl Display for VertexProperties {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.op)
    }
}

impl VertexProperties {
    pub fn new(op: Op) -> Self {
        Self { op }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EdgeProperties {
    pub edge_type: WireType,
}

impl Display for EdgeProperties {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.edge_type)
    }
}

// pub(crate) type DAG = StableDag<VertexProperties, EdgeProperties>;
pub(crate) type Dag = portgraph::graph::Graph<VertexProperties, EdgeProperties>;
pub(crate) type TopSorter<'a> =
    portgraph::toposort::TopSortWalker<'a, VertexProperties, EdgeProperties>;
pub(crate) type Vertex = portgraph::graph::NodeIndex;
pub(crate) type Edge = portgraph::graph::EdgeIndex;
