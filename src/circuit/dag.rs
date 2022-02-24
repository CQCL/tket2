#![allow(dead_code)]

use super::operation::{Op, OpPtr, WireType};
use petgraph::stable_graph::{EdgeIndex, NodeIndex, StableDiGraph};

pub type Port = u16;

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
    pub ports: (Port, Port),
}

pub(crate) type DAG = StableDiGraph<VertexProperties, EdgeProperties>;
pub(crate) type Vertex = NodeIndex;
pub(crate) type Edge = EdgeIndex;
