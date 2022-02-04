#![allow(dead_code)]

use super::operation::{Op, WireType};
use petgraph::stable_graph::{EdgeIndex, NodeIndex, StableDiGraph};

pub type Port = u16;

#[derive(Clone)]
pub struct VertexProperties<'a> {
    pub op: &'a dyn Op,
    pub opgroup: Option<String>,
}

impl<'a> VertexProperties<'a> {
    pub fn new<T: Op + 'a>(op: &'a T) -> Self {
        Self { op, opgroup: None }
    }
}

#[derive(Clone)]
pub struct EdgeProperties {
    pub edge_type: WireType,
    pub ports: (Port, Port),
}

pub(crate) type DAG<'a> = StableDiGraph<VertexProperties<'a>, EdgeProperties>;
pub(crate) type Vertex = NodeIndex;
pub(crate) type Edge = EdgeIndex;
