#![allow(dead_code)]

use super::{
    circuit::UIDRef,
    operation::{OpPtr, WireType},
};
use daggy::stable_dag::{EdgeIndex, NodeIndex, StableDag};
// use daggy:: 
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
    pub uid_ref: UIDRef,
    pub ports: (Port, Port),
}

pub(crate) type DAG = StableDag<VertexProperties, EdgeProperties>;
pub(crate) type Vertex = NodeIndex;
pub(crate) type Edge = EdgeIndex;
