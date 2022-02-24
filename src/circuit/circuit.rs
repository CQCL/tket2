use petgraph::visit::EdgeRef;
use petgraph::EdgeDirection;

use crate::circuit_json::Operation;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

use super::dag::{Edge, EdgeProperties, Port, Vertex, VertexProperties, DAG};
use super::operation::{GateOp, Op, OpPtr, Param, Signature, WireType};

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum UnitID {
    Qubit { name: String, index: Vec<u32> },
    Bit { name: String, index: Vec<u32> },
}

impl UnitID {
    pub fn get_type(&self) -> WireType {
        match self {
            Self::Qubit { .. } => WireType::Quantum,
            Self::Bit { .. } => WireType::Classical,
        }
    }
}

#[derive(PartialEq, Eq, Hash, Debug)]
struct BoundaryElement {
    uid: UnitID,
    inv: Vertex,
    outv: Vertex,
}
pub struct Circuit {
    dag: DAG,
    pub name: Option<String>,
    pub phase: Param,
    boundary: HashSet<BoundaryElement>,
}

impl Circuit {
    pub fn new() -> Self {
        Self {
            dag: DAG::new(),
            name: None,
            phase: "0".into(),
            boundary: HashSet::new(),
        }
    }
    pub fn get_out(&self, uid: &UnitID) -> Result<Vertex, String> {
        self.boundary
            .iter()
            .find(|boundel| boundel.uid == *uid)
            .ok_or("UnitID not found in boundary.".to_string())
            .map(|b| b.outv)
    }

    pub fn rewire(&mut self, new_vert: Vertex, preds: Vec<Edge>) -> Result<(), String> {
        let mut bin: Vec<Edge> = vec![];
        let vert_op_sig = match self
            .dag
            .node_weight(new_vert)
            .ok_or("Vertex not found.".to_string())?
            .op
            .signature()
        {
            Signature::Linear(sig) => sig,
            Signature::NonLinear(..) => {
                return Err("Nonlinear sigs not supported by rewire.".into())
            }
        };

        for (i, (pred, vert_sig_type)) in preds.into_iter().zip(vert_op_sig).enumerate() {
            let edgeprops = self
                .dag
                .edge_weight(pred)
                .ok_or("Edge not found.".to_string())?
                .clone();

            let (old_v1, old_v2) = self
                .dag
                .edge_endpoints(pred)
                .ok_or("Edge not found.".to_string())?;
            match (
                &vert_sig_type,
                &self.dag.edge_weight(pred).unwrap().edge_type,
            ) {
                (WireType::Bool, WireType::Classical) => {
                    self.dag.add_edge(
                        old_v1,
                        new_vert,
                        EdgeProperties {
                            edge_type: WireType::Bool,
                            ports: edgeprops.ports,
                        },
                    );
                }
                (WireType::Bool, _) => {
                    return Err(
                        "Cannot rewire; Boolean needs a classical value to read from.".to_string(),
                    )
                }
                (x, y) if x == y => {
                    self.dag.add_edge(
                        old_v1,
                        new_vert,
                        EdgeProperties {
                            edge_type: x.clone(),
                            ports: (edgeprops.ports.0, i as Port),
                        },
                    );
                    self.dag.add_edge(
                        new_vert,
                        old_v2,
                        EdgeProperties {
                            edge_type: x.clone(),
                            ports: (i as Port, edgeprops.ports.1),
                        },
                    );
                    bin.push(pred);
                }
                _ => return Err("Cannot rewire; Changing type of edge.".to_string()),
            }
        }
        for e in bin {
            self.dag.remove_edge(e);
        }
        Ok(())
    }
    pub fn add_unitid(&mut self, uid: UnitID) {
        let inv = self
            .dag
            .add_node(VertexProperties::new(Box::new(GateOp::Input)));
        let outv = self
            .dag
            .add_node(VertexProperties::new(Box::new(GateOp::Output)));

        self.add_edge((inv, 0), (outv, 0), uid.get_type());
        self.boundary.insert(BoundaryElement { uid, inv, outv });
    }
    pub fn add_edge(
        &mut self,
        source: (Vertex, Port),
        target: (Vertex, Port),
        edge_type: WireType,
    ) -> Edge {
        let ports = (source.1, target.1);
        self.dag
            .add_edge(source.0, target.0, EdgeProperties { edge_type, ports })
    }

    pub fn add_vertex(&mut self, op: OpPtr, _opgroup: Option<String>) -> Vertex {
        let weight = VertexProperties::new(op);
        self.dag.add_node(weight)
    }
    pub fn add_op(
        &mut self,
        op: OpPtr,
        args: &Vec<UnitID>,
        opgroup: Option<String>,
    ) -> Result<Vertex, String> {
        let sig = match op.signature() {
            Signature::Linear(sig) => sig,
            Signature::NonLinear(_, _) => return Err("Only linear ops supported.".to_string()),
        };
        assert!(sig.len() == args.len());

        let new_vert = self.add_vertex(op, opgroup);
        let preds: Result<Vec<Edge>, String> = args
            .iter()
            .map(|uid| -> Result<Edge, String> {
                Ok(self
                    .dag
                    .edges_directed(self.get_out(uid)?, EdgeDirection::Incoming)
                    .next()
                    .ok_or("No outgoing edges".to_string())?
                    .id())
            })
            .collect();
        let preds = preds?;
        // let mut wire_arg_set = HashSet::new();
        // for (arg, sig) in args.iter().zip(sig) {
        //     if sig != WireType::Bool {
        //         if wire_arg_set.contains(arg) {
        //             return Err(format!("Multiple operation arguments reference {arg:?}"));
        //         }
        //         wire_arg_set.insert(arg);
        //     }

        //     let out_v = self.get_out(arg)?;
        //     let pred_out_e = self.dag.edges_directed(a, dir)
        // }
        self.rewire(new_vert, preds)?;
        Ok(new_vert)
    }
}
