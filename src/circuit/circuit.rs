// use daggy::petgraph::visit::{EdgeRef, IntoEdgesDirected};
// use daggy::petgraph::EdgeDirection;
// use daggy::NodeIndex;

use std::fmt::Debug;
use std::hash::Hash;

use crate::graph::graph::{IndexType, NodePort};

use super::dag::{Edge, EdgeProperties, TopSorter, Vertex, VertexProperties, DAG};
use super::operation::{Op, Param, Signature, WireType};

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

// address of internal memory model
#[derive(Copy, Clone, Default, PartialEq, PartialOrd, Eq, Ord, Hash, Debug)]
pub struct UidIndex(u32);

impl IndexType for UidIndex {
    fn index(&self) -> usize {
        self.0 as usize
    }
    fn new(x: usize) -> Self {
        Self(x as u32)
    }
    fn max() -> Self {
        Self(u32::MAX)
    }
}

#[derive(Clone)]
struct Boundary {
    pub inputs: Vec<Vertex>,
    pub outputs: Vec<Vertex>,
}

pub struct CycleInGraph();
impl Debug for CycleInGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("CycleInGraph: Cycle detected or created in graph. Not a DAG.")
            .finish()
    }
}

impl From<CycleInGraph> for String {
    fn from(c: CycleInGraph) -> Self {
        format!("{c:?}")
    }
}

#[derive(Clone)]
pub struct Circuit {
    dag: DAG,
    pub name: Option<String>,
    pub phase: Param,
    boundary: Boundary,
    uids: Vec<UnitID>,
}

impl Circuit {
    pub fn new() -> Self {
        Self {
            dag: DAG::new(),
            name: None,
            phase: "0".into(),
            boundary: Boundary {
                inputs: vec![],
                outputs: vec![],
            },
            uids: vec![],
        }
    }

    pub fn with_uids(uids: Vec<UnitID>) -> Self {
        let n_uids = uids.len();
        let mut slf = Self {
            dag: DAG::with_capacity(n_uids * 2, n_uids),
            name: None,
            phase: "0".into(),
            boundary: Boundary {
                inputs: Vec::with_capacity(n_uids),
                outputs: Vec::with_capacity(n_uids),
            },
            uids: vec![],
        };

        for uid in uids {
            slf.add_unitid(uid);
        }
        slf
    }
    pub fn get_out(&self, uid: &UnitID) -> Result<Vertex, String> {
        let uix = self
            .uids
            .iter()
            .position(|u| u == uid)
            .ok_or("UnitID not found in boundary.".to_string())?;
        self.boundary
            .outputs
            .iter()
            .find(|&&out_v| {
                self.dag
                    .edge_weight(*self.dag.incoming_edges(out_v).next().unwrap())
                    .unwrap()
                    .uid_ref
                    .index()
                    == uix
            })
            .map(|n| *n)
            .ok_or("No output node has incoming edges from UnitID.".to_string())
    }

    pub fn rewire(&mut self, new_vert: Vertex, preds: Vec<Edge>) -> Result<(), String> {
        // let mut bin: Vec<Edge> = vec![];
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
                &self.uids[edgeprops.uid_ref.index()].get_type(),
            ) {
                // (WireType::Bool, WireType::Classical) => {

                //     self.dag
                //         .add_edge(
                //             old_v1,
                //             new_vert,
                //             EdgeProperties {
                //                 edge_type: WireType::Bool,
                //                 ..edgeprops
                //             },
                //         )
                //         .map_err(|_| CycleInGraph())?;
                // }
                // (WireType::Bool, _) => {
                //     return Err(
                //         "Cannot rewire; Boolean needs a classical value to read from.".to_string(),
                //     )
                // }
                (x, y) if x == y => {
                    self.dag.remove_edge(pred);
                    self.dag
                        .add_edge(old_v1, (new_vert, i as u8).into(), edgeprops.clone());
                    // .map_err(|_| CycleInGraph())?;

                    self.dag
                        .add_edge((new_vert, i as u8).into(), old_v2, edgeprops);
                    // .map_err(|_| CycleInGraph())?;
                    // bin.push(pred);
                }
                _ => return Err("Cannot rewire; Changing type of edge.".to_string()),
            }
        }
        // for e in bin {
        //     self.dag.remove_edge(e);
        // }
        Ok(())
    }
    pub fn add_unitid(&mut self, uid: UnitID) {
        let inv = self
            .dag
            .add_node_with_capacity(1, VertexProperties::new(Op::Input));
        let outv = self
            .dag
            .add_node_with_capacity(1, VertexProperties::new(Op::Output));

        self.boundary.inputs.push(inv);
        self.boundary.outputs.push(outv);
        self.uids.push(uid);
        self.add_edge(
            (inv, 0).into(),
            (outv, 0).into(),
            UidIndex::new(self.uids.len() - 1),
        );
        // .unwrap(); // should be cycle free so unwrap
    }
    pub fn add_edge(&mut self, source: NodePort, target: NodePort, uid_ref: UidIndex) -> Edge {
        // let ports = (source.1, target.1);
        self.dag.add_edge(
            source,
            target,
            EdgeProperties {
                uid_ref,
                // ports,
            },
        )
        // .map_err(|_| CycleInGraph())
    }

    pub fn add_vertex(&mut self, op: Op, _opgroup: Option<String>) -> Vertex {
        let siglen = op.signature().len();
        let weight = VertexProperties::new(op);
        self.dag.add_node_with_capacity(siglen, weight)
    }
    pub fn add_op(
        &mut self,
        op: Op,
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
                Ok(*self
                    .dag
                    .incoming_edges(self.get_out(uid)?)
                    .next()
                    .ok_or("No incoming edges".to_string())?)
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

    pub fn to_commands(&self) -> CommandIter {
        CommandIter::new(self)
        // let topo_nodes =
        //     daggy::petgraph::algo::toposort(&self.dag, None).map_err(|_| CycleInGraph())?;
        // // Ok(CommandIter{nodesNodeIndex))
        // todo!()
    }

    pub fn qubits(&self) -> impl Iterator<Item = UnitID> + '_ {
        self.uids.iter().filter_map(|uid| match uid {
            UnitID::Qubit { .. } => Some(uid.clone()),
            UnitID::Bit { .. } => None,
        })
    }

    pub fn bits(&self) -> impl Iterator<Item = UnitID> + '_ {
        self.uids.iter().filter_map(|uid| match uid {
            UnitID::Bit { .. } => Some(uid.clone()),
            UnitID::Qubit { .. } => None,
        })
    }
}

pub struct Command {
    pub op: Op,
    pub args: Vec<UnitID>,
    pub opgroup: Option<String>,
}

pub struct CommandIter<'circ> {
    nodes: TopSorter<'circ>,
    circ: &'circ Circuit,
}

impl<'circ> CommandIter<'circ> {
    fn new(circ: &'circ Circuit) -> Self {
        Self {
            nodes: TopSorter::new(&circ.dag, circ.boundary.inputs.iter().cloned().collect()),
            circ,
        }
    }
}

impl<'circ> Iterator for CommandIter<'circ> {
    type Item = Command;

    fn next(&mut self) -> Option<Self::Item> {
        self.nodes.next().map(|node| {
            let VertexProperties { op, opgroup } =
                self.circ.dag.node_weight(node).expect("Node not found");

            let args = self
                .circ
                .dag
                .incoming_edges(node)
                .map(|e| {
                    self.circ.uids[self.circ.dag.edge_weight(*e).unwrap().uid_ref.index()].clone()
                })
                .collect();
            Command {
                op: op.clone(),
                args,
                opgroup: opgroup.clone(),
            }
        })
    }
}
