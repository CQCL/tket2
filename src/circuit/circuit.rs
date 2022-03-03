// use daggy::petgraph::visit::{EdgeRef, IntoEdgesDirected};
// use daggy::petgraph::EdgeDirection;
// use daggy::NodeIndex;

use std::fmt::Debug;
use std::hash::Hash;
use std::rc::Rc;

use crate::graph::graph::NodePort;

use super::dag::{Edge, EdgeProperties, Vertex, VertexProperties, DAG};
use super::operation::{GateOp, OpPtr, Param, Signature, WireType};

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
pub type UIDRef = usize;

#[derive(PartialEq, Eq, Hash, Debug)]
struct BoundaryElement {
    uid: UnitID,
    inv: Vertex,
    outv: Vertex,
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
pub struct Circuit {
    dag: DAG,
    pub name: Option<String>,
    pub phase: Param,
    boundary: Vec<BoundaryElement>,
}

impl Circuit {
    pub fn new() -> Self {
        Self {
            dag: DAG::new(),
            name: None,
            phase: "0".into(),
            boundary: vec![],
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
            match (&vert_sig_type, &edgeprops.edge_type) {
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
            .add_node_with_capacity(1, VertexProperties::new(Rc::new(GateOp::Input)));
        let outv = self
            .dag
            .add_node_with_capacity(1, VertexProperties::new(Rc::new(GateOp::Output)));

        let edge_type = uid.get_type();
        self.boundary.push(BoundaryElement { uid, inv, outv });
        self.add_edge(
            (inv, 0).into(),
            (outv, 0).into(),
            edge_type,
            self.boundary.len() - 1,
        );
        // .unwrap(); // should be cycle free so unwrap
    }
    pub fn add_edge(
        &mut self,
        source: NodePort,
        target: NodePort,
        edge_type: WireType,
        uid_ref: UIDRef,
    ) -> Edge {
        // let ports = (source.1, target.1);
        self.dag.add_edge(
            source,
            target,
            EdgeProperties {
                edge_type,
                uid_ref,
                // ports,
            },
        )
        // .map_err(|_| CycleInGraph())
    }

    pub fn add_vertex(&mut self, op: OpPtr, _opgroup: Option<String>) -> Vertex {
        let siglen = op.signature().len();
        let weight = VertexProperties::new(op);
        self.dag.add_node_with_capacity(siglen, weight)
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
        self.boundary.iter().filter_map(|bel| match bel.uid {
            UnitID::Qubit { .. } => Some(bel.uid.clone()),
            UnitID::Bit { .. } => None,
        })
    }

    pub fn bits(&self) -> impl Iterator<Item = UnitID> + '_ {
        self.boundary.iter().filter_map(|bel| match bel.uid {
            UnitID::Bit { .. } => Some(bel.uid.clone()),
            UnitID::Qubit { .. } => None,
        })
    }
}

pub struct Command {
    pub op: OpPtr,
    pub args: Vec<UnitID>,
    pub opgroup: Option<String>,
}

pub struct CommandIter<'circ> {
    nodes: Vec<Vertex>,
    current_node: usize,
    circ: &'circ Circuit,
}

impl<'circ> CommandIter<'circ> {
    // fn toposort(circ: &'a Circuit) -> Vec<NodeIndex> {
    //     let dag = circ.dag.clone();
    //     let mut sorted = vec![];

    //     let mut slice: BTreeSet<NodeIndex> =
    //         BTreeSet::from_iter(circ.boundary.iter().map(|BoundaryElement { inv, .. }| {
    //             circ.dag
    //                 .edge_endpoints(
    //                     circ.dag
    //                         .edges_directed(*inv, EdgeDirection::Incoming)
    //                         .next()
    //                         .unwrap()
    //                         .id(),
    //                 )
    //                 .unwrap()
    //                 .1
    //         }));

    //     let mut bin = BTreeSet::new();
    //     while let Some(n) = slice.iter().next() {
    //         bin.insert(n);

    //         sorted.push(*n);

    //     }
    //     sorted
    // }
    fn new(circ: &'circ Circuit) -> Self {
        Self {
            // nodes: daggy::petgraph::algo::toposort(&circ.dag, None)
            //     .map_err(|_| CycleInGraph())
            //     .unwrap(),
            nodes: circ.dag.nodes().collect(),
            current_node: 0,
            circ,
        }
    }
}

impl<'circ> Iterator for CommandIter<'circ> {
    type Item = Command;

    fn next(&mut self) -> Option<Self::Item> {
        self.current_node += 1;
        if self.current_node == self.nodes.len() {
            None
        } else {
            let node = self.nodes[self.current_node];
            let VertexProperties { op, opgroup } =
                self.circ.dag.node_weight(node).expect("Node not found");
            // let mut port_args: Vec<_> = self
            //     .circ
            //     .dag
            //     .edges_directed(node, EdgeDirection::Incoming)
            //     .map(|e| {
            //         (
            //             e.weight().ports.1,
            //             self.circ.boundary[e.weight().uid_ref].uid.clone(),
            //         )
            //     })
            //     .collect();
            // port_args.sort_unstable_by_key(|p| p.0);
            // let args = port_args.into_iter().map(|(_, uid)| uid).collect();

            let args = self
                .circ
                .dag
                .incoming_edges(node)
                .map(|e| {
                    self.circ.boundary[self.circ.dag.edge_weight(*e).unwrap().uid_ref]
                        .uid
                        .clone()
                })
                .collect();
            Some(Command {
                op: op.clone(),
                args,
                opgroup: opgroup.clone(),
            })
        }
    }
}
