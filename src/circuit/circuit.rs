// use daggy::petgraph::visit::{EdgeRef, IntoEdgesDirected};
// use daggy::petgraph::EdgeDirection;
// use daggy::NodeIndex;

use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

use crate::graph::graph::{NodePort, PortIndex};

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

#[derive(Clone)]
struct Boundary {
    pub input: Vertex,
    pub output: Vertex,
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
        Self::with_uids(vec![])
    }

    pub fn with_uids(uids: Vec<UnitID>) -> Self {
        let n_uids = uids.len();
        let mut dag = DAG::with_capacity(2, n_uids);
        let input = dag.add_node(VertexProperties::new(Op::Input));
        let output = dag.add_node(VertexProperties::new(Op::Output));
        let mut slf = Self {
            dag,
            name: None,
            phase: (0 as u64).into(),
            boundary: Boundary { input, output },
            uids: Vec::with_capacity(n_uids),
        };
        for uid in uids {
            slf.add_unitid(uid);
        }
        slf
    }

    pub fn insert(&mut self, new_vert: Vertex, edges: Vec<Edge>) -> Result<(), String> {
        // called rewire in TKET-1
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

        for (i, (edge, vert_sig_type)) in edges.into_iter().zip(vert_op_sig).enumerate() {
            let edgeprops = self
                .dag
                .edge_weight(edge)
                .ok_or("Edge not found.".to_string())?
                .clone();

            let (old_v1, old_v2) = self
                .dag
                .edge_endpoints(edge)
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
                    self.dag.remove_edge(edge);
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
        let (_, inlen) = self.dag.node_boundary_size(self.boundary.input);
        let (outlen, _) = self.dag.node_boundary_size(self.boundary.output);
        self.add_edge(
            (self.boundary.input, inlen as u8).into(),
            (self.boundary.output, outlen as u8).into(),
            uid.get_type(),
        );
        self.uids.push(uid);
        // .unwrap(); // should be cycle free so unwrap
    }
    pub fn add_edge(&mut self, source: NodePort, target: NodePort, edge_type: WireType) -> Edge {
        // let ports = (source.1, target.1);
        self.dag.add_edge(
            source,
            target,
            EdgeProperties {
                edge_type,
                // ports,
            },
        )
        // .map_err(|_| CycleInGraph())
    }

    pub fn add_vertex(&mut self, op: Op) -> Vertex {
        let siglen = op.signature().len();
        let weight = VertexProperties::new(op);
        self.dag.add_node_with_capacity(siglen, weight)
    }
    pub fn append_op(&mut self, op: Op, args: &Vec<PortIndex>) -> Result<Vertex, String> {
        // akin to add-op in TKET-1
        let sig = match op.signature() {
            Signature::Linear(sig) => sig,
            Signature::NonLinear(_, _) => return Err("Only linear ops supported.".to_string()),
        };
        assert!(sig.len() == args.len());

        let new_vert = self.add_vertex(op);
        let insertion_edges = args
            .iter()
            .map(|port| {
                self.dag
                    .in_edge_at_port(NodePort::new(self.boundary.output, *port))
            })
            .collect::<Option<Vec<Edge>>>()
            .ok_or("Invalid output ports".to_string())?;
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
        self.insert(new_vert, insertion_edges)?;
        Ok(new_vert)
    }

    pub fn to_commands(&self) -> CommandIter {
        CommandIter::new(self)
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

    pub fn unitids(&self) -> impl Iterator<Item = &UnitID> + '_ {
        self.uids.iter()
    }
}

pub struct Command {
    pub op: Op,
    pub args: Vec<UnitID>,
}

pub struct CommandIter<'circ> {
    nodes: TopSorter<'circ>,
    circ: &'circ Circuit,
    frontier: HashMap<Edge, &'circ UnitID>,
}

impl<'circ> CommandIter<'circ> {
    fn new(circ: &'circ Circuit) -> Self {
        Self {
            nodes: TopSorter::new(&circ.dag, [circ.boundary.input].into()),
            frontier: circ
                .dag
                .outgoing_edges(circ.boundary.input)
                .enumerate()
                .map(|(i, e)| (e.clone(), circ.uids.get(i).unwrap()))
                .collect(),
            circ,
        }
    }
}

impl<'circ> Iterator for CommandIter<'circ> {
    type Item = Command;

    fn next(&mut self) -> Option<Self::Item> {
        self.nodes.next().map(|node| {
            let VertexProperties { op } = self.circ.dag.node_weight(node).expect("Node not found");
            // assumes linarity
            let args = self
                .circ
                .dag
                .incoming_edges(node)
                .zip(self.circ.dag.outgoing_edges(node))
                .map(|(in_e, out_e)| {
                    let uid = self.frontier.remove(in_e).expect("edge not in frontier");
                    self.frontier.insert(*out_e, uid);
                    uid.clone()
                })
                .collect();

            Command {
                op: op.clone(),
                args,
            }
        })
    }
}
