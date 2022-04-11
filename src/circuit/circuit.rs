use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

use crate::graph::graph::{Direction, NodePort, PortIndex};

use super::dag::{Edge, EdgeProperties, TopSorter, Vertex, VertexProperties, DAG};
use super::operation::{Op, Param, WireType};

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum UnitID {
    Qubit { name: String, index: Vec<u32> },
    Bit { name: String, index: Vec<u32> },
    F64(String),
}

impl UnitID {
    pub fn get_type(&self) -> WireType {
        match self {
            Self::Qubit { .. } => WireType::Qubit,
            Self::Bit { .. } => WireType::LinearBit,
            Self::F64(_) => WireType::F64,
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
    pub(crate) dag: DAG,
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
            slf.add_linear_unitid(uid);
        }
        slf
    }

    pub fn rewire(&mut self, new_vert: Vertex, edges: Vec<Edge>) -> Result<(), String> {
        // called rewire in TKET-1
        let vert_op_sig = match self
            .dag
            .node_weight(new_vert)
            .ok_or("Vertex not found.".to_string())?
            .op
            .signature()
        {
            x if x.purely_linear() => x.linear,
            _ => return Err("Nonlinear sigs not supported by rewire.".into()),
        };

        for (i, (edge, vert_sig_type)) in edges.into_iter().zip(vert_op_sig).enumerate() {
            let edgeprops = self
                .dag
                .edge_weight(edge)
                .ok_or("Edge not found.".to_string())?
                .clone();

            let [old_v1, old_v2] = self
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

    pub fn add_linear_unitid(&mut self, uid: UnitID) {
        let (_, inlen) = self.dag.node_boundary_size(self.boundary.input);
        let (outlen, _) = self.dag.node_boundary_size(self.boundary.output);
        self.add_edge(
            (self.boundary.input, inlen as u8),
            (self.boundary.output, outlen as u8),
            uid.get_type(),
        );
        self.uids.push(uid);
    }

    pub fn add_unitid(&mut self, uid: UnitID) {
        self.uids.push(uid);
    }
    pub fn add_edge<T: Into<NodePort>>(
        &mut self,
        source: T,
        target: T,
        edge_type: WireType,
    ) -> Edge {
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
            x if x.purely_linear() => x.linear,
            _ => return Err("Only linear ops supported.".to_string()),
        };
        assert_eq!(sig.len(), args.len());

        let new_vert = self.add_vertex(op);
        let insertion_edges = args
            .iter()
            .map(|port| {
                self.dag.edge_at_port(
                    NodePort::new(self.boundary.output, *port),
                    Direction::Incoming,
                )
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
        self.rewire(new_vert, insertion_edges)?;
        Ok(new_vert)
    }

    pub fn to_commands(&self) -> CommandIter {
        CommandIter::new(self)
    }

    pub fn qubits(&self) -> impl Iterator<Item = UnitID> + '_ {
        self.uids.iter().filter_map(|uid| match uid {
            UnitID::Qubit { .. } => Some(uid.clone()),
            UnitID::Bit { .. } | UnitID::F64(_) => None,
        })
    }

    pub fn bits(&self) -> impl Iterator<Item = UnitID> + '_ {
        self.uids.iter().filter_map(|uid| match uid {
            UnitID::Bit { .. } => Some(uid.clone()),
            UnitID::Qubit { .. } | UnitID::F64(_) => None,
        })
    }

    pub fn unitids(&self) -> impl Iterator<Item = &UnitID> + '_ {
        self.uids.iter()
    }

    pub fn boundary(&self) -> [Vertex; 2] {
        [self.boundary.input, self.boundary.output]
    }

    /// send an edge in to a copy vertex and return a reference to that vertex
    /// the existing target of the edge will be the only target of the copy node
    /// up to the user to make sure the remaining N-1 edges are connected to something
    pub fn copy_edge(&mut self, e: Edge, copies: u32) -> Result<Vertex, String> {
        let edge_type = match self.dag.edge_weight(e) {
            Some(EdgeProperties { edge_type, .. }) => edge_type,
            _ => return Err("Edge not found".into()),
        };

        let copy_op = match edge_type {
            WireType::Qubit | WireType::LinearBit => {
                return Err("Cannot copy qubit or LinearBit wires.".into())
            }
            _ => Op::Copy {
                n_copies: copies,
                typ: edge_type.clone(),
            },
        };
        let copy_node = self.add_vertex(copy_op);
        let [s, t] = self.dag.edge_endpoints(e).unwrap();
        let edge_type = self.dag.remove_edge(e).unwrap().edge_type;
        self.add_edge(s, (copy_node, 0).into(), edge_type.clone());
        self.add_edge((copy_node, 0).into(), t, edge_type);

        Ok(copy_node)
    }
}

#[derive(Debug, PartialEq)]
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
            nodes: TopSorter::new(&circ.dag, [circ.boundary.input].into()).with_cyclicity_check(),
            frontier: circ
                .dag
                .node_edges(circ.boundary.input, Direction::Outgoing)
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
                .node_edges(node, Direction::Incoming)
                .zip(self.circ.dag.node_edges(node, Direction::Outgoing))
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
