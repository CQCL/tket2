use std::collections::HashMap;
use std::error::Error;
use std::fmt::{Debug, Display};
use std::hash::Hash;

use crate::graph::graph::{DefaultIx, Direction, NodePort, PortIndex};
use crate::graph::substitute::{BoundedSubgraph, OpenGraph};

use super::dag::{Dag, Edge, EdgeProperties, TopSorter, Vertex, VertexProperties};
use super::operation::{ConstValue, Op, Param, WireType};
#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

#[cfg_attr(feature = "pyo3", derive(FromPyObject))]
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum UnitID {
    Qubit { name: String, index: Vec<u32> },
    Bit { name: String, index: Vec<u32> },
    F64(String),
    Angle(String),
}

impl UnitID {
    pub fn get_type(&self) -> WireType {
        match self {
            Self::Qubit { .. } => WireType::Qubit,
            Self::Bit { .. } => WireType::LinearBit,
            Self::F64(_) => WireType::F64,
            Self::Angle(_) => WireType::Angle,
        }
    }
}

#[derive(Clone, PartialEq)]
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

#[cfg_attr(feature = "pyo3", pyclass(name = "RsCircuit"))]
#[derive(Clone, PartialEq)]
pub struct Circuit {
    pub(crate) dag: Dag,
    pub name: Option<String>,
    pub phase: Param,
    boundary: Boundary,
    pub(crate) uids: Vec<UnitID>,
}

impl Default for Circuit {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub struct CircuitError(pub String);

impl Display for CircuitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("CircuitError {}", self.0))
    }
}
impl Error for CircuitError {}

impl From<String> for CircuitError {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for CircuitError {
    fn from(s: &str) -> Self {
        s.to_string().into()
    }
}

#[cfg_attr(feature = "pyo3", pymethods)]
impl Circuit {
    pub fn add_vertex(&mut self, op: Op) -> Vertex {
        let capacity = op.signature().map_or(0, |sig| sig.len());
        let weight = VertexProperties::new(op);
        self.dag.add_node_with_capacity(capacity, weight)
    }

    pub fn add_edge(&mut self, source: NodePort, target: NodePort, edge_type: WireType) -> Edge {
        self.dag
            .add_edge(source, target, EdgeProperties { edge_type })
    }

    pub fn add_unitid(&mut self, uid: UnitID) -> NodePort {
        self.uids.push(uid);
        (self.boundary.input, (self.uids.len() - 1) as u8).into()
    }

    pub fn add_linear_unitid(&mut self, uid: UnitID) {
        let (_, inlen) = self.dag.node_boundary_size(self.boundary.input);
        let (outlen, _) = self.dag.node_boundary_size(self.boundary.output);
        self.tup_add_edge(
            (self.boundary.input, inlen as u8),
            (self.boundary.output, outlen as u8),
            uid.get_type(),
        );
        self.uids.push(uid);
    }

    pub fn node_op(&self, n: Vertex) -> Option<Op> {
        self.dag.node_weight(n).map(|vp| vp.op.clone())
    }

    pub fn edge_type(&self, e: Edge) -> Option<WireType> {
        self.dag.edge_weight(e).map(|ep| ep.edge_type)
    }
    pub fn dot_string(&self) -> String {
        crate::graph::dot::dot_string(self.dag_ref())
    }

    pub fn apply_rewrite(&mut self, rewrite: CircuitRewrite) -> Result<(), CircuitError> {
        self.dag.apply_rewrite(rewrite.graph_rewrite)?;
        self.phase += rewrite.phase;
        Ok(())
    }

    pub fn insert_at_edge(
        &mut self,
        vert: Vertex,
        edge: Edge,
        ports: [PortIndex; 2],
    ) -> Result<(), CircuitError> {
        let [s, t] = self
            .dag
            .edge_endpoints(edge)
            .ok_or_else(|| "Edge not found.".to_string())?;
        self.dag.update_edge(edge, s, NodePort::new(vert, ports[0]));

        self.tup_add_edge(
            NodePort::new(vert, ports[1]),
            t,
            self.dag.edge_weight(edge).unwrap().edge_type,
        );
        Ok(())
    }
}
impl Circuit {
    pub fn new() -> Self {
        Self::with_uids(vec![])
    }

    pub fn with_uids(uids: Vec<UnitID>) -> Self {
        let n_uids = uids.len();
        let mut dag = Dag::with_capacity(2, n_uids);
        let input = dag.add_node(VertexProperties::new(Op::Input));
        let output = dag.add_node(VertexProperties::new(Op::Output));
        let mut slf = Self {
            dag,
            name: None,
            phase: 0.0,
            boundary: Boundary { input, output },
            uids: Vec::with_capacity(n_uids),
        };
        for uid in uids {
            slf.add_linear_unitid(uid);
        }
        slf
    }

    pub fn bind_input<P: Into<PortIndex>>(
        &mut self,
        in_port: P,
        val: ConstValue,
    ) -> Result<Vertex, CircuitError> {
        let e = self
            .dag
            .edge_at_port(
                NodePort::new(self.boundary.input, in_port.into()),
                Direction::Outgoing,
            )
            .ok_or_else(|| "No such input".to_string())?;
        let [_, target] = self.dag.edge_endpoints(e).unwrap();
        let existing_typ = self.dag.remove_edge(e).unwrap().edge_type;
        if val.get_type() != existing_typ {
            return Err("Edge type of input does not match type of provided value.".into());
        }
        let cons = self.add_vertex(Op::Const(val));
        self.tup_add_edge((cons, 0).into(), target, existing_typ);
        Ok(cons)
    }

    pub fn tup_add_edge<T: Into<NodePort>>(
        &mut self,
        source: T,
        target: T,
        edge_type: WireType,
    ) -> Edge {
        self.add_edge(source.into(), target.into(), edge_type)
    }

    pub fn append_op(&mut self, op: Op, args: &[PortIndex]) -> Result<Vertex, CircuitError> {
        // akin to add-op in TKET-1

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
            .ok_or_else(|| "Invalid output ports".to_string())?;
        for (p, e) in insertion_edges.into_iter().enumerate() {
            let p = PortIndex::new(p);
            self.insert_at_edge(new_vert, e, [p; 2])?;
        }

        Ok(new_vert)
    }

    pub fn to_commands(&self) -> CommandIter {
        CommandIter::new(self)
    }

    pub fn qubits(&self) -> impl Iterator<Item = UnitID> + '_ {
        self.uids.iter().filter_map(|uid| match uid {
            UnitID::Qubit { .. } => Some(uid.clone()),
            UnitID::Bit { .. } | UnitID::F64(_) | UnitID::Angle(_) => None,
        })
    }

    pub fn bits(&self) -> impl Iterator<Item = UnitID> + '_ {
        self.uids.iter().filter_map(|uid| match uid {
            UnitID::Bit { .. } => Some(uid.clone()),
            UnitID::Qubit { .. } | UnitID::F64(_) | UnitID::Angle(_) => None,
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
                typ: *edge_type,
            },
        };
        let copy_node = self.add_vertex(copy_op);
        let [s, t] = self.dag.edge_endpoints(e).unwrap();
        let edge_type = self.dag.remove_edge(e).unwrap().edge_type;
        self.tup_add_edge(s, (copy_node, 0).into(), edge_type);
        self.tup_add_edge((copy_node, 0).into(), t, edge_type);

        Ok(copy_node)
    }

    // pub fn apply_rewrite(&mut self, rewrite: CircuitRewrite) -> Result<(), String> {
    //     self.dag.apply_rewrite(rewrite.graph_rewrite)?;
    //     self.phase += rewrite.phase;
    //     Ok(())
    // }
    pub fn remove_invalid(mut self) -> Self {
        let (g, node_map, _) = self.dag.remove_invalid();
        self.dag = g;
        self.boundary = Boundary {
            input: node_map[&self.boundary.input],
            output: node_map[&self.boundary.output],
        };
        self
    }

    pub fn remove_noop(mut self) -> Self {
        let noop_nodes: Vec<_> = self
            .dag
            .node_indices()
            .filter_map(|n| {
                if let Op::Noop(wt) = self.dag.node_weight(n).unwrap().op {
                    Some((wt, n))
                } else {
                    None
                }
            })
            .collect();
        for (edge_type, nod) in noop_nodes {
            let source = self
                .dag
                .neighbours(nod, Direction::Incoming)
                .next()
                .unwrap();
            let target = self
                .dag
                .neighbours(nod, Direction::Outgoing)
                .next()
                .unwrap();
            self.dag.remove_node(nod);
            self.dag
                .add_edge(source, target, EdgeProperties { edge_type });
        }

        self
    }

    pub fn dag_ref(&self) -> &Dag {
        &self.dag
    }
}

#[derive(Debug, PartialEq)]
pub struct Command<'c> {
    pub vertex: Vertex,
    pub op: &'c Op,
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
            nodes: TopSorter::new(
                &circ.dag,
                circ.dag
                    .node_indices()
                    .filter(|n| circ.dag.node_boundary_size(*n).0 == 0)
                    .collect(),
            )
            .with_cyclicity_check(),
            frontier: circ
                .dag
                .node_edges(circ.boundary.input, Direction::Outgoing)
                .enumerate()
                .map(|(i, e)| (*e, circ.uids.get(i).unwrap()))
                .collect(),
            circ,
        }
    }
}

impl<'circ> Iterator for CommandIter<'circ> {
    type Item = Command<'circ>;

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
                vertex: node,
                op,
                args,
            }
        })
    }
}

pub(crate) type CircDagRewrite =
    crate::graph::substitute::Rewrite<VertexProperties, EdgeProperties>;

#[cfg_attr(feature = "pyo3", pyclass)]
#[derive(Debug, Clone)]
pub struct CircuitRewrite {
    pub graph_rewrite: CircDagRewrite,
    pub phase: Param,
}

impl CircuitRewrite {
    pub fn new(
        subg: BoundedSubgraph<DefaultIx>,
        replacement: OpenGraph<VertexProperties, EdgeProperties, DefaultIx>,
        phase: Param,
    ) -> Self {
        Self {
            graph_rewrite: CircDagRewrite { subg, replacement },
            phase,
        }
    }
}

impl From<Circuit> for OpenGraph<VertexProperties, EdgeProperties, DefaultIx> {
    fn from(mut c: Circuit) -> Self {
        let [entry, exit] = c.boundary();
        let in_ports = c.dag.neighbours(entry, Direction::Outgoing).collect();
        let out_ports = c.dag.neighbours(exit, Direction::Incoming).collect();

        c.dag.remove_node(entry);
        c.dag.remove_node(exit);
        Self {
            in_ports,
            out_ports,
            graph: c.dag,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        circuit::operation::{ConstValue, Op, WireType},
        graph::graph::{Direction, PortIndex},
    };

    use super::Circuit;

    #[test]
    fn test_add_identity() {
        let mut circ = Circuit::new();
        let [i, o] = circ.boundary();
        for p in 0..2 {
            let noop = circ.add_vertex(Op::Noop(WireType::Qubit));
            circ.tup_add_edge((i, p), (noop, 0), WireType::Qubit);
            circ.tup_add_edge((noop, 0), (o, p), WireType::Qubit);
        }
    }

    #[test]
    fn test_bind_value() {
        let mut circ = Circuit::new();
        let [i, o] = circ.boundary();
        let add = circ.add_vertex(Op::AngleAdd);

        circ.tup_add_edge((i, 0), (add, 0), WireType::F64);
        circ.tup_add_edge((i, 1), (add, 1), WireType::F64);
        circ.tup_add_edge((add, 0), (o, 0), WireType::F64);

        assert_eq!(circ.dag.edge_count(), 3);
        assert_eq!(circ.dag.node_count(), 3);
        assert_eq!(circ.dag.node_edges(i, Direction::Outgoing).count(), 2);
        circ.bind_input(PortIndex::new(0), ConstValue::F64(1.0))
            .unwrap();
        circ.bind_input(PortIndex::new(1), ConstValue::F64(2.0))
            .unwrap();

        assert_eq!(circ.dag.edge_count(), 3);
        assert_eq!(circ.dag.node_count(), 5);
        assert_eq!(circ.dag.node_edges(i, Direction::Outgoing).count(), 0);

        let mut neis = circ.dag.neighbours(add, Direction::Incoming);

        assert_eq!(
            circ.dag.node_weight(neis.next().unwrap().node).unwrap().op,
            Op::Const(ConstValue::F64(1.0))
        );
        assert_eq!(
            circ.dag.node_weight(neis.next().unwrap().node).unwrap().op,
            Op::Const(ConstValue::F64(2.0))
        );
    }
}
