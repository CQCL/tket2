use std::collections::HashMap;
use std::error::Error;
use std::fmt::{Debug, Display};
use std::hash::Hash;

use portgraph::graph::{ConnectError, Direction, DIRECTIONS};
use portgraph::substitute::{BoundedSubgraph, OpenGraph, RewriteError};

use super::dag::{Dag, Edge, EdgeProperties, TopSorter, Vertex, VertexProperties};
use super::operation::{ConstValue, Op, Param, WireType};
#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

#[cfg_attr(feature = "pyo3", derive(FromPyObject))]
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum UnitID {
    Qubit { reg_name: String, index: Vec<u32> },
    Bit { name: String, index: Vec<u32> },
    I64(String),
    Bool(String),
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
            Self::I64(_) => WireType::I64,
            Self::Bool(_) => WireType::Bool,
        }
    }

    pub fn new_q(i: u32) -> Self {
        Self::Qubit {
            reg_name: "q".into(),
            index: vec![i],
        }
    }

    pub fn new_b(i: u32) -> Self {
        Self::Bit {
            name: "q".into(),
            index: vec![i],
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
        // let capacity = op.signature().map_or(0, |sig| sig.len());
        let weight = VertexProperties::new(op);
        self.dag.add_node(weight)
    }

    pub fn add_insert_edge(
        &mut self,
        source: (Vertex, usize),
        target: (Vertex, usize),
        edge_type: WireType,
    ) -> Result<Edge, ConnectError> {
        let e = self.add_edge(edge_type);
        self.dag
            .insert_edge(source.0, e, Direction::Outgoing, source.1)?;
        self.dag
            .insert_edge(target.0, e, Direction::Incoming, target.1)?;
        Ok(e)
    }

    pub fn add_vertex_with_edges(
        &mut self,
        op: Op,
        incoming: Vec<Edge>,
        outgoing: Vec<Edge>,
    ) -> Vertex {
        let weight = VertexProperties::new(op);
        self.dag
            .add_node_with_edges(weight, incoming, outgoing)
            .unwrap()
    }
    pub fn add_edge(&mut self, edge_type: WireType) -> Edge {
        self.dag.add_edge(EdgeProperties { edge_type })
    }

    pub fn add_unitid(&mut self, uid: UnitID) -> Edge {
        let e = self.add_edge(uid.get_type());
        self.uids.push(uid);

        self.dag
            .connect_last(self.boundary.input, e, Direction::Outgoing)
            .unwrap();
        e
    }

    pub fn add_linear_unitid(&mut self, uid: UnitID) {
        let ie = self.add_edge(uid.get_type());

        self.dag
            .connect_last(self.boundary.input, ie, Direction::Outgoing)
            .unwrap();
        self.dag
            .connect_last(self.boundary.output, ie, Direction::Incoming)
            .unwrap();
        // let [_, inlen] = self.dag.node_boundary_size(self.boundary.input);
        // let [outlen, _] = self.dag.node_boundary_size(self.boundary.output);
        // self.tup_add_edge(
        //     (self.boundary.input, inlen as u8),
        //     (self.boundary.output, outlen as u8),
        //     uid.get_type(),
        // );
        self.uids.push(uid);
    }

    pub fn node_op(&self, n: Vertex) -> Option<&Op> {
        self.dag.node_weight(n).map(|vp| &vp.op)
    }

    pub fn edge_type(&self, e: Edge) -> Option<WireType> {
        self.dag.edge_weight(e).map(|ep| ep.edge_type)
    }
    pub fn dot_string(&self) -> String {
        portgraph::dot::dot_string(self.dag_ref())
    }

    pub fn apply_rewrite(&mut self, rewrite: CircuitRewrite) -> Result<(), RewriteError> {
        self.dag.apply_rewrite(rewrite.graph_rewrite)?;
        self.phase += rewrite.phase;
        Ok(())
    }

    // pub fn insert_at_edge(
    //     &mut self,
    //     vert: Vertex,
    //     edge: Edge,
    //     ports: [PortIndex; 2],
    // ) -> Result<(), CircuitError> {
    //     let [s, t] = self
    //         .dag
    //         .edge_endpoints(edge)
    //         .ok_or_else(|| "Edge not found.".to_string())?;
    //     self.dag.update_edge(edge, s, NodePort::new(vert, ports[0]));

    //     self.tup_add_edge(
    //         NodePort::new(vert, ports[1]),
    //         t,
    //         self.dag.edge_weight(edge).unwrap().edge_type,
    //     );
    //     Ok(())
    // }

    pub fn remove_node(&mut self, n: Vertex) -> Option<Op> {
        self.dag.remove_node(n).map(|v| v.op)
    }

    pub fn remove_edge(&mut self, e: Edge) -> Option<WireType> {
        self.dag.remove_edge(e).map(|ep| ep.edge_type)
    }

    pub fn edge_endpoints(&self, e: Edge) -> Option<(Vertex, Vertex)> {
        let s = self.dag.edge_endpoint(e, Direction::Outgoing)?;
        let t = self.dag.edge_endpoint(e, Direction::Incoming)?;
        // self.dag.edge_endpoints(e).map(|[e1, e2]| (e1, e2))
        Some((s, t))
    }

    pub fn edge_at_port(&self, n: Vertex, port: usize, direction: Direction) -> Option<Edge> {
        self.dag.node_edges(n, direction).nth(port)
    }

    pub fn port_of_edge(&self, n: Vertex, edge: Edge, direction: Direction) -> Option<usize> {
        self.dag
            .node_edges(n, direction)
            .enumerate()
            .find_map(|(i, oe)| (oe == edge).then_some(i))
    }

    pub fn node_edges(&self, n: Vertex, direction: Direction) -> Vec<Edge> {
        self.dag.node_edges(n, direction).collect()
    }

    pub fn node_boundary_size(&self, n: Vertex) -> [usize; 2] {
        DIRECTIONS.map(|direction| self.dag.node_edges(n, direction).count())
    }

    pub fn neighbours(&self, n: Vertex, direction: Direction) -> Vec<Vertex> {
        self.dag
            .node_edges(n, direction)
            .map(|e| self.dag.edge_endpoint(e, direction.reverse()).unwrap())
            .collect()
    }

    pub fn add_const(&mut self, c: ConstValue) -> Vertex {
        self.add_vertex(Op::Const(c))
    }

    pub fn get_const(&self, n: Vertex) -> Option<&ConstValue> {
        self.node_op(n).and_then(|op| match op {
            Op::Const(c) => Some(c),
            _ => None,
        })
    }

    pub fn node_count(&self) -> usize {
        self.dag.node_count()
    }

    pub fn edge_count(&self) -> usize {
        self.dag.edge_count()
    }

    pub fn new_input(&mut self, edge_type: WireType) -> Edge {
        let e = self.add_edge(edge_type);
        self.dag
            .connect_first(self.boundary.input, e, Direction::Outgoing)
            .unwrap();
        e
    }

    pub fn new_output(&mut self, edge_type: WireType) -> Edge {
        let e = self.add_edge(edge_type);
        self.dag
            .connect_first(self.boundary.output, e, Direction::Incoming)
            .unwrap();
        e
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

    pub fn bind_input(&mut self, in_port: usize, val: ConstValue) -> Result<Vertex, CircuitError> {
        let e = self
            .edge_at_port(self.boundary.input, in_port, Direction::Outgoing)
            .ok_or_else(|| "No such input".to_string())?;
        // let target = self.dag.edge_endpoint(e, Direction::Incoming);
        // let [_, target] = self.dag.edge_endpoints(e).unwrap();
        // let existing_typ = self.dag.remove_edge(e).unwrap().edge_type;
        if val.get_type() != self.edge_type(e).unwrap() {
            return Err("Edge type of input does not match type of provided value.".into());
        }
        self.dag.disconnect(e, Direction::Outgoing);
        // let oes = vec![self.add_edge(existing_typ)];
        Ok(self.add_vertex_with_edges(Op::Const(val), vec![], vec![e]))
        // self.tup_add_edge((cons, 0).into(), target, existing_typ);
        // Ok(cons)
    }

    pub fn append_op(&mut self, op: Op, args: &[usize]) -> Result<Vertex, ConnectError> {
        // akin to add-op in TKET-1

        // let new_vert = self.add_vertex(op);
        let out_edges: Vec<_> = self
            .dag
            .node_edges(self.boundary.output, Direction::Incoming)
            .collect();
        let insertion_edges = args
            .iter()
            .map(|port| out_edges.get(*port).copied())
            .collect::<Option<Vec<Edge>>>()
            .ok_or(ConnectError::UnknownEdge)?;

        let mut incoming = vec![];
        // let mut outgoing = vec![];
        for e in insertion_edges.iter() {
            let e_type = self
                .dag
                .edge_weight(*e)
                .expect("Edge should be there.")
                .edge_type;
            let in_e = self.add_edge(e_type);
            self.dag.replace_connection(*e, in_e, Direction::Outgoing)?;
            // let prev = self.dag.edge_endpoint(*e, Direction::Outgoing).unwrap();
            // self.dag.connect_after(prev, in_e, Direction::Outgoing, *e);
            // self.dag.disconnect(*e, Direction::Outgoing);
            incoming.push(in_e);
            // outgoing.push(self.add_edge(e_type));
            // let p = PortIndex::new(p);
            // self.insert_at_edge(new_vert, e, [p; 2])?;
        }
        self.dag
            .add_node_with_edges(VertexProperties { op }, incoming, insertion_edges)

        // Ok(new_vert)
    }

    pub fn to_commands(&self) -> CommandIter {
        CommandIter::new(self)
    }

    pub fn commands_with_unitid(&self, unitid: UnitID) -> impl Iterator<Item = Command> {
        self.to_commands()
            .filter(move |cmd| cmd.args.contains(&unitid))
    }

    pub fn qubits(&self) -> impl Iterator<Item = UnitID> + '_ {
        self.uids.iter().filter_map(|uid| match uid {
            UnitID::Qubit { .. } => Some(uid.clone()),
            UnitID::Bit { .. }
            | UnitID::F64(_)
            | UnitID::Angle(_)
            | UnitID::I64(_)
            | UnitID::Bool(_) => None,
        })
    }

    pub fn bits(&self) -> impl Iterator<Item = UnitID> + '_ {
        self.uids.iter().filter_map(|uid| match uid {
            UnitID::Bit { .. } => Some(uid.clone()),
            UnitID::Qubit { .. }
            | UnitID::F64(_)
            | UnitID::Angle(_)
            | UnitID::I64(_)
            | UnitID::Bool(_) => None,
        })
    }

    pub fn unitids(&self) -> impl Iterator<Item = &UnitID> + '_ {
        self.uids.iter()
    }

    pub fn boundary(&self) -> [Vertex; 2] {
        [self.boundary.input, self.boundary.output]
    }

    /// send an edge in to a copy vertex and return
    /// the N-1 new edges (with the first being connected to the existing target)
    pub fn copy_edge(&mut self, e: Edge, copies: u32) -> Result<Vec<Edge>, String> {
        let edge_type = match self.dag.edge_weight(e) {
            Some(EdgeProperties { edge_type, .. }) => *edge_type,
            _ => return Err("Edge not found".into()),
        };

        let copy_op = match edge_type {
            WireType::Qubit | WireType::LinearBit => {
                return Err("Cannot copy qubit or LinearBit wires.".into())
            }
            _ => Op::Copy {
                n_copies: copies,
                typ: edge_type,
            },
        };

        let mut copy_es: Vec<_> = (0..copies).map(|_| self.add_edge(edge_type)).collect();
        // let copy_node = self.add_vertex(copy_op);
        self.dag
            .replace_connection(e, copy_es[0], Direction::Incoming)
            .unwrap();
        // self.dag.disconnect(e, Direction::Incoming);
        // let edge_type = self.dag.remove_edge(e).unwrap().edge_type;
        // self.tup_add_edge(s, (copy_node, 0).into(), edge_type);
        // self.tup_add_edge((copy_node, 0).into(), t, edge_type);

        // Ok(copy_node)
        self.add_vertex_with_edges(copy_op, vec![e], copy_es.clone());

        copy_es.remove(0);
        Ok(copy_es)
    }

    // pub fn apply_rewrite(&mut self, rewrite: CircuitRewrite) -> Result<(), String> {
    //     self.dag.apply_rewrite(rewrite.graph_rewrite)?;
    //     self.phase += rewrite.phase;
    //     Ok(())
    // }
    pub fn remove_invalid(mut self) -> Self {
        let (node_map, _) = self.dag.compact();
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
            .filter(|n| matches!(self.dag.node_weight(*n).unwrap().op, Op::Noop(_)))
            .collect();
        for nod in noop_nodes {
            let ie = self
                .dag
                .node_edges(nod, Direction::Incoming)
                .next()
                .unwrap();
            let oe = self
                .dag
                .node_edges(nod, Direction::Outgoing)
                .next()
                .unwrap();

            let target = self.dag.edge_endpoint(oe, Direction::Incoming).unwrap();
            self.dag.disconnect(ie, Direction::Incoming);

            self.dag
                .connect_after(target, ie, Direction::Incoming, oe)
                .unwrap();
            self.dag.remove_edge(oe);
            self.dag.remove_node(nod);
            // let source = self
            //     .dag
            //     .neighbours(nod, Direction::Incoming)
            //     .next()
            //     .unwrap();
            // let target = self
            //     .dag
            //     .neighbours(nod, Direction::Outgoing)
            //     .next()
            //     .unwrap();
            // self.dag
            //     .add_edge(source, target, EdgeProperties { edge_type });
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
                    .filter(|n| circ.dag.node_edges(*n, Direction::Incoming).count() == 0)
                    .collect(),
            )
            .with_cyclicity_check(),
            frontier: circ
                .dag
                .node_edges(circ.boundary.input, Direction::Outgoing)
                .enumerate()
                .map(|(i, e)| (e, circ.uids.get(i).unwrap()))
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
                    let uid = self.frontier.remove(&in_e).expect("edge not in frontier");
                    self.frontier.insert(out_e, uid);
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

pub(crate) type CircDagRewrite = portgraph::substitute::Rewrite<VertexProperties, EdgeProperties>;

#[cfg_attr(feature = "pyo3", pyclass)]
#[derive(Debug, Clone)]
pub struct CircuitRewrite {
    pub graph_rewrite: CircDagRewrite,
    pub phase: Param,
}

impl CircuitRewrite {
    pub fn new(
        subg: BoundedSubgraph,
        replacement: OpenGraph<VertexProperties, EdgeProperties>,
        phase: Param,
    ) -> Self {
        Self {
            graph_rewrite: CircDagRewrite { subg, replacement },
            phase,
        }
    }
}

impl From<Circuit> for OpenGraph<VertexProperties, EdgeProperties> {
    fn from(mut c: Circuit) -> Self {
        let [entry, exit] = c.boundary();
        let in_ports = c.dag.node_edges(entry, Direction::Outgoing).collect();
        let out_ports = c.dag.node_edges(exit, Direction::Incoming).collect();

        c.dag.remove_node(entry);
        c.dag.remove_node(exit);
        Self {
            ports: [in_ports, out_ports],
            graph: c.dag,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::circuit::operation::{ConstValue, Op, WireType};
    use portgraph::graph::Direction;

    use super::{Circuit, UnitID};

    #[test]
    fn test_add_identity() {
        let mut circ = Circuit::new();
        // let [i, o] = circ.boundary();
        for _ in 0..2 {
            let ie = circ.new_input(WireType::Qubit);
            let oe = circ.new_output(WireType::Qubit);
            let _noop = circ.add_vertex_with_edges(Op::Noop(WireType::Qubit), vec![ie], vec![oe]);
            // circ.dag.connect_first(i, ie, Direction::Outgoing);
            // circ.dag.connect_first(o, oe, Direction::Incoming);
            // circ.tup_add_edge((i, p), (noop, 0), WireType::Qubit);
            // circ.tup_add_edge((noop, 0), (o, p), WireType::Qubit);
        }
    }

    #[test]
    fn test_bind_value() {
        let mut circ = Circuit::new();
        // let [i, o] = circ.boundary();

        let i1 = circ.new_input(WireType::F64);
        let i0 = circ.new_input(WireType::F64);

        let o = circ.new_output(WireType::F64);
        let add = circ.add_vertex_with_edges(Op::AngleAdd, vec![i0, i1], vec![o]);

        // circ.tup_add_edge((i, 0), (add, 0), WireType::F64);
        // circ.tup_add_edge((i, 1), (add, 1), WireType::F64);
        // circ.tup_add_edge((add, 0), (o, 0), WireType::F64);

        assert_eq!(circ.dag.edge_count(), 3);
        assert_eq!(circ.dag.node_count(), 3);
        assert_eq!(
            circ.dag
                .node_edges(circ.boundary.input, Direction::Outgoing)
                .count(),
            2
        );

        // println!("{}", circ.dag.node_edges(circ.boundary.output, Direction::Incoming).count());
        // println!("{:?}", circ.edge_at_port(circ.boundary.input, 1, Direction::Outgoing));

        circ.bind_input(0, ConstValue::F64(1.0)).unwrap();
        circ.bind_input(0, ConstValue::F64(2.0)).unwrap();
        println!("{}", circ.dot_string());

        assert_eq!(circ.dag.edge_count(), 3);
        assert_eq!(circ.dag.node_count(), 5);
        assert_eq!(
            circ.dag
                .node_edges(circ.boundary.input, Direction::Outgoing)
                .count(),
            0
        );

        let neis = circ.neighbours(add, Direction::Incoming);

        assert_eq!(
            circ.dag.node_weight(neis[0]).unwrap().op,
            Op::Const(ConstValue::F64(1.0))
        );
        assert_eq!(
            circ.dag.node_weight(neis[1]).unwrap().op,
            Op::Const(ConstValue::F64(2.0))
        );
    }

    #[test]
    fn test_add_uid() {
        let q0 = UnitID::Qubit {
            reg_name: "q".into(),
            index: vec![0],
        };
        let a = UnitID::Angle("a".into());

        // An empty circuit with UnitIDs [q0, a]
        let mut c = Circuit::new();
        c.add_linear_unitid(q0.clone());
        c.add_unitid(a.clone());

        // Make sure UnitIDs and edges are stored in right order
        assert_eq!(c.uids, vec![q0.clone(), a.clone()]);
        assert_eq!(
            c.node_edges(c.boundary.input, Direction::Outgoing)
                .into_iter()
                .map(|e| c.edge_type(e).unwrap())
                .collect::<Vec<_>>(),
            vec![WireType::Qubit, WireType::Angle]
        )
    }
}
