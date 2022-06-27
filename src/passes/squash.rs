use crate::{
    circuit::{
        circuit::{Circuit, CircuitRewrite, UnitID},
        dag::{Dag, Edge, Vertex, VertexProperties},
        operation::{ConstValue, Op, Param, WireType},
    },
    graph::{
        graph::{DefaultIx, Direction, EdgeIndex, NodeIndex, PortIndex},
        substitute::{BoundedSubgraph, Rewrite},
    },
    passes::{apply_exhaustive, apply_greedy, classical::find_const_ops},
};

use super::{
    pattern::{node_equality, Match},
    pattern_rewriter, CircFixedStructPattern,
};

pub fn find_singleq_rotations<'c>(circ: &'c Circuit) -> impl Iterator<Item = CircuitRewrite> + '_ {
    RotationRewriteIter {
        circ,
        vertex_it: circ.dag.node_indices(),
    }
}
pub struct RotationRewriteIter<'c, I: Iterator<Item = Vertex>> {
    circ: &'c Circuit,
    vertex_it: I,
}

// fn add_div2(circ: &mut Circuit, inp: Vertex) -> Vertex {
//     let neg = circ.add_vertex(Op::FNeg);
//     let div2 = circ.add_vertex(Op::FMul);
//     let point5 = circ.add_vertex(Op::Const(ConstValue::F64(0.5)));
//     circ.add_edge((inp, 1), (neg, 0), WireType::F64);
//     circ.add_edge((point5, 0), (div2, 0), WireType::F64);
//     circ.add_edge((neg, 0), (div2, 1), WireType::F64);

//     div2
// }

// fn unary_op_f64(circ: &mut Circuit, inp: NodePort, op: Op) -> Vertex {
//     let v = circ.add_vertex(op);
//     circ.add_edge(inp, (v, 0).into(), WireType::F64);
//     v
// }

impl<'circ, I: Iterator<Item = Vertex>> Iterator for RotationRewriteIter<'circ, I> {
    type Item = CircuitRewrite;

    fn next(&mut self) -> Option<Self::Item> {
        self.vertex_it.find_map(|n| {
            let op = &self.circ.dag.node_weight(n).unwrap().op;
            if !op.is_one_qb_gate() || matches!(op, Op::Rotation) {
                return None;
            }
            let (replace, phase) = rotation_replacement(op);
            Some(CircuitRewrite::new(
                BoundedSubgraph::from_node(&self.circ.dag, n),
                replace.into(),
                phase,
            ))
        })
    }
}

pub fn find_singleq_rotations_pattern<'c>(
    circ: &'c Circuit,
) -> impl Iterator<Item = CircuitRewrite> + '_ {
    let mut pattern_circ = Circuit::new();
    pattern_circ.add_unitid(UnitID::Qubit {
        reg_name: "q".into(),
        index: vec![0],
    });
    let [input, output] = pattern_circ.boundary();

    let rx = pattern_circ.add_vertex(Op::RxF64);
    pattern_circ.tup_add_edge((input, 0), (rx, 0), WireType::Qubit);
    pattern_circ.tup_add_edge((input, 1), (rx, 1), WireType::Angle);
    pattern_circ.tup_add_edge((rx, 0), (output, 0), WireType::Qubit);

    let nod_comp =
        |_: &Dag, _: NodeIndex, vert: &VertexProperties| !matches!(vert.op, Op::Rotation);

    let pattern = CircFixedStructPattern::from_circ(pattern_circ, nod_comp);

    let rewriter = |mat: Match<DefaultIx>| {
        let nid = mat.values().next().unwrap(); // there's only 1 node to match
        let op = &circ.dag.node_weight(*nid).unwrap().op;

        rotation_replacement(op)
    };

    pattern_rewriter(pattern, circ, rewriter)
}

// Pairwise squashing using pattern matching
pub fn squash_pattern<'c>(circ: &'c Circuit) -> impl Iterator<Item = CircuitRewrite> + '_ {
    let mut pattern_circ = Circuit::new();
    let [input, output] = pattern_circ.boundary();

    let r1 = pattern_circ.add_vertex(Op::Rotation);
    let r2 = pattern_circ.add_vertex(Op::Rotation);
    pattern_circ.tup_add_edge((input, 0), (r1, 0), WireType::Qubit);
    pattern_circ.tup_add_edge((input, 1), (r1, 1), WireType::Quat64);
    pattern_circ.tup_add_edge((r1, 0), (r2, 0), WireType::Qubit);
    pattern_circ.tup_add_edge((input, 2), (r2, 1), WireType::Quat64);
    pattern_circ.tup_add_edge((r2, 0), (output, 0), WireType::Qubit);

    let pattern = CircFixedStructPattern::from_circ(pattern_circ, node_equality());

    let mut replace_circ = Circuit::new();

    let [input, output] = replace_circ.boundary();

    let r1 = replace_circ.add_vertex(Op::Rotation);
    let mul = replace_circ.add_vertex(Op::QuatMul);
    replace_circ.tup_add_edge((input, 0), (r1, 0), WireType::Qubit);
    replace_circ.tup_add_edge((input, 1), (mul, 0), WireType::Quat64);
    replace_circ.tup_add_edge((input, 2), (mul, 1), WireType::Quat64);
    replace_circ.tup_add_edge((mul, 0), (r1, 1), WireType::Quat64);
    replace_circ.tup_add_edge((r1, 0), (output, 0), WireType::Qubit);
    let rewriter = move |_: Match<DefaultIx>| (replace_circ.clone(), 0.0);

    pattern_rewriter(pattern, circ, rewriter)
}

fn rotation_replacement(op: &Op) -> (Circuit, Param) {
    let mut replace = Circuit::new();
    let [inp, out] = replace.boundary();
    let make_quat = replace.add_vertex(Op::ToRotation);
    let rot = replace.add_vertex(Op::Rotation);
    replace.tup_add_edge((inp, 0), (rot, 0), WireType::Qubit);
    replace.tup_add_edge((make_quat, 0), (rot, 1), WireType::Quat64);
    let incoming_ports = match op {
        Op::RxF64 => {
            // replace.add_edge((inp, 1), (make_quat, 0), WireType::F64);
            let x = replace.add_vertex(Op::Const(ConstValue::F64(1.0)));
            let y = replace.add_vertex(Op::Const(ConstValue::F64(0.0)));
            let z = replace.add_vertex(Op::Const(ConstValue::F64(0.0)));
            [(inp, 1), (x, 0), (y, 0), (z, 0)]
        }
        Op::RzF64 => {
            // replace.add_edge((inp, 1), (make_quat, 0), WireType::F64);
            let x = replace.add_vertex(Op::Const(ConstValue::F64(0.0)));
            let y = replace.add_vertex(Op::Const(ConstValue::F64(0.0)));
            let z = replace.add_vertex(Op::Const(ConstValue::F64(1.0)));
            [(inp, 1), (x, 0), (y, 0), (z, 0)]
        }

        _ => panic!("Op {:?} should not have made it to this point.", op),
    };
    for (port_index, source) in incoming_ports.into_iter().enumerate() {
        let edge_type = if port_index == 0 {
            WireType::Angle
        } else {
            WireType::F64
        };
        replace.tup_add_edge(source, (make_quat, port_index as u8), edge_type);
    }
    replace.tup_add_edge((rot, 0), (out, 0), WireType::Qubit);
    (replace, 0.0)
}

// Find strings of rotation and use cacading quaternion multiplications to
// squash them
pub struct SquashFindIter<'c> {
    circ: &'c Circuit,
    current_edge: Edge,
    current_qubit_index: usize,
    qubit_edges: Vec<Edge>,
    direction: Direction,
}

impl<'c> SquashFindIter<'c> {
    pub fn new(circ: &'c Circuit) -> Self {
        let qubit_edges: Vec<_> = circ
            .dag
            .node_edges(circ.boundary()[0], Direction::Outgoing)
            .filter(|e| circ.dag.edge_weight(**e).unwrap().edge_type == WireType::Qubit)
            .copied()
            .collect();
        Self {
            circ,
            current_edge: qubit_edges[0],
            qubit_edges,
            current_qubit_index: 0,
            direction: Direction::Outgoing,
        }
    }
}
impl<'circ> Iterator for SquashFindIter<'circ> {
    type Item = CircuitRewrite;

    fn next(&mut self) -> Option<Self::Item> {
        // search along qubit edges until two rotations found
        let target_node = loop {
            let target_node = self
                .circ
                .dag
                .edge_endpoints(self.current_edge)
                .expect("edge not found.")[self.direction as usize]
                .node;

            let target_op = &self
                .circ
                .dag
                .node_weight(target_node)
                .expect("node not found")
                .op;
            match target_op {
                &Op::Input | &Op::Output => {
                    self.current_qubit_index += 1;
                    if self.current_qubit_index == self.qubit_edges.len() {
                        // finished scanning qubits
                        return None;
                    }
                    self.current_edge = self.qubit_edges[self.current_qubit_index]
                }
                _ => (),
            }
            if target_op.is_one_qb_gate() {
                if let &Op::Rotation = target_op {
                    break target_node;
                } else {
                    panic!("This finder expects only Rotation single qubit ops.")
                }
            }
        };
        let mut current_node = target_node;
        let mut rot_nodes = vec![];
        loop {
            rot_nodes.push(current_node);
            let kid = self
                .circ
                .dag
                .neighbours(current_node, self.direction)
                .next()
                .expect("single qubit op should have a successor")
                .node;
            if let &Op::Rotation = &self.circ.dag.node_weight(kid).expect("node not found").op {
                current_node = kid;
            } else {
                break;
            }
        }
        self.current_edge = *self
            .circ
            .dag
            .node_edges(rot_nodes[rot_nodes.len() - 1], self.direction)
            .next()
            .expect("single qubit op should have a successor");

        if rot_nodes.len() < 2 {
            return self.next();
        }

        let mut replace = Circuit::new();
        let [i, o] = replace.boundary();
        let rot = replace.add_vertex(Op::Rotation);
        replace.tup_add_edge((i, 0), (rot, 0), WireType::Qubit);
        replace.tup_add_edge((rot, 0), (o, 0), WireType::Qubit);

        let mut in_edges: Vec<EdgeIndex> = self
            .circ
            .dag
            .node_edges(rot_nodes[0], Direction::Incoming)
            .copied()
            .collect();
        let out_edges = vec![*self
            .circ
            .dag
            .node_edges(rot_nodes[rot_nodes.len() - 1], Direction::Outgoing)
            .next()
            .unwrap()];

        let mut accum_port = (i, 1);
        // in_edges.
        for (node_i, nod) in rot_nodes[1..].iter().enumerate() {
            let mut new_edges = self.circ.dag.node_edges(*nod, Direction::Incoming);
            new_edges.next(); // skip the qubit edge
            in_edges.extend(new_edges);

            let mul = replace.add_vertex(Op::QuatMul);
            replace.tup_add_edge(accum_port, (mul, 0), WireType::Quat64);
            replace.tup_add_edge((i, node_i as u8 + 2), (mul, 1), WireType::Quat64);
            accum_port = (mul, 0);
        }
        replace.tup_add_edge(accum_port, (rot, 1), WireType::Quat64);
        Some(CircuitRewrite {
            graph_rewrite: Rewrite::new(
                BoundedSubgraph::new(rot_nodes.into_iter().into(), [in_edges, out_edges]),
                replace.into(),
            ),
            phase: 0.0,
        })
    }
}

pub fn squash_pass(circ: Circuit) -> (Circuit, bool) {
    let mut overall_suc = false;
    let rot_replacer =
        |circuit| apply_exhaustive(circuit, |c| find_singleq_rotations(c).collect()).unwrap();
    let (circ, success) = rot_replacer(circ);

    overall_suc |= success;

    let squasher = |circuit| apply_greedy(circuit, |c| squash_pattern(c).next()).unwrap();
    let (circ, success) = squasher(circ);
    overall_suc |= success;

    let constant_folder =
        |circuit| apply_exhaustive(circuit, |c| find_const_ops(c).collect()).unwrap();
    let (circ, success) = constant_folder(circ);

    overall_suc |= success;

    (circ, overall_suc)
}

fn cx_pattern<'c>(circ: &'c Circuit) -> impl Iterator<Item = CircuitRewrite> + '_ {
    let qubits = vec![
        UnitID::Qubit {
            reg_name: "q".into(),
            index: vec![0],
        },
        UnitID::Qubit {
            reg_name: "q".into(),
            index: vec![1],
        },
    ];
    let mut replace_c = Circuit::with_uids(qubits.clone());
    replace_c
        .append_op(Op::Noop(WireType::Qubit), &[PortIndex::new(0)])
        .unwrap();
    replace_c
        .append_op(Op::Noop(WireType::Qubit), &[PortIndex::new(1)])
        .unwrap();
    let mut pattern_c = Circuit::with_uids(qubits);
    pattern_c
        .append_op(Op::CX, &[PortIndex::new(0), PortIndex::new(1)])
        .unwrap();
    pattern_c
        .append_op(Op::CX, &[PortIndex::new(0), PortIndex::new(1)])
        .unwrap();
    let pattern = CircFixedStructPattern::from_circ(pattern_c, node_equality());
    pattern_rewriter(pattern, circ, move |_| (replace_c.clone(), 0.0))
}

pub fn cx_cancel_pass(circ: Circuit) -> (Circuit, bool) {
    let (circ, suc) = apply_greedy(circ, |c| cx_pattern(c).next()).unwrap();
    let circ = circ.remove_noop();
    (circ, suc)
}
