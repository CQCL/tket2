//use super::{pattern::node_equality, CircFixedStructPattern};
use crate::circuit::{circuit::Circuit, dag::Dag, operation::Op};
use portgraph::{
    graph::{Direction, NodeIndex},
    toposort::TopSortWalker,
};
use std::collections::{HashMap, VecDeque};

fn op_hash(op: &Op) -> Option<usize> {
    Some(match op {
        Op::H => 1,
        Op::CX => 2,
        Op::ZZMax => 3,
        Op::Reset => 4,
        // These shouldn't happen in the normal course of hashing
        // Op::Input => 5,
        // Op::Output => 6,
        Op::Noop(_) => 7,
        Op::Measure => 8,
        Op::Barrier => 9,
        Op::AngleAdd => 10,
        Op::AngleMul => 11,
        Op::AngleNeg => 12,
        Op::QuatMul => 13,
        // Op::Copy { n_copies, typ } => todo!(),
        Op::RxF64 => 14,
        Op::RzF64 => 15,
        Op::TK1 => 16,
        Op::Rotation => 17,
        Op::ToRotation => 18,
        // should Const of different values be hash different?
        Op::Const(_) => 19,
        // Op::Custom(_) => todo!(),
        _ => return None,
    })
}

#[derive(Debug, Clone)]
pub struct PermHash {
    hash_val: usize,
    output_order: Vec<usize>,
}

fn combine_non_assoc(ph: &PermHash, portnum: usize) -> PermHash {
    PermHash {
        hash_val: ph.hash_val * (portnum + 1),
        output_order: ph.output_order.clone(),
    }
}

fn hash_node(dag: &Dag, n: NodeIndex, edge_hashes: impl IntoIterator<Item = PermHash>) -> PermHash {
    let op_hash =
        op_hash(&dag.node_weight(n).expect("No weight for node").op).expect("Unhashable op");

    let (edge_hashes, edge_outords): (Vec<usize>, Vec<Vec<usize>>) = edge_hashes
        .into_iter()
        .enumerate()
        .map(|(near_portnum, far_end_hash)| combine_non_assoc(&far_end_hash, near_portnum))
        .map(|ph| (ph.hash_val, ph.output_order))
        .unzip();
    PermHash {
        hash_val: edge_hashes
            .iter()
            // Edge combining function here. Of course the arithmetic gets chained, so for three edges we'll have 3(3a + 5b) + 5c == 9a + 15b + 5c
            .fold(op_hash, |nh, eh| (3 * nh) + (5 * eh)),
        output_order: edge_outords.into_iter().flatten().collect(),
    }
}

// TODO take an iterator of edge classes rather than just a number
fn hash_ports(node_hash: PermHash, edges: usize) -> Vec<PermHash> {
    (0..edges)
        .map(|i| combine_non_assoc(&node_hash, i))
        .collect()
}

pub fn invariant_hash(circ: &Circuit) -> Vec<PermHash> {
    // Firstly compute "forwards" (depending on inputs) hashes of parts
    // of the graph that do not depend upon the graph input.
    let mut fwd_hashes: HashMap<NodeIndex, Vec<PermHash>> = HashMap::new();
    let source_nodes = circ
        .dag
        .node_indices()
        .filter(|n| circ.dag.node_edges(*n, Direction::Incoming).all(|_| false))
        .collect();
    for n in TopSortWalker::new(&circ.dag, source_nodes) {
        // Note, could make this more efficient using a worklist rather than topsort,
        // as topsort continues traversal over the entire graph whereas we could tell when to stop.
        if circ.dag.node_weight(n).unwrap().op == Op::Input {
            continue;
        }
        if !circ.dag.node_edges(n, Direction::Incoming).all(|e| {
            fwd_hashes.contains_key(&circ.dag.edge_endpoint(e, Direction::Incoming).unwrap())
        }) {
            continue;
        }
        let node_hash = hash_node(
            &circ.dag,
            n,
            circ.dag.node_edges(n, Direction::Incoming).map(|e| {
                let in_node = circ.dag.edge_endpoint(e, Direction::Incoming).unwrap();
                fwd_hashes.get(&in_node).unwrap()
                    [circ.port_of_edge(in_node, e, Direction::Outgoing).unwrap()]
                .clone()
            }),
        );
        fwd_hashes.insert(
            n,
            hash_ports(
                node_hash,
                circ.dag.node_edges(n, Direction::Outgoing).count(),
            ),
        );
    }
    // Now hash rest of circuit backwards, from output back to input
    let output_nodes: VecDeque<NodeIndex> = circ
        .dag
        .node_indices()
        .filter(|n| circ.dag.node_edges(*n, Direction::Outgoing).all(|_| false))
        .collect();
    assert!(
        output_nodes.len() == 1
            && circ.dag.node_weight(output_nodes[0]).map(|v| v.op.clone()) == Some(Op::Output)
    );
    // hash per input for nodes with inputs
    let mut node_hashes: HashMap<NodeIndex, Vec<PermHash>> = HashMap::new();
    let mut input_hash = None;
    node_hashes.insert(
        output_nodes[0],
        circ.dag
            .node_edges(output_nodes[0], Direction::Incoming)
            .enumerate()
            .map(|(i, _)| PermHash {
                hash_val: 6,
                output_order: [i].to_vec(),
            })
            .collect(),
    );
    for n in TopSortWalker::new(&circ.dag, output_nodes).reversed() {
        // Again, could possibly make this a bit more efficient using a worklist,
        // but the major part of that saving would probably be achieved by returning early
        if fwd_hashes.contains_key(&n) {
            continue;
        }
        let edge_targets = circ.dag.node_edges(n, Direction::Outgoing).map(|e| {
            let target = circ.dag.edge_endpoint(e, Direction::Incoming).unwrap();
            let target_port = circ.port_of_edge(target, e, Direction::Incoming).unwrap();
            node_hashes.get(&target).expect("Edge target not found")[target_port].clone()
            // TODO where output ports are equivalent, use same source_port
        });
        if circ.dag.node_weight(n).unwrap().op == Op::Input {
            // we could just return here, but we'll continue as a sanity check
            assert!(input_hash.is_none());
            input_hash = Some(edge_targets.collect());
        } else {
            // Also include fwd_hashes of any *incoming* edge
            let in_edges = circ.dag.node_edges(n, Direction::Incoming).flat_map(|e| {
                let src_node = circ.dag.edge_endpoint(e, Direction::Outgoing).unwrap();
                fwd_hashes.get(&src_node).map(|v| {
                    v[circ.port_of_edge(src_node, e, Direction::Outgoing).unwrap()].clone()
                })
            });
            let node_hash = hash_node(&circ.dag, n, edge_targets.chain(in_edges));

            node_hashes.insert(
                n,
                hash_ports(
                    node_hash,
                    circ.dag.node_edges(n, Direction::Incoming).count(),
                ),
            );
        }
    }
    input_hash.unwrap()
}

pub fn circuit_hash(circ: &Circuit) -> usize {
    let mut total: usize = 0;

    let mut hash_vals: HashMap<NodeIndex, usize> = HashMap::new();
    let [i, _] = circ.boundary();

    let _ophash = |o| 17 * 13 + op_hash(o).expect("unhashable op");
    hash_vals.insert(i, _ophash(&Op::Input));

    let initial_nodes = circ
        .dag
        .node_indices()
        .filter(|n| circ.dag.node_edges(*n, Direction::Incoming).count() == 0)
        .collect();

    for nid in TopSortWalker::new(circ.dag_ref(), initial_nodes) {
        if hash_vals.contains_key(&nid) {
            continue;
        }

        let mut myhash = _ophash(circ.node_op(nid).expect("op not found."));

        for ine in circ.node_edges(nid, Direction::Incoming) {
            let (src, _) = circ.edge_endpoints(ine).expect("edge not found.");
            debug_assert!(hash_vals.contains_key(&src));

            let mut edgehash = hash_vals[&src];

            // TODO check if overflow arithmetic is intended

            edgehash = edgehash.wrapping_mul(31).wrapping_add(
                circ.port_of_edge(src, ine, Direction::Outgoing)
                    .expect("edge not found."),
            );
            edgehash = edgehash.wrapping_mul(31).wrapping_add(
                circ.port_of_edge(nid, ine, Direction::Incoming)
                    .expect("edge not found."),
            );

            myhash = myhash.wrapping_add(edgehash);
        }
        hash_vals.insert(nid, myhash);
        total = total.wrapping_add(myhash);
    }

    total
}

#[cfg(test)]
mod tests {
    use crate::circuit::{
        circuit::UnitID,
        operation::{ConstValue, WireType},
    };

    use super::*;
    #[test]
    fn test_simple_hash() {
        let mut circ1 = Circuit::new();
        let [input, output] = circ1.boundary();

        let point5 = circ1.add_vertex(Op::Const(ConstValue::f64_angle(0.5)));
        let rx = circ1.add_vertex(Op::RzF64);
        circ1
            .add_insert_edge((input, 0), (rx, 0), WireType::Qubit)
            .unwrap();
        circ1
            .add_insert_edge((point5, 0), (rx, 1), WireType::Angle)
            .unwrap();
        circ1
            .add_insert_edge((rx, 0), (output, 0), WireType::Qubit)
            .unwrap();

        let mut circ = Circuit::new();
        let [input, output] = circ.boundary();

        // permute addition operations
        let rx = circ.add_vertex(Op::RzF64);
        let point5 = circ.add_vertex(Op::Const(ConstValue::f64_angle(0.5)));
        circ.add_insert_edge((rx, 0), (output, 0), WireType::Qubit)
            .unwrap();
        circ.add_insert_edge((input, 0), (rx, 0), WireType::Qubit)
            .unwrap();
        circ.add_insert_edge((point5, 0), (rx, 1), WireType::Angle)
            .unwrap();

        assert_eq!(circuit_hash(&circ), circuit_hash(&circ1));

        let mut circ = Circuit::new();
        let [input, output] = circ.boundary();

        let point5 = circ.add_vertex(Op::Const(ConstValue::f64_angle(0.5)));
        let rx = circ.add_vertex(Op::RxF64); // rx rather than rz
        circ.add_insert_edge((input, 0), (rx, 0), WireType::Qubit)
            .unwrap();
        circ.add_insert_edge((point5, 0), (rx, 1), WireType::Angle)
            .unwrap();
        circ.add_insert_edge((rx, 0), (output, 0), WireType::Qubit)
            .unwrap();

        assert_ne!(circuit_hash(&circ), circuit_hash(&circ1));
    }
}
