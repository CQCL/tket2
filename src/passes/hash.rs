//use super::{pattern::node_equality, CircFixedStructPattern};
use crate::circuit::{circuit::Circuit, operation::Op};
use portgraph::{
    graph::{Direction, EdgeIndex, NodeIndex},
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

#[derive(Debug)]
pub struct PermHash {
    hash_val: usize,
    output_order: Vec<usize>,
}

// Hash result for a node that is a source (has no inputs) in the graph
enum SourceHash {
    InputNode(Vec<PermHash>), // a hash for each *output*
    Constant(PermHash),
}

fn combine_non_assoc(ph: &PermHash, portnum: usize) -> PermHash {
    PermHash {
        hash_val: ph.hash_val * (portnum + 1),
        output_order: ph.output_order.clone(),
    }
}

pub fn invariant_hash(circ: &Circuit) -> Vec<PermHash> {
    let initial_nodes: VecDeque<NodeIndex> = circ
        .dag
        .node_indices()
        .filter(|n| circ.dag.node_edges(*n, Direction::Outgoing).all(|_| false))
        .collect();
    assert!(
        initial_nodes.len() == 1
            && circ.dag.node_weight(initial_nodes[0]).map(|v| v.op.clone()) == Some(Op::Output)
    );
    // hash per input for nodes with inputs
    let mut input_hashes: HashMap<NodeIndex, Vec<PermHash>> = HashMap::new();
    input_hashes.insert(
        /*key*/ initial_nodes[0],
        /*value*/
        circ.dag
            .node_edges(initial_nodes[0], Direction::Incoming)
            .enumerate()
            .map(|(i, _)| PermHash {
                hash_val: 6,
                output_order: [i].to_vec(),
            })
            .collect(),
    );
    let mut source_hashes: HashMap<NodeIndex, SourceHash> = HashMap::new();
    for n in TopSortWalker::new(&circ.dag, initial_nodes).reversed() {
        let edge_targets: Vec<PermHash> = circ
            .dag
            .node_edges(n, Direction::Outgoing)
            .enumerate()
            .map(|(source_port, e)| {
                let target = circ.dag.edge_endpoint(e, Direction::Incoming).unwrap();
                let target_port = circ.port_of_edge(target, e, Direction::Incoming).unwrap();
                let ph = &input_hashes.get(&target).expect("Edge target not found")[target_port];
                // TODO where output ports are equivalent, use same source_port
                combine_non_assoc(ph, source_port)
            })
            .collect();
        if circ.dag.node_weight(n).unwrap().op == Op::Input {
            source_hashes.insert(n, SourceHash::InputNode(edge_targets));
        } else {
            let op_hash = op_hash(&circ.dag.node_weight(n).expect("No weight for node").op)
                .expect("Unhashable op");

            let node_hash = {
                let (edge_hashes, edge_outputs): (Vec<usize>, Vec<Vec<usize>>) = edge_targets
                    .into_iter()
                    .map(|p| (p.hash_val, p.output_order))
                    .unzip();
                PermHash {
                    // Edge combining function here. Of course the arithmetic gets chained, so for three edges we'll have 3(3a + 5b) + 5c == 9a + 15b + 5c
                    hash_val: edge_hashes
                        .into_iter()
                        .fold(op_hash, |a, b| (a * 3) + (5 * b)),
                    output_order: edge_outputs.into_iter().flatten().collect(),
                }
            };
            let in_edges: Vec<EdgeIndex> = circ.dag.node_edges(n, Direction::Incoming).collect();
            if in_edges.len() == 0 {
                source_hashes.insert(n, SourceHash::Constant(node_hash));
            } else {
                input_hashes.insert(
                    n,
                    in_edges
                        .iter()
                        .enumerate()
                        // TODO where input ports are equivalent, use same target_port
                        .map(|(target_port, _)| combine_non_assoc(&node_hash, target_port))
                        .collect(),
                );
            }
        }
    }
    let mut input_node = None;
    let mut all_constants_hash = 0;
    for (_, nh) in source_hashes {
        match nh {
            SourceHash::InputNode(outs) => {
                assert!(input_node.is_none());
                input_node = Some(outs)
            }
            SourceHash::Constant(ph) => all_constants_hash ^= ph.hash_val, // (must?) ignore output order
        }
    }
    input_node
        .unwrap()
        .iter()
        .map(|ph| combine_non_assoc(ph, all_constants_hash))
        .collect()
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
