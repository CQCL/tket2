//use super::{pattern::node_equality, CircFixedStructPattern};
use crate::circuit::{circuit::Circuit, dag::Dag, operation::Op};
use portgraph::{
    graph::{Direction, NodeIndex},
    toposort::TopSortWalker,
};
use std::collections::{HashMap, HashSet, VecDeque};

use super::outputs_table::{OutputsId, OutputsTable};

fn op_hash(op: &Op) -> usize {
    match op {
        Op::H => 1,
        Op::CX => 2,
        Op::ZZMax => 3,
        Op::Reset => 4,
        Op::Input => 5,
        Op::Output => 6,
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
        _ => panic!("Unhashable op {:?}", op),
    }
}

fn invariant_op_hash(op: &Op) -> usize {
    match op {
        // These shouldn't happen in the normal course of hashing
        Op::Input => panic!("SHouldn't hash input"),
        Op::Output => panic!("Shouldn't hash output"),
        _ => op_hash(op),
    }
}

#[derive(Debug, Clone)]
pub struct PermHash {
    hash_val: usize,
    outputs_reached: Vec<usize>, // Excluding previously-seen outputs for this input or preceding inputs
}

// Hash, returning the hash of each input and enough information
// to reconstruct the circuit from another with the same invariant_hash
pub fn invariant_hash_perm(circ: &Circuit) -> Vec<PermHash> {
    let [_, o] = circ.boundary();
    let num_outputs = circ.node_edges(o, Direction::Incoming).len();
    let mut ot = OutputsTable::new(num_outputs);
    let with_dup_outputs = invariant_hash_perm2(&mut ot, circ);
    let mut seen_outputs = HashSet::new();
    with_dup_outputs
        .iter()
        .map(|(hash_val, oid)| {
            let mut outputs_reached = Vec::new();
            //if let Some(oid) = oid {
            ot.onto_seq_deduped(*oid, &mut seen_outputs, &mut outputs_reached);
            //}
            PermHash {
                hash_val: *hash_val,
                outputs_reached,
            }
        })
        .collect()
}

pub fn invariant_hash(circ: &Circuit) -> usize {
    // Combine associatively and ignore output ordering
    invariant_hash_perm(circ)
        .into_iter()
        .fold(0, |u, ph| u ^ ph.hash_val)
}

type HashWDupOutputs = (usize, OutputsId);

fn hash_node(dag: &Dag, n: NodeIndex, edge_hashes: impl IntoIterator<Item = usize>) -> usize {
    let op_hash = invariant_op_hash(&dag.node_weight(n).expect("No weight for node").op);

    edge_hashes
        .into_iter()
        // Edge combining function here. Of course the arithmetic gets chained, so for three edges we'll have 3(3a + 5b) + 5c == 9a + 15b + 5c
        .fold(op_hash, |nh, eh| {
            nh.wrapping_mul(3).wrapping_add(eh.wrapping_mul(5))
        })
}

// TODO take an iterator of edge classes rather than just a number
fn hash_for_ports(node_hash: usize, edges: usize) -> Vec<usize> {
    (0..edges)
        .map(|i| node_hash.wrapping_mul(i.wrapping_add(1)))
        .collect()
}

fn extract_edge_hashes<T>(
    circ: &Circuit,
    node_hashes: &HashMap<NodeIndex, Vec<T>>,
    n: NodeIndex,
    d: Direction,
) -> Vec<T>
where
    T: Clone,
{
    circ.dag
        .node_edges(n, d)
        .map(|e| {
            assert!(circ.dag.edge_endpoint(e, d) == Some(n));
            let far_end = circ.dag.edge_endpoint(e, d.reverse()).unwrap();
            node_hashes.get(&far_end).expect("Edge target not found")
                [circ.port_of_edge(far_end, e, d.reverse()).unwrap()]
            .clone()
            // TODO where ports are equivalent, use same port "number" (change number -> port-class/key)
        })
        .collect()
}

fn invariant_hash_perm2(ot: &mut OutputsTable, circ: &Circuit) -> Vec<HashWDupOutputs> {
    // Firstly compute "forwards" (depending on inputs) hashes of parts
    // of the graph that do not depend upon the graph input.
    let mut fwd_hashes: HashMap<NodeIndex, Vec<usize>> = HashMap::new();
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
        assert_ne!(circ.dag.node_weight(n).unwrap().op, Op::Output);
        let node_hash = hash_node(
            &circ.dag,
            n,
            extract_edge_hashes(&circ, &fwd_hashes, n, Direction::Incoming),
        );
        fwd_hashes.insert(
            n,
            hash_for_ports(
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
    let mut node_hashes: HashMap<NodeIndex, Vec<HashWDupOutputs>> = HashMap::new();
    let mut input_hash = None;
    node_hashes.insert(
        output_nodes[0],
        circ.dag
            .node_edges(output_nodes[0], Direction::Incoming)
            .enumerate()
            .map(|(i, _)| (op_hash(&Op::Output), ot.for_graph_output(i)))
            .collect(),
    );
    for n in TopSortWalker::new(&circ.dag, output_nodes).reversed() {
        // Again, could possibly make this a bit more efficient using a worklist,
        // but the major part of that saving would probably be achieved by returning early
        if fwd_hashes.contains_key(&n) {
            continue;
        }
        let op = circ.dag.node_weight(n).unwrap().op.clone();
        if node_hashes.contains_key(&n) {
            assert!(op == Op::Output);
            continue;
        }
        let out_edges = extract_edge_hashes(circ, &node_hashes, n, Direction::Outgoing);
        if op == Op::Input {
            // we could just return here, but we'll continue as a sanity check
            assert!(input_hash.is_none());
            input_hash = Some(out_edges);
        } else {
            // Hash node, also include fwd_hashes of any *incoming* edge
            let in_edges = circ.dag.node_edges(n, Direction::Incoming).flat_map(|e| {
                let src_node = circ.dag.edge_endpoint(e, Direction::Outgoing).unwrap();
                fwd_hashes
                    .get(&src_node)
                    .map(|v| v[circ.port_of_edge(src_node, e, Direction::Outgoing).unwrap()])
            });
            let all_edges = out_edges.iter().map(|(hv, _)| *hv).chain(in_edges);
            let node_hash = hash_node(&circ.dag, n, all_edges);
            let mut outps_reached = out_edges.into_iter().map(|(_, oid)| oid);
            let first = outps_reached.next().unwrap();
            let oid = outps_reached.fold(first, |a, b| ot.sequence(a, b));

            node_hashes.insert(
                n,
                hash_for_ports(
                    node_hash,
                    circ.dag.node_edges(n, Direction::Incoming).count(),
                )
                .into_iter()
                .map(|hv| (hv, oid))
                .collect(),
            );
        }
    }
    input_hash.unwrap()
}

pub fn reinstate_permutation(
    circ: &Circuit,
    desired: &Vec<PermHash>,
) -> Result<Circuit, &'static str> {
    let num_outputs = desired
        .iter()
        .flat_map(|ph| ph.outputs_reached.clone())
        .max()
        .unwrap()
        + 1; // make exclusive
    let mut outputs_table = OutputsTable::new(num_outputs);
    let current_h = invariant_hash_perm2(&mut outputs_table, circ);
    if current_h.len() != desired.len() {
        return Err("Wrong number of inputs");
    }
    if num_outputs
        != current_h
            .iter()
            .flat_map(|(_, oid)| outputs_table.to_seq_deduped(*oid))
            .max()
            .unwrap()
            + 1
    {
        return Err("Wrong number of outputs");
    }
    let current_to_desired_input: Vec<usize> = {
        // Distinguish inputs by hash-val of each input and also the number of paths to the output
        let current_hvs: Vec<usize> = current_h.iter().map(|(hv, _)| *hv).collect();
        let desired_hvs: Vec<usize> = desired.iter().map(|ph| ph.hash_val).collect();
        // Sanity checks
        let dh_s = desired_hvs.iter().cloned().collect::<HashSet<_>>();
        let ch_s = current_hvs.iter().cloned().collect::<HashSet<_>>();
        if dh_s != ch_s {
            return Err("Not a permutation");
        }
        if dh_s.len() != desired_hvs.len() {
            // TODO of course a circuit might *want* to treat several inputs identically!
            // And then there are hash collisions, from which it is now too late to recover.
            // We could try every permutation of inputs (that satisfies equality of hash_val + #outputs)
            // and take any that succeeds (some input permutations may fail to find a valid output permutation).
            // If there is more than one, we should probably then check the circuits are actually identical
            // (because they treated several inputs the same way) and fail if not (==> not enough info in the
            // hash to distinguish between multiple permutations).
            return Err("Ambiguous, several inputs have same hash");
        }
        let hv_to_desired: HashMap<usize, usize> =
            HashMap::from_iter(desired_hvs.into_iter().enumerate().map(|(i, hv)| (hv, i)));
        current_hvs
            .iter()
            .map(|hv| *hv_to_desired.get(hv).unwrap())
            .collect()
    };
    // Try to build output map (substitution).
    // Fail if a current output has to be >1 different output in the desired circuit.
    let current_to_desired_output: Vec<usize> = {
        let mut mapping: Vec<Option<usize>> = (0..num_outputs).map(|_| None).collect();
        let mut c_outs_seen = HashSet::new();
        for (i, (_, c_out_id)) in current_h.iter().enumerate() {
            let d_outs = &desired[current_to_desired_input[i]].outputs_reached;
            let mut c_outs = Vec::new();
            outputs_table.onto_seq_deduped(*c_out_id, &mut c_outs_seen, &mut c_outs);

            assert_eq!(c_outs.len(), d_outs.len());
            for (c_output, d_output) in c_outs.iter().zip(d_outs.iter()) {
                match mapping[*c_output] {
                    None => mapping[*c_output] = Some(*d_output),
                    Some(d) => {
                        if d != *d_output {
                            return Err("Conflicting outputs");
                        }
                    }
                };
            }
        }
        mapping.into_iter().map(|o| o.unwrap()).collect()
    };
    // Finally, copy the circuit and permute inputs+outputs
    let mut res = circ.clone();
    let [i, o] = res.boundary();
    for e in res.node_edges(i, Direction::Outgoing) {
        res.remove_edge(e).unwrap();
    }
    for e in res.node_edges(o, Direction::Incoming) {
        res.remove_edge(e).unwrap();
    }
    // Node indices are the same in both circuits
    for in_idx in 0..current_h.len() {
        let e = circ.node_edges(i, Direction::Outgoing)[current_to_desired_input[in_idx]];
        let (_, target) = circ.edge_endpoints(e).unwrap();
        res.add_insert_edge(
            (i, in_idx),
            (
                target,
                circ.port_of_edge(target, e, Direction::Incoming).unwrap(),
            ),
            circ.edge_type(e).unwrap(),
        )
        .unwrap();
    }
    for out_idx in 0..num_outputs {
        let e = circ.node_edges(o, Direction::Incoming)[current_to_desired_output[out_idx]];
        let (source, _) = circ.edge_endpoints(e).unwrap();
        if source == i {
            continue;
        }; // Edges from input added by previous loop
        let src_idx = circ.port_of_edge(source, e, Direction::Outgoing).unwrap();
        res.add_insert_edge((source, src_idx), (o, out_idx), circ.edge_type(e).unwrap())
            .unwrap();
    }
    Ok(res)
}

pub fn circuit_hash(circ: &Circuit) -> usize {
    let mut total: usize = 0;

    let mut hash_vals: HashMap<NodeIndex, usize> = HashMap::new();
    let [i, _] = circ.boundary();

    let _ophash = |o| 17 * 13 + op_hash(o);
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

            // ALAN multiply to distinguish different inputs (each multiplied by 3^^different n)
            myhash = myhash.wrapping_mul(3).wrapping_add(edgehash);
        }
        hash_vals.insert(nid, myhash);
        total = total.wrapping_add(myhash);
    }

    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::operation::{ConstValue, WireType};
    use std::collections::HashSet;

    fn hash_tests(hash_fn: impl Fn(&Circuit) -> usize) {
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

        assert_eq!(hash_fn(&circ), hash_fn(&circ1));

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

        assert_ne!(hash_fn(&circ), hash_fn(&circ1));
    }

    #[test]
    fn simple_hash_test() {
        hash_tests(circuit_hash);
    }

    #[test]
    fn invariant_hash_test() {
        hash_tests(invariant_hash);
    }

    fn count_distinct_hashes(circs: &Vec<Circuit>, hash_fn: impl Fn(&Circuit) -> usize) -> usize {
        circs.iter().map(hash_fn).collect::<HashSet<_>>().len()
    }

    #[test]
    fn test_perm_hash() {
        let mut circs = Vec::new();
        for rotate_input in [0, 1] {
            for rotate_output in [0, 1] {
                // rotate one input, propagate the other
                let mut circ = Circuit::new();
                let [input, output] = circ.boundary();

                let point5 = circ.add_vertex(Op::Const(ConstValue::f64_angle(0.5)));
                let rx = circ.add_vertex(Op::RzF64);
                circ.add_insert_edge((input, rotate_input), (rx, 0), WireType::Qubit)
                    .unwrap();
                circ.add_insert_edge((point5, 0), (rx, 1), WireType::Angle)
                    .unwrap();
                circ.add_insert_edge((rx, 0), (output, rotate_output), WireType::Qubit)
                    .unwrap();
                circ.add_insert_edge(
                    (input, 1 - rotate_input),
                    (output, 1 - rotate_output),
                    WireType::Qubit,
                )
                .unwrap();
                circs.push(circ);
            }
        }
        assert_eq!(circs.len(), 4);
        assert_eq!(count_distinct_hashes(&circs, circuit_hash), 4);
        assert_eq!(count_distinct_hashes(&circs, invariant_hash), 1);

        for (_i, circ) in circs.iter().enumerate() {
            let h = invariant_hash_perm(&circ);
            for (_j, circ2) in circs.iter().enumerate() {
                let reconstituted = reinstate_permutation(circ2, &h).unwrap();
                // Circuit equality requires same ordering of edge(indices), etc. so we don't have that
                // But, we know from above that the different circuits have different circuit_hash's, so:
                assert_eq!(circuit_hash(circ), circuit_hash(&reconstituted));
            }
        }
    }
}
