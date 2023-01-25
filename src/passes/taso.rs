use crate::circuit::{circuit::Circuit, operation::Op};
use portgraph::{
    graph::{Direction, NodeIndex},
    toposort::TopSortWalker,
};
use priority_queue::PriorityQueue;
use rayon::prelude::*;
use std::{
    collections::{HashMap, HashSet},
    sync::Mutex,
};

use super::{pattern::node_equality, pattern_rewriter, CircFixedStructPattern};

type Transformation = (Circuit, Circuit);

pub fn taso<C>(
    circ: Circuit,
    transforms: Vec<Transformation>,
    gamma: f64,
    cost: C,
    _timeout: i64,
) -> Circuit
where
    C: Fn(&Circuit) -> usize + Send + Sync,
{
    // TODO timeout
    let tra_patterns: Vec<_> = transforms
        .into_iter()
        .map(|(c1, c2)| (CircFixedStructPattern::from_circ(c1, node_equality()), c2))
        .collect();

    let _rev_cost = |x: &HashCirc| usize::MAX - cost(&x.0);

    let mut pq = PriorityQueue::new();
    let hc = HashCirc(circ);
    let mut cbest = hc.clone();
    let cin_cost = _rev_cost(&hc);
    let mut cbest_cost = cin_cost;
    let dseen: Mutex<HashSet<usize>> = Mutex::new(HashSet::from_iter([circuit_hash(&hc.0)]));
    pq.push(hc, cin_cost);

    while let Some((hc, priority)) = pq.pop() {
        if priority > cbest_cost {
            // TODO here is where a optimal data-sharing copy would be handy

            cbest = hc.clone();
            cbest_cost = priority;
        }
        let pq = Mutex::new(&mut pq);
        tra_patterns.par_iter().for_each(|(pattern, c2)| {
            pattern_rewriter(pattern.clone(), &hc.0, |_| (c2.clone(), 0.0)).for_each(|rewrite| {
                let mut newc = hc.0.clone();
                newc.apply_rewrite(rewrite).expect("rewrite failure");
                let newchash = circuit_hash(&newc);
                let mut dseen = dseen.lock().unwrap();
                if dseen.contains(&newchash) {
                    return;
                }
                let newhc = HashCirc(newc);
                let newcost = _rev_cost(&newhc);
                if gamma * (newcost as f64) > (cbest_cost as f64) {
                    let mut pq = pq.lock().unwrap();
                    pq.push(newhc, newcost);
                    dseen.insert(newchash);
                }
            });
        })

        // non-parallel implementation:

        // for (pattern, c2) in tra_patterns.iter() {
        //     for rewrite in pattern_rewriter(pattern.clone(), &hc.0, |_| (c2.clone(), 0.0)) {
        //         // TODO here is where a optimal data-sharing copy would be handy
        //         let mut newc = hc.0.clone();
        //         newc.apply_rewrite(rewrite).expect("rewrite failure");
        //         let newchash = circuit_hash(&newc);
        //         if dseen.contains(&newchash) {
        //             continue;
        //         }
        //         let newhc = HashCirc(newc);
        //         let newcost = _rev_cost(&newhc);
        //         if gamma * (newcost as f64) > (cbest_cost as f64) {
        //             pq.push(newhc, newcost);
        //             dseen.insert(newchash);
        //         }
        //     }
        // }
    }
    cbest.0
}

fn op_hash(op: &Op) -> Option<usize> {
    Some(match op {
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
        _ => return None,
    })
}

fn circuit_hash(circ: &Circuit) -> usize {
    // adapted from Quartz (Apache 2.0)
    // https://github.com/quantum-compiler/quartz/blob/2e13eb7ffb3c5c5fe96cf5b4246f4fd7512e111e/src/quartz/tasograph/tasograph.cpp#L410
    let mut total = 0;

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
        total += myhash;
    }

    total
}

#[derive(Clone)]
struct HashCirc(pub Circuit);
impl PartialEq for HashCirc {
    fn eq(&self, other: &Self) -> bool {
        circuit_hash(&self.0) == circuit_hash(&other.0)
    }
}

impl Eq for HashCirc {}
impl std::hash::Hash for HashCirc {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        circuit_hash(&self.0).hash(state);
    }
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

    // TODO test that takes a list of circuits (some of them very related to
    // each other but distinct) and checks for no hash collisions

    #[test]
    fn test_taso_small() {
        // Figure 6 from Quartz paper https://arxiv.org/pdf/2204.09033.pdf

        let mut two_h = Circuit::with_uids(vec![UnitID::Qubit {
            reg_name: "q".into(),
            index: vec![0],
        }]);
        two_h.append_op(Op::H, &[0]).unwrap();
        two_h.append_op(Op::H, &[0]).unwrap();

        let mut oneqb_id = Circuit::with_uids(vec![UnitID::Qubit {
            reg_name: "q".into(),
            index: vec![0],
        }]);

        oneqb_id.append_op(Op::Noop(WireType::Qubit), &[0]).unwrap();

        let mut cx_left = Circuit::with_uids(vec![
            UnitID::Qubit {
                reg_name: "q".into(),
                index: vec![0],
            },
            UnitID::Qubit {
                reg_name: "q".into(),
                index: vec![1],
            },
        ]);
        cx_left.append_op(Op::CX, &[0, 1]).unwrap();
        cx_left.append_op(Op::H, &[0]).unwrap();
        cx_left.append_op(Op::H, &[1]).unwrap();

        let mut cx_right = Circuit::with_uids(vec![
            UnitID::Qubit {
                reg_name: "q".into(),
                index: vec![0],
            },
            UnitID::Qubit {
                reg_name: "q".into(),
                index: vec![1],
            },
        ]);
        cx_right.append_op(Op::H, &[0]).unwrap();
        cx_right.append_op(Op::H, &[1]).unwrap();
        cx_right.append_op(Op::CX, &[1, 0]).unwrap();

        let four_qbs: Vec<UnitID> = (0..4)
            .map(|i| UnitID::Qubit {
                reg_name: "q".into(),
                index: vec![i],
            })
            .collect();
        let mut circ = Circuit::with_uids(four_qbs.clone());
        circ.append_op(Op::H, &[1]).unwrap();
        circ.append_op(Op::H, &[2]).unwrap();
        circ.append_op(Op::CX, &[2, 3]).unwrap();
        circ.append_op(Op::H, &[3]).unwrap();
        circ.append_op(Op::CX, &[1, 2]).unwrap();
        circ.append_op(Op::H, &[2]).unwrap();
        circ.append_op(Op::CX, &[0, 1]).unwrap();
        circ.append_op(Op::H, &[1]).unwrap();
        circ.append_op(Op::H, &[0]).unwrap();

        let cout = taso(
            circ.clone(),
            vec![(two_h, oneqb_id), (cx_left, cx_right)],
            1.2,
            Circuit::node_count,
            10,
        );
        let cout = cout.remove_noop();

        let mut correct = Circuit::with_uids(four_qbs);
        correct.append_op(Op::H, &[0]).unwrap();
        correct.append_op(Op::H, &[3]).unwrap();
        correct.append_op(Op::CX, &[3, 2]).unwrap();
        correct.append_op(Op::CX, &[2, 1]).unwrap();
        correct.append_op(Op::CX, &[1, 0]).unwrap();

        assert_ne!(circuit_hash(&circ), circuit_hash(&cout));
        assert_eq!(circuit_hash(&correct), circuit_hash(&cout));
    }
}
