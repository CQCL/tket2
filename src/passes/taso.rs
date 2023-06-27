use hugr::{
    ops::{LeafOp, OpType},
    Hugr,
};
// use super::{pattern::node_equality, CircFixedStructPattern, PatternRewriter, RewriteGenerator};
// use crate::circuit::{
//     circuit::{Circuit, CircuitRewrite},
//     operation::Op,
// };
use portgraph::{
    graph::{Direction, NodeIndex},
    toposort::TopSortWalker,
};
use priority_queue::PriorityQueue;
use std::sync::mpsc;
use std::thread;
use std::{collections::HashMap, sync::Arc};

mod qtz_circuit;

#[derive(Clone)]
pub struct RepCircSet {
    rep_circ: Hugr,
    others: Vec<Hugr>,
}

// TODO refactor so both implementations share more code

pub fn rep_sets_from_path(path: &str) -> Vec<RepCircSet> {
    let all_circs = qtz_circuit::load_ecc_set(path);

    all_circs
        .into_values()
        .map(|mut all| {
            // TODO is the rep circ always the first??
            let rep_circ = all.remove(0);

            RepCircSet {
                rep_circ,
                others: all,
            }
        })
        .collect()
}

// impl RepCircSet {
//     // remove blank wires (up to an optional target or all) and report how many there were
//     fn remove_blanks(circ: &mut Circuit, target: Option<usize>) -> usize {
//         let mut blankedges: Vec<_> = circ
//             .node_edges(circ.boundary()[0], Direction::Outgoing)
//             .into_iter()
//             .filter(|e| circ.edge_endpoints(*e).unwrap().1 == circ.boundary()[1])
//             .collect();
//         let nblank = blankedges.len();
//         if let Some(target) = target {
//             assert!(nblank >= target, "not enough blank wires to reach target.");
//             blankedges.drain(target..).for_each(drop);
//         }
//         for e in blankedges {
//             circ.remove_edge(e);
//         }

//         nblank
//     }

//     fn to_rewrites<'s, 'a: 's>(
//         &'s self,
//         base_circ: &'a Circuit,
//     ) -> impl Iterator<Item = CircuitRewrite> + 's {
//         /*
//         generates rewrites from all circuits in set to representative circuit
//         and reverse rewrites (chained)

//         assumes all circuits in set have same boundary

//         for pattern matching, removes blank wires (Input -> Output edges) in pattern
//         and tries to remove the *same number* of blank wires from the
//         replacement

//         (Therefore replacing blank wires with non-blank wires is not a supported operation,
//         but the opposite is)
//         */
//         let mut rep = self.rep_circ.clone();
//         let rep_blanks = Self::remove_blanks(&mut rep, None);
//         let patterns = self.others.iter().map(|c2| {
//             let mut c2 = c2.clone();
//             let blanks = Self::remove_blanks(&mut c2, None);
//             (
//                 CircFixedStructPattern::from_circ(c2, node_equality()),
//                 &self.rep_circ,
//                 blanks,
//             )
//         });

//         let patterns = patterns.chain(self.others.iter().map(move |c2| {
//             (
//                 CircFixedStructPattern::from_circ(rep.clone(), node_equality()),
//                 c2,
//                 rep_blanks,
//             )
//         }));
//         patterns.flat_map(|(pattern, c2, blanks)| {
//             PatternRewriter::new(pattern, move |_| {
//                 let mut replacement = c2.clone();
//                 Self::remove_blanks(&mut replacement, Some(blanks));
//                 (replacement, 0.0)
//             })
//             .into_rewrites(base_circ)
//             // pattern_rewriter(pattern, base_circ, )
//         })
//     }
// }

// pub fn taso_mpsc<C>(
//     circ: Circuit,
//     repset: Vec<RepCircSet>,
//     gamma: f64,
//     cost: C,
//     _timeout: i64,
//     max_threads: usize,
// ) -> Circuit
// where
//     C: Fn(&Circuit) -> usize + Send + Sync,
// {
//     // bound the number of threads, chunk up the patterns in to each thread
//     let n_threads = std::cmp::min(max_threads, repset.len());

//     let (t_main, r_main) = mpsc::channel();

//     let mut chunks: Vec<Vec<_>> = (0..n_threads).map(|_| vec![]).collect();

//     for (count, p) in repset.into_iter().enumerate() {
//         chunks[count % n_threads].push((count, p));
//     }

//     let _rev_cost = |x: &Circuit| usize::MAX - cost(x);

//     let mut pq = PriorityQueue::new();
//     let mut cbest = circ.clone();
//     let cin_cost = _rev_cost(&circ);
//     let mut cbest_cost = cin_cost;
//     let chash = circuit_hash(&circ);
//     // map of seen circuits, if the circuit has been popped from the queue,
//     // holds None
//     let mut dseen: HashMap<usize, Option<Circuit>> = HashMap::from_iter([(chash, Some(circ))]);
//     pq.push(chash, cin_cost);

//     // each thread scans for rewrites using all the patterns in its chunk, and
//     // sends rewritten circuits back to main
//     let (joins, thread_ts): (Vec<_>, Vec<_>) = chunks
//         .into_iter()
//         .enumerate()
//         .map(|(i, repsets)| {
//             // channel for sending circuits to each thread
//             let (t_this, r_this) = mpsc::channel();
//             let tn = t_main.clone();
//             let jn = thread::spawn(move || {
//                 for received in r_this {
//                     let sent_circ: Arc<Circuit> = if let Some(hc) = received {
//                         hc
//                     } else {
//                         // main has signalled no more circuits will be sent
//                         return;
//                     };
//                     println!("thread {i} got one");
//                     for (set_i, rcs) in &repsets {
//                         for rewrite in rcs.to_rewrites(&sent_circ) {
//                             // TODO here is where a optimal data-sharing copy would be handy
//                             let mut newc = sent_circ.as_ref().clone();
//                             newc.apply_rewrite(rewrite).expect("rewrite failure");

//                             tn.send(Some(newc)).unwrap();
//                             println!("thread {i} sent one back from set {set_i}");
//                         }
//                     }
//                     // no more circuits will be generated, tell main this thread is
//                     // done with this circuit
//                     tn.send(None).unwrap();
//                 }
//             });

//             (jn, t_this)
//         })
//         .unzip();

//     while let Some((hc, priority)) = pq.pop() {
//         let seen_circ = dseen
//             .insert(hc, None)
//             .flatten()
//             .expect("seen circ missing.");
//         println!("\npopped one of size {}", &seen_circ.node_count());

//         if priority > cbest_cost {
//             cbest = seen_circ.clone();
//             cbest_cost = priority;
//         }
//         let seen_circ = Arc::new(seen_circ);
//         // send the popped circuit to each thread
//         for thread_t in &thread_ts {
//             thread_t.send(Some(seen_circ.clone())).unwrap();
//         }
//         let mut done_tracker = 0;
//         for received in &r_main {
//             let newc = if let Some(newc) = received {
//                 newc
//             } else {
//                 done_tracker += 1;
//                 if done_tracker == n_threads {
//                     // all threads have said they are done with this circuit
//                     break;
//                 } else {
//                     continue;
//                 }
//             };
//             println!("Main got one");
//             let newchash = circuit_hash(&newc);
//             if dseen.contains_key(&newchash) {
//                 continue;
//             }
//             let newcost = _rev_cost(&newc);
//             if gamma * (newcost as f64) > (cbest_cost as f64) {
//                 pq.push(newchash, newcost);
//                 dseen.insert(newchash, Some(newc));
//             }
//         }
//     }
//     for (join, tx) in joins.into_iter().zip(thread_ts.into_iter()) {
//         // tell all the threads we're done and join the threads
//         tx.send(None).unwrap();
//         join.join().unwrap();
//     }
//     cbest
// }

// pub fn taso<C>(
//     circ: Circuit,
//     repset: Vec<RepCircSet>,
//     gamma: f64,
//     cost: C,
//     _timeout: i64,
// ) -> Circuit
// where
//     C: Fn(&Circuit) -> usize + Send + Sync,
// {
//     // TODO timeout

//     let _rev_cost = |x: &Circuit| usize::MAX - cost(x);

//     let mut pq = PriorityQueue::new();
//     let mut cbest = circ.clone();
//     let cin_cost = _rev_cost(&circ);
//     let mut cbest_cost = cin_cost;
//     let chash = circuit_hash(&circ);
//     // map of seen circuits, if the circuit has been popped from the queue,
//     // holds None
//     let mut dseen: HashMap<usize, Option<Circuit>> = HashMap::from_iter([(chash, Some(circ))]);
//     pq.push(chash, cin_cost);

//     while let Some((hc, priority)) = pq.pop() {
//         // remove circuit from map and replace with None
//         let seen_circ = dseen
//             .insert(hc, None)
//             .flatten()
//             .expect("seen circ missing.");
//         if priority > cbest_cost {
//             cbest = seen_circ.clone();
//             cbest_cost = priority;
//         }
//         // par_iter implementation

//         // let pq = Mutex::new(&mut pq);
//         // tra_patterns.par_iter().for_each(|(pattern, c2)| {
//         //     pattern_rewriter(pattern.clone(), &hc.0, |_| (c2.clone(), 0.0)).for_each(|rewrite| {
//         //         let mut newc = hc.0.clone();
//         //         newc.apply_rewrite(rewrite).expect("rewrite failure");
//         //         let newchash = circuit_hash(&newc);
//         //         let mut dseen = dseen.lock().unwrap();
//         //         if dseen.contains(&newchash) {
//         //             return;
//         //         }
//         //         let newhc = HashCirc(newc);
//         //         let newcost = _rev_cost(&newhc);
//         //         if gamma * (newcost as f64) > (cbest_cost as f64) {
//         //             let mut pq = pq.lock().unwrap();
//         //             pq.push(newhc, newcost);
//         //             dseen.insert(newchash);
//         //         }
//         //     });
//         // })

//         // non-parallel implementation:

//         for rp in repset.iter() {
//             for rewrite in rp.to_rewrites(&seen_circ) {
//                 // TODO here is where a optimal data-sharing copy would be handy
//                 let mut newc = seen_circ.clone();
//                 newc.apply_rewrite(rewrite).expect("rewrite failure");
//                 let newchash = circuit_hash(&newc);
//                 if dseen.contains_key(&newchash) {
//                     continue;
//                 }
//                 let newcost = _rev_cost(&newc);
//                 if gamma * (newcost as f64) > (cbest_cost as f64) {
//                     pq.push(newchash, newcost);
//                     dseen.insert(newchash, Some(newc));
//                 }
//             }
//         }
//     }
//     cbest
// }

// fn op_hash(op: &LeafOp) -> Option<usize> {
//     type Op = LeafOp;
//     Some(match op {
//         Op::H => 1,
//         Op::CX => 2,
//         Op::ZZMax => 3,
//         Op::Reset => 4,
//         Op::Input => 5,
//         Op::Output => 6,
//         Op::Noop(_) => 7,
//         Op::Measure => 8,
//         Op::Barrier => 9,
//         Op::AngleAdd => 10,
//         Op::AngleMul => 11,
//         Op::AngleNeg => 12,
//         Op::QuatMul => 13,
//         // Op::Copy { n_copies, typ } => todo!(),
//         Op::RxF64 => 14,
//         Op::RzF64 => 15,
//         Op::TK1 => 16,
//         Op::Rotation => 17,
//         Op::ToRotation => 18,
//         // should Const of different values be hash different?
//         Op::Const(_) => 19,
//         // Op::Custom(_) => todo!(),
//         _ => return None,
//     })
// }

// fn circuit_hash(circ: &Circuit) -> usize {
//     // adapted from Quartz (Apache 2.0)
//     // https://github.com/quantum-compiler/quartz/blob/2e13eb7ffb3c5c5fe96cf5b4246f4fd7512e111e/src/quartz/tasograph/tasograph.cpp#L410
//     let mut total: usize = 0;

//     let mut hash_vals: HashMap<NodeIndex, usize> = HashMap::new();
//     let [i, _] = circ.boundary();

//     let _ophash = |o| 17 * 13 + op_hash(o).expect("unhashable op");
//     hash_vals.insert(i, _ophash(&Op::Input));

//     let initial_nodes = circ
//         .dag
//         .node_indices()
//         .filter(|n| circ.dag.node_edges(*n, Direction::Incoming).count() == 0)
//         .collect();

//     for nid in TopSortWalker::new(circ.dag_ref(), initial_nodes) {
//         if hash_vals.contains_key(&nid) {
//             continue;
//         }

//         let mut myhash = _ophash(circ.node_op(nid).expect("op not found."));

//         for ine in circ.node_edges(nid, Direction::Incoming) {
//             let (src, _) = circ.edge_endpoints(ine).expect("edge not found.");
//             debug_assert!(hash_vals.contains_key(&src));

//             let mut edgehash = hash_vals[&src];

//             // TODO check if overflow arithmetic is intended

//             edgehash = edgehash.wrapping_mul(31).wrapping_add(
//                 circ.port_of_edge(src, ine, Direction::Outgoing)
//                     .expect("edge not found."),
//             );
//             edgehash = edgehash.wrapping_mul(31).wrapping_add(
//                 circ.port_of_edge(nid, ine, Direction::Incoming)
//                     .expect("edge not found."),
//             );

//             myhash = myhash.wrapping_add(edgehash);
//         }
//         hash_vals.insert(nid, myhash);
//         total = total.wrapping_add(myhash);
//     }

//     total
// }

// #[cfg(test)]
// mod tests;
