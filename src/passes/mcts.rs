#![allow(dead_code)]

use crate::circuit::circuit::{Circuit, CircuitRewrite};
use crate::circuit::dag::{Edge, Vertex};
use crate::circuit::operation::{Op, WireType};
use petgraph::algo::{floyd_warshall, NegativeCycle};
use petgraph::graph::{NodeIndex, UnGraph};
use portgraph::substitute::{BoundedSubgraph, SubgraphRef};
use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct QAddr(pub u32);

impl From<QAddr> for NodeIndex<u32> {
    fn from(val: QAddr) -> Self {
        NodeIndex::from(val.0)
    }
}

type Architecture = UnGraph<(), ()>;
type Distances = HashMap<(NodeIndex, NodeIndex), u32>;
type Mapping = bimap::BiMap<Edge, QAddr>;

fn distances(arc: &Architecture) -> Result<Distances, NegativeCycle> {
    floyd_warshall(arc, |_| 1)
}

#[derive(Clone)]
struct NodeState {
    circ: Circuit,
    // frontier: Vec<Edge>,
    mapping: Mapping,
}

#[derive(Clone, Debug)]
enum Move {
    Swap([QAddr; 2]),
}

fn swap_circ() -> Circuit {
    let mut swapc = Circuit::new();

    let inputs = [(); 2].map(|_| swapc.new_input(WireType::Qubit));
    let outputs = [(); 2].map(|_| swapc.new_output(WireType::Qubit));

    swapc.add_vertex_with_edges(
        Op::Swap,
        [inputs[1], inputs[0]].into(),
        [outputs[0], outputs[1]].into(),
    );

    swapc
}

impl NodeState {
    fn gen_rewrite(&self, mve: Move) -> CircuitRewrite {
        let Move::Swap(swap) = mve;

        let es = swap.map(|qddr| {
            *self
                .mapping
                .get_by_right(&qddr)
                .expect("addr not found in mapping")
        });

        CircuitRewrite::new(
            BoundedSubgraph::new(SubgraphRef::new(HashSet::new()), [es.into(), es.into()]),
            swap_circ().into(),
            0.0,
        )
    }

    fn all_moves(&self, arc: &Architecture) -> impl Iterator<Item = Move> {
        let frontier_physicals: HashSet<_> = self
            .mapping
            .iter()
            .filter_map(|(e, q)| {
                let target = self.circ.edge_endpoints(*e).unwrap().1;
                self.circ
                    .node_op(target)
                    .unwrap()
                    .is_two_qb_gate()
                    .then_some(*q)
                // .map(|qad| (e, qad))
            })
            .collect();
        let mut qddr_pairs = HashSet::new();
        for qddr in frontier_physicals {
            let neighs = arc.neighbors(qddr.0.into());

            for n in neighs {
                // TODO physical qubits not already in mapping are not
                // considered
                let n_addr = QAddr(n.index() as u32);
                if self.mapping.contains_right(&n_addr) {
                    let mut p = [qddr, n_addr];
                    p.sort();
                    qddr_pairs.insert(p);
                }
            }
        }
        qddr_pairs.into_iter().map(Move::Swap)
        // // TODO bridges and more
    }

    fn all_rewrites(&self, arc: &Architecture) -> impl Iterator<Item = CircuitRewrite> + '_ {
        self.all_moves(arc).map(|mve| self.gen_rewrite(mve))
        // .collect()
    }

    fn child_state(&self, mve: Move, arc: &Architecture) -> Self {
        let rw = self.gen_rewrite(mve.clone());
        let mut circ = self.circ.clone();
        circ.apply_rewrite(rw).expect("rewrite failure");
        let mut mapping = move_update(&circ, self.mapping.clone(), &mve, true);

        advance_frontier(&circ, arc, &mut mapping);

        Self { circ, mapping }
    }

    fn sim_heuristic(
        &self,
        mve: &Move,
        dists: &Distances,
        n_layers: usize,
        layer_discount: f64,
    ) -> f64 {
        let mut total = 0.0;
        let mut discount = layer_discount;

        let layiter = TwoqbLayerIter {
            frontier: self.mapping.left_values().cloned().collect(),
            circ: &self.circ,
        };
        let mut mapping = self.mapping.clone();

        for layer in layiter.take(n_layers) {
            total +=
                (layer_dist_change(&self.circ, &layer, &mapping, dists, mve) as f64) * discount;
            discount *= layer_discount;
            mapping = update_layer_mapping(&self.circ, &layer, mapping);
        }
        total
    }

    fn solved(&self) -> bool {
        self.mapping.left_values().all(|e| {
            matches!(
                self.circ
                    .node_op(self.circ.edge_endpoints(*e).unwrap().1)
                    .unwrap(),
                Op::Output
            )
        })
    }
}

fn next_edge(circ: &Circuit, n: Vertex, e: Edge) -> (Edge, Vertex) {
    let port = circ
        .port_of_edge(n, e, portgraph::graph::Direction::Incoming)
        .unwrap();
    let e = circ
        .edge_at_port(n, port, portgraph::graph::Direction::Outgoing)
        .expect("Node doesn't have edge at equivalent port.");

    let (_, tgt) = circ.edge_endpoints(e).unwrap();
    (e, tgt)
}

fn move_update(circ: &Circuit, mut mapping: Mapping, mve: &Move, swap_inserted: bool) -> Mapping {
    let Move::Swap(qs) = mve;

    let es = qs.map(|q| mapping.remove_by_right(&q).expect("missing in map").0);

    let outes = if swap_inserted {
        let (_, swap_node) = circ.edge_endpoints(es[0]).expect("edge not in circuit");
        es.map(|e| next_edge(circ, swap_node, e).0)
    } else {
        es
    };

    mapping.insert(outes[1], qs[0]);
    mapping.insert(outes[0], qs[1]);

    mapping
}

fn advance_frontier(circ: &Circuit, arc: &Architecture, mapping: &mut Mapping) -> u64 {
    let mut count = 0;

    loop {
        let mut e_updates: Vec<_> = vec![];
        let mut seen = HashSet::new();
        for (map_in, map_q) in mapping.iter() {
            let (_, tgt) = circ.edge_endpoints(*map_in).expect("edge not in circuit");
            if !seen.insert(tgt) {
                continue;
            }
            let op = circ.node_op(tgt).expect("node missing");
            let out_q_edges: Vec<_> = circ
                .node_edges(tgt, portgraph::graph::Direction::Outgoing)
                .into_iter()
                .filter(|e| circ.edge_type(*e) == Some(WireType::Qubit))
                .collect();
            if op.is_two_qb_gate() {
                let ins = circ.node_edges(tgt, portgraph::graph::Direction::Incoming);
                let e_other = ins.iter().find(|e| *e != map_in).unwrap();
                debug_assert!(ins.contains(map_in));
                let other_q = if let Some(q2) = mapping.get_by_left(e_other) {
                    q2
                } else {
                    // gate not yet at front
                    continue;
                };
                if arc.contains_edge((*map_q).into(), (*other_q).into()) {
                    for p in ins.into_iter().zip(out_q_edges.into_iter()) {
                        e_updates.push(p);
                    }
                    count += 1;
                }
            } else if out_q_edges.len() == 1 {
                e_updates.push((*map_in, out_q_edges.into_iter().next().unwrap()));
            }
        }
        if e_updates.is_empty() {
            return count;
        }
        for (e1, e2) in e_updates {
            let (_, q) = mapping
                .remove_by_left(&e1)
                .expect("edge missing in mapping");
            mapping.insert(e2, q);
        }
    }
}
enum NodeStateEnum {
    Root(Box<NodeState>),
    Child(Box<Move>, Option<Box<NodeState>>),
}

struct MCTNode {
    state: NodeStateEnum,
    visits: u64,
    reward: f64,
    val: f64,
}

impl MCTNode {
    fn score(&self) -> f64 {
        self.reward + self.val
    }
}

type MCTree = petgraph::graph::DiGraph<MCTNode, ()>;

struct Mcts {
    graph: MCTree,
    root: NodeIndex,
    arc: Architecture,
    distances: Distances,
    visit_weight: f64,
    simulate_layers: usize,
    discount: f64,
    num_backprop: u32,
}

struct TwoqbLayerIter<'c> {
    frontier: HashSet<Edge>,
    circ: &'c Circuit,
}

impl<'c> Iterator for TwoqbLayerIter<'c> {
    type Item = HashSet<Vertex>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut out = HashSet::new();

        let mut newfront: HashSet<_> = self
            .frontier
            .iter()
            .map(|e| {
                let mut e = *e;
                loop {
                    let (_, tgt) = self.circ.edge_endpoints(e).expect("edge missing");
                    let op = self.circ.node_op(tgt).expect("nodemissing");

                    // two qb gate with parameter inputs will fail here, compare
                    // edge types directly
                    if op.is_two_qb_gate() {
                        out.insert(tgt);
                        break e;
                    } else if matches!(op, Op::Output) {
                        break e;
                    } else {
                        e = next_edge(self.circ, tgt, e).0;
                    }
                }
            })
            .collect();

        let out: HashSet<_> = out
            .into_iter()
            .filter(|tgt| {
                let ins = self
                    .circ
                    .node_edges(*tgt, portgraph::graph::Direction::Incoming);
                ins.iter().all(|e| newfront.contains(e))
            })
            .collect();

        for tgt in &out {
            let outs = self
                .circ
                .node_edges(*tgt, portgraph::graph::Direction::Outgoing);

            let ins = self
                .circ
                .node_edges(*tgt, portgraph::graph::Direction::Incoming);
            for ine in ins {
                newfront.remove(&ine);
            }
            for oute in outs {
                newfront.insert(oute);
            }
        }

        self.frontier = newfront;
        (!out.is_empty()).then_some(out)
    }
}

fn layer_dist_change(
    circ: &Circuit,
    layer: &HashSet<Vertex>,
    mapping: &Mapping,
    dists: &Distances,
    mve: &Move,
) -> i64 {
    let Move::Swap(swap_qs) = mve;

    let relevant_qs: Vec<[QAddr; 2]> = layer
        .iter()
        .filter_map(|n| {
            let ins = circ.node_edges(*n, portgraph::graph::Direction::Incoming);
            let ins_qs: Vec<_> = ins
                .into_iter()
                .map(|e| mapping.get_by_left(&e).expect("Edge missing in map"))
                .cloned()
                .collect();

            ins_qs
                .iter()
                .any(|q| swap_qs.contains(q))
                .then_some((&ins_qs[..2]).try_into().expect("Should be two qubits."))
        })
        .collect();
    let dist =
        |q1: QAddr, q2: QAddr| *dists.get(&(q1.into(), q2.into())).expect("dist missing") as i64;
    let [sq1, sq2] = swap_qs;
    relevant_qs
        .into_iter()
        .map(|rel_qs| match rel_qs {
            // TODO make this nicer
            [lq, rq] if &lq == sq1 => dist(*sq1, rq) - dist(*sq2, rq),
            [lq, rq] if &rq == sq1 => dist(*sq1, lq) - dist(*sq2, lq),
            [lq, rq] if &lq == sq2 => dist(*sq2, rq) - dist(*sq1, lq),
            [lq, rq] if &rq == sq2 => dist(*sq2, lq) - dist(*sq1, lq),
            _ => panic!("all qubits should have something do with the swap."),
        })
        .sum()
}

fn update_layer_mapping(circ: &Circuit, layer: &HashSet<Vertex>, mut mapping: Mapping) -> Mapping {
    for tgt in layer {
        let ins = circ.node_edges(*tgt, portgraph::graph::Direction::Incoming);
        let out = circ.node_edges(*tgt, portgraph::graph::Direction::Outgoing);
        for (ine, oute) in ins.into_iter().zip(out.into_iter()) {
            let mut edge_tgt = circ.edge_endpoints(oute).unwrap().1;
            let mut edge = oute;
            while {
                let op = circ.node_op(edge_tgt).unwrap();
                !(op.is_two_qb_gate() || matches!(op, Op::Output))
            } {
                (edge, edge_tgt) = next_edge(circ, edge_tgt, edge);
            }
            let (_, q) = mapping.remove_by_left(&ine).expect("missing in map");
            mapping.insert(edge, q);
        }
        // for (i, o) in ins.into_iter().zip(out.into_iter()) {
        // }

        // if circ.node_op(tgt).expect("op missing").is_two_qb_gate() {
        //     break;
        // }
        // TODO fragile, expects only 1 and 2qb gates
        // tgt = circ.edge_endpoints(o1).expect("edge missing").1;
    }

    mapping
}

fn select_criteria(b: &MCTNode, parent_visits: u64, visit_weight: f64) -> f64 {
    b.score() + visit_weight * ((parent_visits as f64).ln() / (b.visits as f64)).sqrt()
}
impl Mcts {
    fn new(circ: Circuit, arc: Architecture, visit_weight: f64) -> Self {
        let distances = distances(&arc).expect("distance calculation failed.");
        // let's use unit ids to get the initial mapping
        let mut mapping = mapping_from_circ(&circ);
        advance_frontier(&circ, &arc, &mut mapping);
        let mut graph = MCTree::new();
        let root_state = MCTNode {
            state: NodeStateEnum::Root(Box::new(NodeState { circ, mapping })),
            visits: 0,
            reward: 0.0,
            val: 0.0,
        };
        let root = graph.add_node(root_state);
        Self {
            graph,
            root,
            arc,
            distances,
            visit_weight,
            simulate_layers: 4,
            discount: 0.7,
            num_backprop: 20,
        }
    }

    fn node(&self, n: NodeIndex) -> &MCTNode {
        self.graph.node_weight(n).expect("Node not found")
    }
    fn node_mut(&mut self, n: NodeIndex) -> &mut MCTNode {
        self.graph.node_weight_mut(n).expect("Node not found")
    }

    fn get_root_state(&self) -> &NodeState {
        match &self.node(self.root).state {
            NodeStateEnum::Root(x) => x,
            NodeStateEnum::Child(_, _) => panic!("not root!"),
        }
    }

    fn parent(&self, n: NodeIndex) -> NodeIndex {
        self.graph
            .neighbors_directed(n, petgraph::Direction::Incoming)
            .next()
            .expect("no parent.")
    }

    fn is_leaf(&self, n: NodeIndex) -> bool {
        self.graph.edges(n).next().is_none()
    }
    fn select(&mut self) -> NodeIndex {
        // let mut path = vec![];
        let mut s = self.root;

        self.node_mut(s).visits += 1;
        while !self.is_leaf(s) {
            let visits = self.node(s).visits;
            let weight = self.visit_weight;
            let best_child_index = self
                .graph
                .neighbors(s)
                .max_by(|a, b| {
                    let a = self.node(*a);
                    let b = self.node(*b);
                    select_criteria(a, visits, weight)
                        .partial_cmp(&select_criteria(b, visits, weight))
                        .unwrap()
                })
                .unwrap();

            self.node_mut(best_child_index).visits += 1;
            s = best_child_index;
        }
        s
    }

    fn set_state(&mut self, n: NodeIndex) {
        let mut calc_parent = false;
        if let NodeStateEnum::Child(_, nsopt) = &self.node(n).state {
            if nsopt.is_none() {
                calc_parent = true;
                // let mve = *mve.clone();
                // let parstate = self.get_or_set_state(self.parent(n));
                // // let mut c = parstate.circ.clone();
                // // c.apply_rewrite(mve).expect("Rewrite failed");

                // let ns = Box::new(NodeState {
                //     circ: c,
                //     frontier: todo!(),
                //     mapping: todo!(),
                // });
                // *nsopt = Some(ns);
            }
        };

        if calc_parent {
            let parent = self.parent(n);
            self.set_state(parent);
            let parstate = self.get_state(parent).unwrap();
            let mve = if let NodeStateEnum::Child(mve, _) = &self.node(n).state {
                *mve.clone()
            } else {
                panic!();
            };
            let child = parstate.child_state(mve, &self.arc);

            // // TODO make this caching optional
            if let NodeStateEnum::Child(_, nsopt) = &mut self.node_mut(n).state {
                *nsopt = Some(Box::new(child));
            }
        };

        // self.get_state(n).unwrap()

        // match &self.node(n).state {
        //     NodeStateEnum::Child(rw, nsopt) => {
        //         if let Some(ns) = nsopt {
        //             return ns;
        //         } else {
        //             let rw = *rw.clone();
        //             let parstate = self.get_state(self.parent(n));
        //             let mut c = parstate.circ.clone();
        //             c.apply_rewrite(rw).expect("Rewrite failed");

        //             let ns = Box::new(NodeState {
        //                 circ: c,
        //                 frontier: todo!(),
        //                 mapping: todo!(),
        //             });

        //             match &self.node_mut(n).state {
        //                 NodeStateEnum::Root(_) => unreachable!("should have returned above."),
        //                 NodeStateEnum::Child(_, nsopt) => {
        //                     *nsopt = Some(ns);

        //                     return &ns;
        //                 }
        //             }

        //         }
        //     }
        //     NodeStateEnum::Root(ns) => return ns,
        // };
    }

    fn get_state(&self, n: NodeIndex) -> Option<&NodeState> {
        match &self.node(n).state {
            NodeStateEnum::Root(ns) | NodeStateEnum::Child(_, Some(ns)) => Some(ns),
            _ => None,
        }
    }
    fn expand(&mut self, s: NodeIndex) -> Vec<NodeIndex> {
        // hack, avoids having to deal with a mut reference
        self.set_state(s);

        let state = self
            .get_state(s)
            .expect("state should be set by this point.");
        let child_states: Vec<_> = state
            .all_moves(&self.arc)
            .map(|mve| {
                let mut mapping = move_update(&state.circ, state.mapping.clone(), &mve, false);
                let reward = advance_frontier(&state.circ, &self.arc, &mut mapping) as f64;
                MCTNode {
                    state: NodeStateEnum::Child(Box::new(mve), None),
                    visits: 0,
                    val: 0.0,
                    reward,
                }
            })
            .collect();

        child_states
            .into_iter()
            .map(|child| {
                let child_node = self.graph.add_node(child);

                self.graph.add_edge(s, child_node, ());
                child_node
            })
            .collect()
    }

    fn simulate(&mut self, s: NodeIndex) {
        if s == self.root {
            return;
        }
        let parent = self.parent(s);

        self.set_state(parent);
        let parstate = self.get_state(parent).expect("pls work");

        let child_node = self.node(s);
        let mve = if let NodeStateEnum::Child(mve, _) = &child_node.state {
            mve
        } else {
            panic!("ill formed tree");
        };
        let val = parstate.sim_heuristic(mve, &self.distances, self.simulate_layers, self.discount);
        self.node_mut(s).val = val;
    }

    fn backpropagate(&mut self, mut s: NodeIndex) {
        while s != self.root {
            let parent = self.parent(s);
            let curr_n = self.node(s);
            let val = std::cmp::max_by(
                self.node(parent).val,
                self.discount * curr_n.score(),
                compare_f64,
            );
            self.node_mut(parent).val = val;
            s = parent;
        }
    }

    fn decide(&mut self) {
        let best_child = self
            .graph
            .neighbors(self.root)
            .max_by(|a, b| compare_f64(&self.node(*a).score(), &self.node(*b).score()));
        let best_child = best_child.expect("no children");
        self.set_state(best_child);
        let childn = self.node_mut(best_child);
        make_root(&mut childn.state);

        keep_branch(&mut self.graph, self.root, best_child);
        self.root = best_child;
    }

    fn solve(&mut self) -> MCTNode {
        while !self.get_root_state().solved() {
            for _ in 0..self.num_backprop {
                let s = self.select();
                self.expand(s);
                self.simulate(s);
                self.backpropagate(s);
            }
            self.decide();
        }
        self.graph.remove_node(self.root).unwrap()
    }
}

fn make_root(state_en: &mut NodeStateEnum) {
    match state_en {
        NodeStateEnum::Root(_) => (),
        NodeStateEnum::Child(_, x) if x.is_some() => {
            *state_en = NodeStateEnum::Root(x.take().unwrap())
        }
        _ => panic!("State must be calculated first."),
    };
}

fn compare_f64(a: &f64, b: &f64) -> std::cmp::Ordering {
    a.partial_cmp(b).unwrap()
}

// removes all nodes except the tree with keep_branch_root as root and returns
// the weight of the root
fn keep_branch<T>(
    tree: &mut petgraph::graph::DiGraph<T, ()>,
    root: NodeIndex,
    keep_branch_root: NodeIndex,
) -> T {
    let mut queue: VecDeque<_> = tree
        .neighbors(root)
        .filter(|n| *n != keep_branch_root)
        .collect();

    while let Some(next) = queue.pop_front() {
        queue.extend(tree.neighbors(next));

        tree.remove_node(next);
    }
    tree.remove_node(root).unwrap()
}

fn mapping_from_circ(circ: &Circuit) -> Mapping {
    circ.unitids()
        .zip(
            circ.node_edges(circ.boundary()[0], portgraph::graph::Direction::Outgoing)
                .into_iter(),
        )
        .map(|(uid, in_e)| (in_e, uid_to_qddr(uid)))
        .collect()
}

fn uid_to_qddr(uid: &crate::circuit::circuit::UnitID) -> QAddr {
    match uid {
        crate::circuit::circuit::UnitID::Qubit { reg_name, index } if reg_name == "q" => {
            QAddr(index[0])
        }
        _ => panic!("gotta be a 'q' qubit for mapping."),
    }
}
#[cfg(test)]
mod tests {
    use portgraph::{
        graph::Direction,
        substitute::{BoundedSubgraph, SubgraphRef},
    };

    use crate::{circuit::operation::Op, utils::n_qbs, validate::check_soundness};

    use super::*;
    fn insert_mirror_dists(dists: Distances) -> Distances {
        let mut out = HashMap::new();
        for ((i, j), d) in dists {
            if i <= j {
                out.insert((i, j), d);
                out.insert((j, i), d);
            }
        }
        out
    }
    fn simple_arc() -> Architecture {
        /*
        0 - 1 - 2
         \ / \ /
          3 - 4
        */
        Architecture::from_edges(&[(0, 1), (0, 3), (1, 2), (1, 3), (1, 4), (2, 4), (3, 4)])
    }
    #[test]
    fn test_distances() -> Result<(), NegativeCycle> {
        let g = simple_arc();

        let dists = distances(&g)?;

        let correct = HashMap::from_iter(
            [
                ((0, 1), 1),
                ((0, 2), 2),
                ((0, 3), 1),
                ((0, 4), 2),
                ((1, 2), 1),
                ((1, 3), 1),
                ((1, 4), 1),
                ((2, 4), 1),
                ((2, 3), 2),
                ((3, 4), 1),
                ((0, 0), 0),
                ((1, 1), 0),
                ((2, 2), 0),
                ((3, 3), 0),
                ((4, 4), 0),
            ]
            .map(|((i, j), d)| ((NodeIndex::new(i), NodeIndex::new(j)), d)),
        );

        let correct = insert_mirror_dists(correct);
        assert_eq!(dists, correct);
        Ok(())
    }

    #[test]
    fn test_insert_swap() {
        let mut c = Circuit::with_uids(n_qbs(2));
        let i = c.boundary()[0];
        let edges = c.node_edges(i, Direction::Outgoing);

        let subg = BoundedSubgraph::new(
            SubgraphRef::new(HashSet::new()),
            [edges.clone(), edges.clone()],
        );

        let rw = CircuitRewrite::new(subg, swap_circ().into(), 0.0);

        c.apply_rewrite(rw).unwrap();
        assert_eq!(c.node_count(), 3);
        check_soundness(&c).unwrap();
        // make sure incoming edges weren't deleted
        c.edge_endpoints(edges[0]).unwrap();
        c.edge_endpoints(edges[1]).unwrap();
    }

    fn simple_circ() -> Circuit {
        let mut circ = Circuit::with_uids(n_qbs(3));
        circ.append_op(Op::CX, &[0, 1]).unwrap();
        circ.append_op(Op::CX, &[1, 2]).unwrap();
        circ
    }

    #[test]
    fn test_swap_gen() {
        let circ = simple_circ();
        let frontier = circ.node_edges(circ.boundary()[0], Direction::Outgoing);
        let mapping = Mapping::from_iter([
            (frontier[0], QAddr(0)),
            (frontier[1], QAddr(2)),
            (frontier[2], QAddr(3)),
        ]);
        let ns = NodeState {
            circ: circ.clone(),
            mapping,
        };
        let mut swaps: Vec<_> = ns.all_moves(&simple_arc()).collect();

        // only 0, 3 can be swapped because 2 is disconnected
        assert_eq!(swaps.len(), 1);

        let mut newc = circ;
        newc.apply_rewrite(ns.gen_rewrite(swaps.remove(0))).unwrap();
        check_soundness(&newc).unwrap();
    }

    #[test]
    fn test_advance_frontier() {
        let mut circ = Circuit::with_uids(n_qbs(3));
        // TODO add RzF64 + ZZphase gate
        circ.append_op(Op::H, &[0]).unwrap();
        let cx01 = circ.append_op(Op::CX, &[0, 1]).unwrap();
        let cx12 = circ.append_op(Op::CX, &[1, 2]).unwrap();

        let frontier = circ.node_edges(circ.boundary()[0], Direction::Outgoing);
        let mut mapping = Mapping::from_iter([
            (frontier[0], QAddr(0)),
            (frontier[1], QAddr(3)),
            (frontier[2], QAddr(2)),
        ]);

        let arc = simple_arc();
        let gates_traversed = advance_frontier(&circ, &arc, &mut mapping);
        assert_eq!(gates_traversed, 1);

        let cx12in = circ.node_edges(cx12, Direction::Incoming);
        let cx01out = circ.node_edges(cx01, Direction::Outgoing);
        let correct_mapping = Mapping::from_iter([
            (cx01out[0], QAddr(0)),
            (cx12in[0], QAddr(3)),
            (frontier[2], QAddr(2)),
        ]);

        assert_eq!(correct_mapping, mapping);
    }

    fn simple_mcts() -> Mcts {
        let mut circ = Circuit::with_uids(n_qbs(3));
        circ.append_op(Op::CX, &[0, 2]).unwrap();
        circ.append_op(Op::CX, &[1, 2]).unwrap();
        let arc = simple_arc();

        Mcts::new(circ, arc, 0.8)
    }
    #[test]
    fn test_trivial_select() {
        let mut mcts = simple_mcts();

        let s = mcts.select();

        assert_eq!(s.index(), 0);
    }

    #[test]
    fn test_first_expand() {
        let mut mcts = simple_mcts();

        let s = mcts.select();

        let expanded = mcts.expand(s);
        assert_eq!(expanded.len(), 2);
    }

    #[test]
    fn test_twoqb_layers() {
        let mut circ = Circuit::with_uids(n_qbs(3));
        // TODO add RzF64 + ZZphase gate
        circ.append_op(Op::H, &[0]).unwrap();
        circ.append_op(Op::H, &[0]).unwrap();
        let cx1 = circ.append_op(Op::CX, &[0, 1]).unwrap();
        let cx2 = circ.append_op(Op::CX, &[1, 2]).unwrap();
        circ.append_op(Op::H, &[2]).unwrap();

        let frontier = circ.node_edges(circ.boundary()[0], Direction::Outgoing);
        let layers: Vec<_> = TwoqbLayerIter {
            frontier: frontier.into_iter().collect(),
            circ: &circ,
        }
        .collect();

        let cor_layers = vec![HashSet::from_iter([cx1]), HashSet::from_iter([cx2])];
        assert_eq!(layers, cor_layers);
    }

    #[test]
    fn test_dist_heuristic() {
        let mut circ = Circuit::with_uids(n_qbs(4));
        let cx1 = circ.append_op(Op::CX, &[1, 3]).unwrap();
        let cx2 = circ.append_op(Op::CX, &[2, 0]).unwrap();
        circ.append_op(Op::H, &[2]).unwrap();
        let arc = simple_arc();
        let dists = distances(&arc).unwrap();
        let frontier = circ.node_edges(circ.boundary()[0], Direction::Outgoing);
        let mapping = Mapping::from_iter([
            (frontier[0], QAddr(0)),
            (frontier[1], QAddr(3)),
            (frontier[2], QAddr(2)),
            (frontier[3], QAddr(1)),
        ]);

        let mve = Move::Swap([QAddr(0), QAddr(1)]);

        let layer = HashSet::from_iter([cx1, cx2]);

        let d_change = layer_dist_change(&circ, &layer, &mapping, &dists, &mve);

        assert_eq!(d_change, 1);

        let d_change = layer_dist_change(
            &circ,
            &layer,
            &mapping,
            &dists,
            &Move::Swap([QAddr(0), QAddr(3)]),
        );
        assert_eq!(d_change, 0);

        let d_change = layer_dist_change(
            &circ,
            &layer,
            &mapping,
            &dists,
            &Move::Swap([QAddr(4), QAddr(3)]),
        );
        assert_eq!(d_change, 0);
    }

    #[test]
    fn test_sim_heuristic() {
        let mut circ = Circuit::with_uids(n_qbs(4));
        circ.append_op(Op::CX, &[1, 3]).unwrap();

        circ.append_op(Op::CX, &[2, 0]).unwrap();
        circ.append_op(Op::CX, &[2, 1]).unwrap();
        circ.append_op(Op::H, &[0]).unwrap();
        circ.append_op(Op::CX, &[2, 0]).unwrap();
        circ.append_op(Op::H, &[2]).unwrap();

        circ.append_op(Op::CX, &[3, 1]).unwrap();
        circ.append_op(Op::CX, &[1, 3]).unwrap();

        let arc = simple_arc();
        let dists = distances(&arc).unwrap();
        let frontier = circ.node_edges(circ.boundary()[0], Direction::Outgoing);

        let mapping = Mapping::from_iter([
            (frontier[0], QAddr(0)),
            (frontier[1], QAddr(3)),
            (frontier[2], QAddr(2)),
            (frontier[3], QAddr(1)),
        ]);

        let ns = NodeState { circ, mapping };

        let val = ns.sim_heuristic(&Move::Swap([QAddr(2), QAddr(1)]), &dists, 3, 0.5);

        assert_eq!(val, 0.375);
    }

    #[test]
    fn test_calc_state() {
        let mcts = simple_mcts();
        let NodeState { circ, .. } = mcts
            .get_root_state()
            .child_state(Move::Swap([QAddr(1), QAddr(2)]), &mcts.arc);

        check_soundness(&circ).unwrap();
    }

    #[test]
    fn test_first_sim_backprop_decide() {
        let mut mcts = simple_mcts();

        let s = mcts.select();

        let expanded = mcts.expand(s);
        assert_eq!(expanded.len(), 2);
        let child = expanded[0];
        mcts.simulate(child);
        mcts.backpropagate(child);
        mcts.decide();
    }

    #[test]
    fn test_prune() {
        let mut g = petgraph::graph::DiGraph::<(), ()>::new();

        let root = g.add_node(());

        let mut child = root;

        for _ in 0..3 {
            let childx = g.add_node(());
            g.add_edge(root, childx, ());

            let childx1 = g.add_node(());
            g.add_edge(childx, childx1, ());

            let childx2 = g.add_node(());
            g.add_edge(childx, childx2, ());

            child = childx;
        }

        assert_eq!(g.node_count(), 10);

        keep_branch(&mut g, root, child);
        assert_eq!(g.node_count(), 3);
    }

    // #[test]
    // fn test_simple_solve() {
    //     let mut mcts = simple_mcts();

    //     let res = mcts.solve();
    // }
}
