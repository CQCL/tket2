#![allow(dead_code)]

use crate::circuit::circuit::{Circuit, CircuitRewrite};
use crate::circuit::dag::{Edge, Vertex};
use crate::circuit::operation::{Op, WireType};
use petgraph::algo::{floyd_warshall, NegativeCycle};
use petgraph::graph::{NodeIndex, UnGraph};
use portgraph::substitute::{BoundedSubgraph, SubgraphRef};
use std::collections::{HashMap, HashSet};

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
        advance_frontier(&self.circ, arc, &mut mapping);

        Self { circ, mapping }
    }
}

fn next_edge(circ: &Circuit, n: Vertex, e: Edge) -> Edge {
    let port = circ
        .port_of_edge(n, e, portgraph::graph::Direction::Incoming)
        .unwrap();
    circ.edge_at_port(n, port, portgraph::graph::Direction::Outgoing)
        .expect("Node doesn't have edge at equivalent port.")
}

fn move_update(circ: &Circuit, mut mapping: Mapping, mve: &Move, swap_inserted: bool) -> Mapping {
    let Move::Swap(qs) = mve;

    let es = qs.map(|q| mapping.remove_by_right(&q).expect("missing in map").0);

    let outes = if swap_inserted {
        let (_, swap_node) = circ.edge_endpoints(es[0]).expect("edge not in circuit");
        es.map(|e| next_edge(circ, swap_node, e))
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

type MCTree = petgraph::graph::DiGraph<MCTNode, ()>;

struct Mcts {
    graph: MCTree,
    root: NodeIndex,
    arc: Architecture,
    distances: Distances,
    visit_weight: f64,
}

fn select_criteria(b: &MCTNode, parent_visits: u64, visit_weight: f64) -> f64 {
    b.reward + b.val + visit_weight * ((parent_visits as f64).ln() / (b.visits as f64)).sqrt()
}
impl Mcts {
    fn new(circ: Circuit, arc: Architecture, visit_weight: f64) -> Self {
        let distances = distances(&arc).expect("distance calculation failed.");
        // let's use unit ids to get the initial mapping
        let mapping = mapping_from_circ(&circ);
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
        }
    }

    fn node(&self, n: NodeIndex) -> &MCTNode {
        self.graph.node_weight(n).expect("Node not found")
    }
    fn node_mut(&mut self, n: NodeIndex) -> &mut MCTNode {
        self.graph.node_weight_mut(n).expect("Node not found")
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
}

fn mapping_from_circ(circ: &Circuit) -> bimap::BiHashMap<portgraph::graph::EdgeIndex, QAddr> {
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

        let subg = BoundedSubgraph::new(SubgraphRef::new(HashSet::new()), [edges.clone(), edges]);

        let rw = CircuitRewrite::new(subg, swap_circ().into(), 0.0);

        c.apply_rewrite(rw).unwrap();
        assert_eq!(c.node_count(), 3);
        check_soundness(&c).unwrap();
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
        let circ = simple_circ();

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
    fn test_trivial_expand() {
        let mut mcts = simple_mcts();

        let s = mcts.select();

        let expanded = mcts.expand(s);
        assert_eq!(expanded.len(), 2);
    }
}
