#![allow(dead_code)]

use crate::circuit::circuit::{Circuit, CircuitRewrite};
use crate::circuit::dag::Edge;
use crate::circuit::operation::Op;
use crate::utils::n_qbs;
use petgraph::algo::{floyd_warshall, NegativeCycle};
use petgraph::graph::{NodeIndex, UnGraph};
use portgraph::substitute::{BoundedSubgraph, SubgraphRef};
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct QAddr(pub u32);

type Architecture = UnGraph<(), ()>;
type Distances = HashMap<(NodeIndex, NodeIndex), u32>;
type Mapping = bimap::BiMap<Edge, QAddr>;

fn distances(arc: &Architecture) -> Result<Distances, NegativeCycle> {
    floyd_warshall(arc, |_| 1)
}

struct NodeState {
    circ: Circuit,
    frontier: Vec<Edge>,
    mapping: Mapping,
}

fn swap_circ() -> Circuit {
    let mut swapc = Circuit::with_uids(n_qbs(2));
    swapc.append_op(Op::Swap, &[1, 0]).unwrap();
    swapc
}
impl NodeState {
    fn all_rewrites(&self, arc: &Architecture) -> Vec<CircuitRewrite> {
        let frontier_physicals: HashSet<_> = self
            .frontier
            .iter()
            .filter_map(|e| {
                let target = self.circ.edge_endpoints(*e).unwrap().1;
                self.circ
                    .node_op(target)
                    .unwrap()
                    .is_two_qb_gate()
                    .then(|| self.mapping.get_by_left(e))
                    .flatten()
                    .map(|qad| (e, qad))
            })
            .collect();
        let mut e_pairs = HashSet::new();
        for (e, qddr) in frontier_physicals {
            let neighs = arc.neighbors(qddr.0.into());

            for n in neighs {
                // TODO physical qubits not already in mapping are not considered
                if let Some(other_e) = self.mapping.get_by_right(&QAddr(n.index() as u32)) {
                    let mut es = [*e, *other_e];
                    es.sort();

                    e_pairs.insert(es);
                }
            }
        }
        e_pairs
            .into_iter()
            .map(|es| {
                CircuitRewrite::new(
                    BoundedSubgraph::new(SubgraphRef::new(HashSet::new()), [es.into(), es.into()]),
                    swap_circ().into(),
                    0.0,
                )
            })
            .collect()

        // TODO bridges and more
    }
}

enum NodeStateEnum {
    Root(Box<NodeState>),
    Child(Box<(CircuitRewrite, Option<NodeState>)>),
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
    fn node(&self, n: NodeIndex) -> &MCTNode {
        self.graph.node_weight(n).expect("Node not found")
    }
    fn node_mut(&mut self, n: NodeIndex) -> &mut MCTNode {
        self.graph.node_weight_mut(n).expect("Node not found")
    }

    fn is_leaf(&self, n: NodeIndex) -> bool {
        self.graph.edges(n).next().is_none()
    }
    fn select(&mut self) -> NodeIndex {
        // let mut path = vec![];
        let mut s = self.root;

        self.node_mut(s).visits += 1;
        while self.is_leaf(s) {
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
}
#[cfg(test)]
mod tests {
    use portgraph::{
        graph::Direction,
        substitute::{BoundedSubgraph, SubgraphRef},
    };

    use crate::{utils::n_qbs, validate::check_soundness};

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

    #[test]
    fn test_swap_gen() {
        let mut circ = Circuit::with_uids(n_qbs(3));
        circ.append_op(Op::CX, &[0, 1]).unwrap();
        circ.append_op(Op::CX, &[1, 2]).unwrap();

        let frontier = circ.node_edges(circ.boundary()[0], Direction::Outgoing);
        let mapping = Mapping::from_iter([
            (frontier[0], QAddr(0)),
            (frontier[1], QAddr(2)),
            (frontier[2], QAddr(3)),
        ]);
        let ns = NodeState {
            circ: circ.clone(),
            frontier,
            mapping,
        };
        let mut rewrites = ns.all_rewrites(&simple_arc());

        // only 0, 3 can be swapped because 2 is disconnected
        assert_eq!(rewrites.len(), 1);

        let mut newc = circ;
        newc.apply_rewrite(rewrites.remove(0)).unwrap();
        check_soundness(&newc).unwrap();
    }
}
