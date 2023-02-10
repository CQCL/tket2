use crate::circuit::circuit::{Circuit, UnitID};
use petgraph::graph::NodeIndex;
use std::collections::{HashSet, VecDeque};

use super::nodestate::{NodeState, NodeStateEnum};
use super::{distances, Architecture, Distances, Mapping, QAddr};

pub(super) struct MCTNode {
    pub(super) state: NodeStateEnum,
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

pub(super) struct Mcts {
    graph: MCTree,
    root: NodeIndex,
    arc: Architecture,
    distances: Distances,
    visit_weight: f64,
    simulate_layers: usize,
    discount: f64,
    num_backprop: u32,
}

fn select_criteria(b: &MCTNode, parent_visits: u64, visit_weight: f64) -> f64 {
    b.score() + visit_weight * ((parent_visits as f64).ln() / (b.visits as f64)).sqrt()
}

fn uid_to_qddr(uid: &crate::circuit::circuit::UnitID) -> QAddr {
    match uid {
        UnitID::Qubit { index, .. } => QAddr(index[0]),
        _ => panic!("gotta be a qubit for mapping."),
    }
}

fn mapping_from_circ(circ: &Circuit) -> Mapping {
    circ.unitids()
        .zip(
            circ.node_edges(circ.boundary()[0], portgraph::graph::Direction::Outgoing)
                .into_iter(),
        )
        .filter_map(|(uid, in_e)| {
            matches!(uid, UnitID::Qubit { .. }).then(|| (in_e, uid_to_qddr(uid)))
        })
        .collect()
}

impl Mcts {
    pub(super) fn new(circ: Circuit, arc: Architecture, visit_weight: f64) -> Self {
        let distances = distances(&arc).expect("distance calculation failed.");
        // let's use unit ids to get the initial mapping
        let mapping = mapping_from_circ(&circ);
        let mut graph = MCTree::new();
        let root_state = NodeStateEnum::root_state(circ, &arc, mapping);
        let root_node = MCTNode {
            state: root_state,
            visits: 0,
            reward: 0.0,
            val: 0.0,
        };
        let root = graph.add_node(root_node);
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
    pub(super) fn select(&mut self) -> NodeIndex {
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
    }

    fn get_state(&self, n: NodeIndex) -> Option<&NodeState> {
        match &self.node(n).state {
            NodeStateEnum::Root(ns) | NodeStateEnum::Child(_, Some(ns)) => Some(ns),
            _ => None,
        }
    }

    pub(super) fn expand(&mut self, s: NodeIndex) -> Vec<NodeIndex> {
        // hack, avoids having to deal with a mut reference
        self.set_state(s);

        let state = self
            .get_state(s)
            .expect("state should be set by this point.");
        let child_states: Vec<_> = state
            .all_moves(&self.arc)
            .map(|(mve, reward)| {
                // let mut mapping = move_update(&state.circ, state.mapping.clone(), &mve, false);
                // let reward = advance_frontier(&state.circ, &self.arc, &mut mapping) as f64;

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

    pub(super) fn simulate(&mut self, s: NodeIndex) {
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

    pub(super) fn backpropagate(&mut self, mut s: NodeIndex) {
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

    pub(super) fn decide(&mut self) {
        let best_child = self
            .graph
            .neighbors(self.root)
            .max_by(|a, b| compare_f64(&self.node(*a).score(), &self.node(*b).score()));
        if let Some(best_child) = best_child {
            self.set_state(best_child);
            let childn = self.node_mut(best_child);
            childn.state.make_root();

            let best_child = keep_branch(&mut self.graph, best_child);
            self.root = best_child;
        }
    }

    pub(super) fn solve(&mut self) -> MCTNode {
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

fn compare_f64(a: &f64, b: &f64) -> std::cmp::Ordering {
    a.partial_cmp(b).unwrap()
}

// removes all nodes except the tree with keep_branch_root as root and returns
// the weight of the root
fn keep_branch<T>(
    tree: &mut petgraph::graph::DiGraph<T, ()>,
    keep_branch_root: NodeIndex,
) -> NodeIndex {
    let mut queue = VecDeque::new();

    queue.push_back(keep_branch_root);

    // TODO extra allocation could be avoided with stablegraph - benchmark
    let mut keep_nodes = HashSet::new();
    while let Some(next) = queue.pop_front() {
        queue.extend(tree.neighbors(next));
        keep_nodes.insert(next);
    }

    tree.retain_nodes(|_, n| keep_nodes.contains(&n));
    tree.shrink_to_fit();

    // find new root, should hopefully be near the start
    tree.node_indices()
        .find(|n| {
            tree.edges_directed(*n, petgraph::Direction::Incoming)
                .next()
                .is_none()
        })
        .unwrap()
}

#[cfg(test)]
mod tests {

    use crate::{
        circuit::operation::Op,
        passes::mcts::{check_mapped, tests::simple_arc, Move},
        utils::n_qbs,
        validate::check_soundness,
    };

    use super::*;

    fn simple_mcts() -> Mcts {
        let mut circ = Circuit::with_uids(n_qbs(3));
        circ.append_op(Op::CX, &[0, 2]).unwrap();
        circ.append_op(Op::ZZMax, &[1, 2]).unwrap();
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

        keep_branch(&mut g, child);
        assert_eq!(g.node_count(), 3);
    }

    #[test]
    fn test_simple_solve() {
        let mut mcts = simple_mcts();

        let res = mcts.solve();
        if let NodeStateEnum::Root(s) = res.state {
            let circ = s.circ;
            check_soundness(&circ).unwrap();
            check_mapped(&circ, &mcts.arc).unwrap();
        } else {
            panic!();
        }
    }

    #[test]
    fn test_calc_state() {
        let mcts = simple_mcts();
        let NodeState { circ, .. } = mcts
            .get_root_state()
            .child_state(Move::Swap([QAddr(1), QAddr(2)]), &mcts.arc);

        check_soundness(&circ).unwrap();
    }
}
