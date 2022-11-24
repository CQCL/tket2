use std::{
    cmp::{self, Reverse},
    collections::{BTreeMap, HashMap},
    iter::Rev,
};

use portgraph::{
    graph::{Direction, NodeIndex},
    substitute::BoundedSubgraph,
};

use super::circuit::{Circuit, Command};

// Compute whether a BoundedSubgraph is convex.
//
// Optimised for many convexity queries on an immutable circuit. Precomputes the causal structure
// of the circuit so that individual convexity queries are very fast.
//
// This module exports ConvexMemoization, which can be used to make multiple convexity queries
// for a given circuit.
//
// Instances should be built using ConvexMemoization::memoize(&Circuit) and queries can be made
// using either ConvexMemoization::is_convex or ConvexMemoization::is_convex_subgraph.



fn cmp_key(dir: &Direction) -> impl Fn(&Option<NodeIndex>) -> OptionOrReverse<NodeIndex> + '_ {
    move |key| match dir {
        Direction::Incoming => (*key).into(),
        Direction::Outgoing => key.map(|ind| Reverse(ind)).into(),
    }
}

fn get_causal_dependencies<'a>(
    circ: &'a Circuit,
    cmds: &Vec<Command>,
    node2ind: &'a BTreeMap<NodeIndex, usize>,
    dir: Direction,
) -> HashMap<NodeIndex, Vec<Option<NodeIndex>>> {
    // In this function we deal with node indices that are topsorted usizes
    let uid2ind: HashMap<_, _> = circ
        .linear_unitids()
        .enumerate()
        .map(|(i, uid)| (uid.clone(), i))
        .collect();
    let n_lin_uids = uid2ind.len();
    let mut causal_deps = vec![vec![None; n_lin_uids]; cmds.len()];
    let cmds_it: ItOrRev<_> = match dir {
        Direction::Incoming => cmds.iter().enumerate().into(),
        Direction::Outgoing => cmds.iter().enumerate().rev().into(),
    };
    for (cmd_ind, cmd) in cmds_it {
        for nei in circ.neighbours(cmd.vertex, dir) {
            let nei_ind = node2ind[&nei];
            let (nei_causal_deps, curr_causal_deps) =
                borrow_two_mut(&mut causal_deps, nei_ind, cmd_ind)
                    .expect("neighbour is equal to self");
            // set the past to the max of its predecessors
            curr_causal_deps
                .iter_mut()
                .zip(nei_causal_deps.iter())
                .for_each(|(curr, nei)| *curr = cmp::max_by_key(*curr, *nei, cmp_key(&dir)));
            // overwrite the past with the last node where appropriate
            for qb_ind in cmds[nei_ind]
                .args
                .iter()
                .filter_map(|qb| uid2ind.get(qb).copied())
            {
                let curr = &mut curr_causal_deps[qb_ind];
                *curr = cmp::max_by_key(*curr, Some(nei), cmp_key(&dir));
            }
        }
    }
    causal_deps
        .into_iter()
        .enumerate()
        .map(|(i, n)| (cmds[i].vertex, n))
        .collect()
}

#[derive(Clone, PartialEq)]
pub enum ConvexMemoization<'a> {
    Uninitialised,
    Initialised {
        // For each node, the closest node in the past on each linear UnitID
        pred: HashMap<NodeIndex, Vec<Option<NodeIndex>>>,
        // For each node, the closest node in the future on each linear UnitID
        succ: HashMap<NodeIndex, Vec<Option<NodeIndex>>>,
        // A topsort of all nodes
        node2ind: BTreeMap<NodeIndex, usize>,
        // A reference to the circuit
        circ: &'a Circuit,
    },
}

#[derive(PartialEq, Eq, Hash, Clone, Debug)]
struct SortedNodeIndex<'a>(NodeIndex, &'a BTreeMap<NodeIndex, usize>);

impl<'a> From<SortedNodeIndex<'a>> for NodeIndex {
    fn from(ind: SortedNodeIndex<'a>) -> Self {
        ind.0
    }
}

impl<'a> PartialOrd for SortedNodeIndex<'a> {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<'a> Ord for SortedNodeIndex<'a> {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        assert_eq!(self.1, other.1);
        let node2ind = self.1;
        node2ind[&self.0].cmp(&node2ind[&other.0])
    }
}

impl<'a> ConvexMemoization<'a> {
    fn precompute(&mut self, circ: &'a Circuit) {
        if let Self::Uninitialised = *self {
            let cmds: Vec<_> = circ.to_commands().collect();
            let node2ind = cmds
                .iter()
                .enumerate()
                .map(|(i, cmd)| (cmd.vertex, i))
                .collect();
            let pred = get_causal_dependencies(circ, &cmds, &node2ind, Direction::Incoming);
            let succ = get_causal_dependencies(circ, &cmds, &node2ind, Direction::Outgoing);
            *self = Self::Initialised {
                pred,
                succ,
                node2ind,
                circ,
            };
        }
    }

    fn as_sorted_node_index(&self, n: NodeIndex) -> SortedNodeIndex<'_> {
        if let Self::Initialised {
            pred: _,
            succ: _,
            node2ind,
            circ: _,
        } = &*self
        {
            SortedNodeIndex(n, node2ind)
        } else {
            panic!("Must precompute")
        }
    }

    // For each linear UnitID, the closest node in the graph that is in the past
    fn get_preds<'b, I: Iterator<Item = &'b NodeIndex>>(
        &self,
        nodes: I,
    ) -> Vec<Option<SortedNodeIndex>> {
        self.get_closest(nodes, Direction::Incoming)
    }

    // For each linear UnitID, the closest node in the graph that is in the past
    fn get_succs<'b, I: Iterator<Item = &'b NodeIndex>>(
        &self,
        nodes: I,
    ) -> Vec<Option<SortedNodeIndex>> {
        self.get_closest(nodes, Direction::Outgoing)
    }

    // For each linear UnitID, the closest node in the graph that is in the past/future
    fn get_closest<'b, I: Iterator<Item = &'b NodeIndex>>(
        &self,
        nodes: I,
        dir: Direction,
    ) -> Vec<Option<SortedNodeIndex<'_>>> {
        if let Self::Initialised {
            pred,
            succ,
            node2ind: _,
            circ,
        } = &*self
        {
            let mut all_closest = vec![None; circ.linear_unitids().count()];
            let closest = match dir {
                Direction::Incoming => pred,
                Direction::Outgoing => succ,
            };
            for n in nodes {
                let closest_to_n = &closest[n];
                all_closest
                    .iter_mut()
                    .zip(closest_to_n.iter())
                    .for_each(|(curr, new)| {
                        *curr = cmp::max_by_key(*curr, *new, cmp_key(&dir));
                    });
            }
            all_closest
                .into_iter()
                .map(|o_n| o_n.map(|n| self.as_sorted_node_index(n)))
                .collect()
        } else {
            panic!("Must be precomputed")
        }
    }

    pub fn memoize(circ: &'a Circuit) -> Self {
        let mut s = Self::default();
        s.precompute(circ);
        s
    }

    pub fn is_convex(&self, nodes: Vec<NodeIndex>) -> bool {
        let preds = self.get_preds(nodes.iter());
        let succs = self.get_succs(nodes.iter());
        for (beg, end) in preds.iter().zip(succs.iter()) {
            if end.is_some() && beg >= end {
                // Any node that is in the past and future of some `nodes` must be contained
                let n_end = end.clone().unwrap().into();
                if !nodes.contains(&n_end) {
                    return false;
                }
            }
        }
        return true;
    }

    pub fn is_convex_subgraph(&self, subg: &BoundedSubgraph) -> bool {
        let nodes = subg.subgraph.nodes.iter();
        self.is_convex(nodes.cloned().collect())
    }
}

impl<'a> Default for ConvexMemoization<'a> {
    fn default() -> Self {
        Self::Uninitialised
    }
}

/////////////////// Some Utils structures /////////////////
///////////////////////////////////////////////////////////

// A union of it and its reverse
enum ItOrRev<I> {
    It1(I),
    It2(Rev<I>),
}

impl<I: DoubleEndedIterator> From<I> for ItOrRev<I> {
    fn from(it: I) -> Self {
        ItOrRev::It1(it)
    }
}

impl<I: DoubleEndedIterator> From<Rev<I>> for ItOrRev<I> {
    fn from(it: Rev<I>) -> Self {
        ItOrRev::It2(it)
    }
}

impl<I: DoubleEndedIterator> Iterator for ItOrRev<I> {
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            ItOrRev::It1(it) => it.next(),
            ItOrRev::It2(it) => it.next(),
        }
    }
}

// An option and its reverse
#[derive(PartialEq, Eq, PartialOrd, Ord)]
enum OptionOrReverse<I> {
    El(Option<I>),
    Rev(Option<Reverse<I>>),
}

impl<I> From<Option<I>> for OptionOrReverse<I> {
    fn from(el: Option<I>) -> Self {
        Self::El(el)
    }
}
impl<I> From<Option<Reverse<I>>> for OptionOrReverse<I> {
    fn from(rev_el: Option<Reverse<I>>) -> Self {
        Self::Rev(rev_el)
    }
}

fn borrow_two_mut<T>(a: &mut [T], mut i: usize, mut j: usize) -> Option<(&T, &mut T)> {
    let swap;
    if i == j {
        None
    } else {
        if i > j {
            (i, j) = (j, i);
            swap = true;
        } else {
            swap = false;
        }
        let (p1, p2) = a.split_at_mut(j);
        if swap {
            Some((&p2[0], &mut p1[i]))
        } else {
            Some((&p1[i], &mut p2[0]))
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::circuit::{
        circuit::{Circuit, UnitID},
        operation::Op,
    };

    use super::ConvexMemoization;

    fn make_3qb() -> (UnitID, UnitID, UnitID) {
        let q0 = UnitID::Qubit {
            reg_name: "q".into(),
            index: vec![0],
        };
        let q1 = UnitID::Qubit {
            reg_name: "q".into(),
            index: vec![1],
        };
        let q2 = UnitID::Qubit {
            reg_name: "q".into(),
            index: vec![2],
        };
        (q0, q1, q2)
    }

    #[test]
    fn test_precompute() {
        let (q0, q1, q2) = make_3qb();
        let mut c = Circuit::with_uids(vec![q0, q1, q2]);
        let cx1 = c.append_op(Op::CX, &[0, 1]).unwrap();
        let rx_0 = c.append_op(Op::RxF64, &[0]).unwrap();
        let cx2 = c.append_op(Op::CX, &[1, 2]).unwrap();
        let h_2 = c.append_op(Op::H, &[2]).unwrap();
        let rx_1 = c.append_op(Op::RxF64, &[1]).unwrap();
        let cx3 = c.append_op(Op::CX, &[0, 2]).unwrap();

        let mem = ConvexMemoization::memoize(&c);
        if let ConvexMemoization::Initialised {
            pred,
            succ,
            node2ind: _,
            circ: _,
        } = mem
        {
            assert_eq!(pred[&cx1], vec![None, None, None]);
            assert_eq!(pred[&cx2], vec![Some(cx1), Some(cx1), None]);
            assert_eq!(pred[&cx3], vec![Some(rx_0), Some(cx2), Some(h_2)]);

            assert_eq!(succ[&cx1], vec![Some(rx_0), Some(cx2), Some(cx2)]);
            assert_eq!(succ[&cx2], vec![Some(cx3), Some(rx_1), Some(h_2)]);
            assert_eq!(succ[&cx3], vec![None, None, None]);
        } else {
            panic!()
        }
    }

    #[test]
    fn test_convex() {
        let (q0, q1, q2) = make_3qb();
        let mut circ = Circuit::with_uids(vec![q0, q1, q2]);
        let _before1 = circ.append_op(Op::CX, &[1, 2]).unwrap();
        let _before2 = circ.append_op(Op::CX, &[0, 2]).unwrap();
        let cx1 = circ.append_op(Op::CX, &[0, 1]).unwrap();
        let cx2 = circ.append_op(Op::CX, &[0, 2]).unwrap();
        let cx3 = circ.append_op(Op::CX, &[1, 2]).unwrap();
        let _after1 = circ.append_op(Op::CX, &[1, 2]).unwrap();
        let _after2 = circ.append_op(Op::CX, &[0, 2]).unwrap();

        let mem = ConvexMemoization::memoize(&circ);
        assert!(mem.is_convex(vec![]));
        assert!(mem.is_convex(vec![cx1, cx2]));
        assert!(mem.is_convex(vec![cx2, cx3]));
        assert!(!mem.is_convex(vec![cx1, cx3]));
    }

    #[test]
    fn test_convex_2() {
        let (q0, q1, q2) = make_3qb();
        let mut circ = Circuit::with_uids(vec![q0, q1, q2]);
        let n1 = circ.append_op(Op::H, &[1]).unwrap();
        let n2 = circ.append_op(Op::CX, &[1, 0]).unwrap();
        let n3 = circ.append_op(Op::CX, &[2, 1]).unwrap();
        let n4 = circ.append_op(Op::H, &[2]).unwrap();
        let n5 = circ.append_op(Op::CX, &[2, 0]).unwrap();

        let mem = ConvexMemoization::memoize(&circ);
        assert!(mem.is_convex(vec![n1, n2, n3, n4, n5]));
    }

    #[test]
    fn test_convex_3() {
        let (q0, q1, q2) = make_3qb();
        let mut circ = Circuit::with_uids(vec![q0, q1, q2]);
        let n2 = circ.append_op(Op::CX, &[1, 0]).unwrap();

        let mem = ConvexMemoization::memoize(&circ);
        assert!(mem.is_convex(vec![n2]));
    }
}
