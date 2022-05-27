use std::collections::{HashSet, VecDeque};

use super::graph::{DefaultIx, Direction, EdgeIndex, Graph, IndexType, NodeIndex};

/*
Implementation of Kahn's algorithm
https://en.wikipedia.org/wiki/Topological_sorting#Kahn's_algorithm

For Circuits, the input vertices are already known,
 this iterator starts with them and walks the graph in a topologically sorted order
Panics if a cycle is detected in the graph.
A VecDeque is used for the node list to produce a canonical ordering,
 as successors of nodes already have a canonical ordering due to ports.

 */
pub struct TopSortWalker<'graph, N, E, Ix = DefaultIx> {
    g: &'graph Graph<N, E, Ix>,
    remaining_edges: HashSet<EdgeIndex<Ix>>,
    candidate_nodes: VecDeque<NodeIndex<Ix>>,
    cyclicity_check: bool,
    reversed: bool,
}

impl<'graph, N, E, Ix: IndexType> TopSortWalker<'graph, N, E, Ix> {
    pub fn new(g: &'graph Graph<N, E, Ix>, candidate_nodes: VecDeque<NodeIndex<Ix>>) -> Self {
        let remaining_edges = g.edges().collect();
        Self {
            g,
            candidate_nodes,
            remaining_edges,
            cyclicity_check: false,
            reversed: false,
        }
    }

    pub fn with_cyclicity_check(mut self) -> Self {
        self.cyclicity_check = true;
        self
    }

    pub fn reversed(mut self) -> Self {
        self.reversed = true;
        self
    }
}

impl<'graph, N, E, Ix: IndexType> Iterator for TopSortWalker<'graph, N, E, Ix> {
    type Item = NodeIndex<Ix>;

    fn next(&mut self) -> Option<Self::Item> {
        let (forward, backward) = if self.reversed {
            (Direction::Incoming, Direction::Outgoing)
        } else {
            (Direction::Outgoing, Direction::Incoming)
        };

        if let Some(n) = self.candidate_nodes.pop_front() {
            for e in self.g.node_edges(n, forward) {
                let m = self.g.edge_endpoints(*e).unwrap()[forward as usize].node;
                self.remaining_edges.remove(e);
                if !self
                    .g
                    .node_edges(m, backward)
                    .any(|e2| self.remaining_edges.contains(e2))
                {
                    self.candidate_nodes.push_back(m);
                }
            }

            Some(n)
        } else {
            assert!(!(self.cyclicity_check && !self.remaining_edges.is_empty()), "Edges remaining, graph may contain cycle.");
            None
        }
    }
}
