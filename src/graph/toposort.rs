use std::collections::{HashSet, VecDeque};

use super::graph::{DefaultIx, EdgeIndex, Graph, IndexType, NodeIndex, NodePort};

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
}

impl<'graph, N, E, Ix: IndexType> TopSortWalker<'graph, N, E, Ix> {
    pub fn new(g: &'graph Graph<N, E, Ix>, candidate_nodes: VecDeque<NodeIndex<Ix>>) -> Self {
        let remaining_edges = g.edges().collect();
        Self {
            g,
            candidate_nodes,
            remaining_edges,
        }
    }
}

impl<'graph, N, E, Ix: IndexType> Iterator for TopSortWalker<'graph, N, E, Ix> {
    type Item = NodeIndex<Ix>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(n) = self.candidate_nodes.pop_front() {
            for e in self.g.outgoing_edges(n) {
                let (_, NodePort { node: m, .. }) = self.g.edge_endpoints(*e).unwrap();
                self.remaining_edges.remove(e);
                if self
                    .g
                    .incoming_edges(m)
                    .filter(|&e2| self.remaining_edges.contains(e2))
                    .next()
                    .is_none()
                {
                    self.candidate_nodes.push_back(m);
                }
            }

            Some(n)
        } else {
            if !self.remaining_edges.is_empty() {
                panic!("Edges remaining, graph may contain cycle.");
            }
            None
        }
    }
}
