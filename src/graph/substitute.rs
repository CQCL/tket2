use super::graph::{DefaultIx, Direction, EdgeIndex, Graph, IndexType, NodeIndex, NodePort};
use std::collections::HashSet;
use std::hash::Hash;

pub trait HashIx: Eq + Hash + IndexType {}
impl<T: Eq + Hash + IndexType> HashIx for T {}

pub struct SubgraphRef<HashIx> {
    pub nodes: HashSet<NodeIndex<HashIx>>,
}

impl<Ix: HashIx> SubgraphRef<Ix> {
    pub fn new(nodes: HashSet<NodeIndex<Ix>>) -> Self {
        Self { nodes }
    }
}

impl<Ix: HashIx, T: Iterator<Item = NodeIndex<Ix>>> From<T> for SubgraphRef<Ix> {
    fn from(it: T) -> Self {
        Self {
            nodes: HashSet::from_iter(it),
        }
    }
}

pub struct BoundedSubgraph<Ix: HashIx> {
    pub subg: SubgraphRef<Ix>,
    pub edges: [Vec<EdgeIndex<Ix>>; 2],
}

impl<Ix: HashIx> BoundedSubgraph<Ix> {
    pub fn new(subg: SubgraphRef<Ix>, edges: [Vec<EdgeIndex<Ix>>; 2]) -> Self {
        Self { subg, edges }
    }

    pub fn from_node<N, E>(graph: &Graph<N, E, Ix>, node: NodeIndex<Ix>) -> Self {
        Self {
            subg: [node].into_iter().into(),
            edges: [
                graph
                    .node_edges(node, Direction::Incoming)
                    .copied()
                    .collect(),
                graph
                    .node_edges(node, Direction::Outgoing)
                    .copied()
                    .collect(),
            ],
        }
    }
}

pub struct ClosedGraph<N, E, Ix> {
    pub graph: Graph<N, E, Ix>,
    pub entry: NodeIndex<Ix>,
    pub exit: NodeIndex<Ix>,
}

impl<N, E, Ix> ClosedGraph<N, E, Ix> {
    pub fn new(graph: Graph<N, E, Ix>, entry: NodeIndex<Ix>, exit: NodeIndex<Ix>) -> Self {
        Self { graph, entry, exit }
    }
}
pub struct Rewrite<N, E, Ix: HashIx = DefaultIx> {
    pub subg: BoundedSubgraph<Ix>,
    pub replacement: ClosedGraph<N, E, Ix>,
}

impl<N, E, Ix: HashIx> Rewrite<N, E, Ix> {
    pub fn new(subg: BoundedSubgraph<Ix>, replacement: ClosedGraph<N, E, Ix>) -> Self {
        Self { subg, replacement }
    }
}

impl<N: Default, E, Ix: HashIx> Graph<N, E, Ix> {
    /// Remove all edges going in and out of the subgraph
    fn make_cuts(&mut self, subg: &BoundedSubgraph<Ix>) {
        for edgevec in &subg.edges {
            for e in edgevec {
                self.remove_edge(*e).expect("Invalid edge.");
            }
        }
        // for (i, e) in subg.in_nodes.into_iter().enumerate() {
        //     let target = self.edges[e.index()].node_ports[1];
        //     let weight = self.remove_edge(e).expect("Invalid edge.");
        //     self.add_edge(NodePort::new(entry, PortIndex::new(i)), target, weight);
        // }

        // for (i, e) in subg.out_nodes.into_iter().enumerate() {
        //     let source = self.edges[e.index()].node_ports[0];
        //     let weight = self.remove_edge(e).expect("Invalid edge.");
        //     self.add_edge(source, NodePort::new(exit, PortIndex::new(i)), weight);
        // }
    }
    // fn remove_subgraph_directed(&mut self, subg: BoundedSubgraph<Ix>) -> Vec<Option<N>> {
    //     self.make_cuts(&subg);
    //     // match direction {
    //         //     Direction::Incoming => TopSortWalker::new(self, subg.out_nodes.into()).reversed(),
    //         //     Direction::Outgoing => TopSortWalker::new(self, subg.in_nodes.into()),
    //         // }
    //         // .collect::<Vec<_>>()
    //         subg.subg
    //         .nodes
    //         .into_iter()
    //         .map(|n| self.remove_node(n))
    //         .collect()
    //         // let removed_nodes: Vec<_> = walker.collect();
    //         // walker
    //     }

    /**
    Remove subgraph formed by subg and remove weights of nodes inside subg
    */
    pub fn remove_subgraph(&mut self, subg: BoundedSubgraph<Ix>) -> Vec<Option<N>> {
        self.make_cuts(&subg);
        subg.subg
            .nodes
            .into_iter()
            .map(|n| self.remove_node(n))
            .collect()
    }

    fn merge_edgelists(
        &mut self,
        left_edges: &Vec<EdgeIndex<Ix>>,
        right_edges: &Vec<EdgeIndex<Ix>>,
    ) -> Result<Vec<(NodePort<Ix>, NodePort<Ix>)>, &'static str> {
        left_edges
            .iter()
            .zip(right_edges.iter())
            .map(|(e_l, e_r)| self.redirect_edges(*e_l, *e_r))
            .collect()
    }

    fn redirect_edges(
        &self,
        first: EdgeIndex<Ix>,
        second: EdgeIndex<Ix>,
    ) -> Result<(NodePort<Ix>, NodePort<Ix>), &'static str> {
        Ok((
            self.edges
                .get(first.index())
                .ok_or("Edge not found")?
                .node_ports[0],
            self.edges
                .get(second.index())
                .ok_or("Edge not found")?
                .node_ports[1],
        ))
    }

    // fn cut_boundary(&self, subg: &BoundedSubgraph<Ix>) -> [Vec<EdgeIndex<Ix>>; 2] {
    //     [
    //         (&subg.in_nodes, Direction::Incoming),
    //         (&subg.out_nodes, Direction::Outgoing),
    //     ]
    //     .into_iter()
    //     .map(|(nodes, dir)| {
    //         nodes
    //             .iter()
    //             .map(|n| self.node_edges(*n, dir).cloned())
    //             .flatten()
    //             .collect()
    //     })
    //     .collect::<Vec<_>>()
    //     .try_into()
    //     .unwrap()
    // }

    fn replace_with_identity(
        &mut self,
        subg: BoundedSubgraph<Ix>,
        new_weights: Vec<E>,
    ) -> Result<Vec<EdgeIndex<Ix>>, &str> {
        if subg.edges[0].len() != subg.edges[1].len() {
            return Err("Boundary size mismatch.");
        }

        let new_edges = self.merge_edgelists(&subg.edges[0], &subg.edges[1])?;

        self.remove_subgraph(subg);

        Ok(new_edges
            .into_iter()
            .zip(new_weights)
            .map(|((a, b), weight)| self.add_edge(a, b, weight))
            .collect())
    }

    fn replace_subgraph(
        &mut self,
        subg: BoundedSubgraph<Ix>,
        replacement: ClosedGraph<N, E, Ix>,
    ) -> Result<Vec<Option<N>>, &str> {
        // get all the entry and exit edges in replacement graph
        let new_in_edges: Vec<_> = replacement
            .graph
            .node_edges(replacement.entry, Direction::Outgoing)
            .cloned()
            .collect();
        let new_out_edges: Vec<_> = replacement
            .graph
            .node_edges(replacement.exit, Direction::Incoming)
            .cloned()
            .collect();

        let [incoming_edges, outgoing_edges] = &subg.edges;

        if incoming_edges.len() != new_in_edges.len() || outgoing_edges.len() != new_out_edges.len()
        {
            return Err("Boundary size mismatch.");
        }

        // insert new graph and update edge references accordingly
        let (node_map, edge_map) = self.insert_graph(replacement.graph);
        let new_in_edges: Vec<_> = new_in_edges.into_iter().map(|e| edge_map[&e]).collect();
        let new_out_edges: Vec<_> = new_out_edges.into_iter().map(|e| edge_map[&e]).collect();
        // get references to nodeports that need wiring up in self before
        // removing subg
        let left_nodeports = incoming_edges
            .iter()
            .map(|e| self.edge_endpoints(*e).map(|x| x[0]))
            .collect::<Option<Vec<NodePort<Ix>>>>()
            .ok_or("Invalid edge at subg in_edges")?;

        let right_nodeports = outgoing_edges
            .iter()
            .map(|e| self.edge_endpoints(*e).map(|x| x[1]))
            .collect::<Option<Vec<NodePort<Ix>>>>()
            .ok_or("Invalid edge at subg out_edges")?;

        // remove according to subg (including the subg edges themselves)
        // dbg!(&new_in_edges);
        // println!("{}", crate::graph::dot::dot_string(&self));

        let removed_node_weights = self.remove_subgraph(subg);
        // rewire replacement I/O edges from their entry/exit nodes to the
        // appropriate nodeports in self
        for (e, np) in new_in_edges.into_iter().zip(left_nodeports) {
            let [_, target] = self.edge_endpoints(e).unwrap();
            let weight = self.remove_edge(e).unwrap();

            self.add_edge(np, target, weight);
        }

        for (e, np) in new_out_edges.into_iter().zip(right_nodeports) {
            let [source, _] = self.edge_endpoints(e).unwrap();
            let weight = self.remove_edge(e).unwrap();

            self.add_edge(source, np, weight);
        }

        // all edges from entry/exit should already have been removed, these
        // nodes are safe to delete

        self.remove_node(node_map[&replacement.entry]);
        self.remove_node(node_map[&replacement.exit]);
        Ok(removed_node_weights)
    }

    pub fn apply_rewrite(&mut self, rewrite: Rewrite<N, E, Ix>) -> Result<(), String> {
        self.replace_subgraph(rewrite.subg, rewrite.replacement)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use crate::graph::substitute::{BoundedSubgraph, ClosedGraph};

    use super::Graph;

    #[test]
    fn test_remove_subgraph() {
        let mut g = Graph::<u8, u8, u8>::with_capacity(3, 2);

        let n0 = g.add_node(0);
        let n1 = g.add_node(1);
        let n2 = g.add_node(2);

        let e1 = g.add_edge((n0, 0), (n1, 0), 3);
        let e2 = g.add_edge((n1, 0), (n2, 0), 4);
        let _e3 = g.add_edge((n0, 1), (n2, 1), 5);

        assert_eq!(g.node_count(), 3);
        assert_eq!(g.edge_count(), 3);
        let mut new_g = g.clone();
        let rem_nodes = new_g.remove_subgraph(BoundedSubgraph::new(
            [n1].into_iter().into(),
            [vec![e1], vec![e2]],
        ));

        assert_eq!(rem_nodes, vec![Some(1)]);

        let correct_weights: HashSet<_> = HashSet::from_iter([0, 2].into_iter());
        assert_eq!(
            HashSet::from_iter(new_g.nodes().map(|n| *g.node_weight(n).unwrap())),
            correct_weights
        );

        let correct_weights: HashSet<_> = HashSet::from_iter([5].into_iter());
        assert_eq!(
            HashSet::from_iter(new_g.edges().map(|e| *g.edge_weight(e).unwrap())),
            correct_weights
        );

        assert_eq!(new_g.edge_count(), 1);
        assert_eq!(new_g.node_count(), 2);
    }

    #[test]
    fn test_replace_with_identity() {
        let mut g = Graph::<u8, u8, u8>::with_capacity(3, 2);

        let n0 = g.add_node(0);
        let n1 = g.add_node(1);
        let n2 = g.add_node(2);

        let e1 = g.add_edge((n0, 0), (n1, 0), 3);
        let e2 = g.add_edge((n1, 0), (n2, 0), 4);
        let _e3 = g.add_edge((n0, 1), (n2, 1), 5);

        assert_eq!(g.node_count(), 3);
        assert_eq!(g.edge_count(), 3);

        g.replace_with_identity(
            BoundedSubgraph::new([n1].into_iter().into(), [vec![e1], vec![e2]]),
            vec![6],
        )
        .unwrap();

        let correct_weights: HashSet<_> = HashSet::from_iter([0, 2].into_iter());
        assert_eq!(
            HashSet::from_iter(g.nodes().map(|n| *g.node_weight(n).unwrap())),
            correct_weights
        );
        let correct_weights: HashSet<_> = HashSet::from_iter([5, 6].into_iter());
        assert_eq!(
            HashSet::from_iter(g.edges().map(|e| *g.edge_weight(e).unwrap())),
            correct_weights
        );
    }
    #[test]
    fn test_insert_graph() {
        let mut g = Graph::<i8, i8, u8>::with_capacity(3, 2);

        let n0 = g.add_node(0);
        let n1 = g.add_node(1);
        let n2 = g.add_node(2);

        let _e1 = g.add_edge((n0, 0), (n1, 0), -1);
        let _e2 = g.add_edge((n1, 0), (n2, 0), -2);

        let mut g2 = Graph::<i8, i8, u8>::with_capacity(2, 1);

        let g20 = g2.add_node(3);
        let g21 = g2.add_node(4);

        let _g2e1 = g2.add_edge((g20, 0), (g21, 0), -3);

        g.insert_graph(g2);

        let correct_weights: HashSet<_> = HashSet::from_iter([0, 1, 2, 3, 4].into_iter());
        assert_eq!(
            HashSet::from_iter(g.nodes().map(|n| *g.node_weight(n).unwrap())),
            correct_weights
        );

        let correct_weights: HashSet<_> = HashSet::from_iter([-1, -2, -3].into_iter());
        assert_eq!(
            HashSet::from_iter(g.edges().map(|e| *g.edge_weight(e).unwrap())),
            correct_weights
        );
    }

    #[test]
    fn test_replace_subgraph() {
        let mut g = Graph::<i8, i8, u8>::with_capacity(3, 2);

        let n0 = g.add_node(0);
        let n1 = g.add_node(1);
        let n2 = g.add_node(2);

        let e1 = g.add_edge((n0, 0), (n1, 0), -1);
        let e2 = g.add_edge((n1, 0), (n2, 0), -2);
        let _e3 = g.add_edge((n0, 1), (n2, 1), -3);

        let mut g2 = Graph::<i8, i8, u8>::with_capacity(2, 1);
        //entry
        let g20 = g2.add_node(3);
        // node to be inserted
        let g21 = g2.add_node(4);
        //exit
        let g22 = g2.add_node(5);

        let _g2e1 = g2.add_edge((g20, 0), (g21, 0), -4);
        let _g2e2 = g2.add_edge((g21, 0), (g22, 0), -5);

        let rem_nodes = g
            .replace_subgraph(
                BoundedSubgraph::new([n1].into_iter().into(), [vec![e1], vec![e2]]),
                ClosedGraph::new(g2, g20, g22),
            )
            .unwrap();

        assert_eq!(rem_nodes, vec![Some(1)]);

        let correct_weights: HashSet<_> = HashSet::from_iter([0, 2, 4].into_iter());
        assert_eq!(
            HashSet::from_iter(g.nodes().map(|n| *g.node_weight(n).unwrap())),
            correct_weights
        );

        let correct_weights: HashSet<_> = HashSet::from_iter([-4, -5, -3].into_iter());
        assert_eq!(
            HashSet::from_iter(g.edges().map(|e| *g.edge_weight(e).unwrap())),
            correct_weights
        );

        // assert_eq!(g.node_count(), 3);
        assert_eq!(g.edge_count(), 3);
    }
}
