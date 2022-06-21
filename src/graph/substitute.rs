use super::graph::{DefaultIx, Direction, EdgeIndex, Graph, IndexType, NodeIndex, NodePort};
use std::collections::HashSet;
use std::fmt::{Debug, Display};
use std::hash::Hash;

pub trait HashIx: Eq + Hash + IndexType {}
impl<T: Eq + Hash + IndexType> HashIx for T {}

#[derive(Debug, Clone)]
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

#[derive(Debug, Clone)]
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

#[derive(Clone)]
pub struct OpenGraph<N, E, Ix: IndexType> {
    pub graph: Graph<N, E, Ix>,
    pub in_ports: Vec<NodePort<Ix>>,
    pub out_ports: Vec<NodePort<Ix>>,
}

impl<N, E, Ix: IndexType> OpenGraph<N, E, Ix> {
    pub fn new(
        graph: Graph<N, E, Ix>,
        in_ports: Vec<NodePort<Ix>>,
        out_ports: Vec<NodePort<Ix>>,
    ) -> Self {
        Self {
            graph,
            in_ports,
            out_ports,
        }
    }
}

impl<N: Debug, E: Debug, Ix: IndexType> Debug for OpenGraph<N, E, Ix> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenGraph")
            .field("graph", &self.graph)
            .field("in_ports", &self.in_ports)
            .field("out_ports", &self.out_ports)
            .finish()
    }
}

#[derive(Debug, Clone)]
pub struct Rewrite<N, E, Ix: HashIx = DefaultIx> {
    pub subg: BoundedSubgraph<Ix>,
    pub replacement: OpenGraph<N, E, Ix>,
}

impl<N, E, Ix: HashIx> Rewrite<N, E, Ix> {
    pub fn new(subg: BoundedSubgraph<Ix>, replacement: OpenGraph<N, E, Ix>) -> Self {
        Self { subg, replacement }
    }
}

impl<N: Default + Debug + Display, E: Debug + Display, Ix: HashIx> Graph<N, E, Ix> {
    /**
    Remove subgraph formed by subg and return weights of nodes inside subg
    */
    fn remove_subgraph(&mut self, subg: BoundedSubgraph<Ix>) -> Vec<Option<N>> {
        subg.subg
            .nodes
            .into_iter()
            .map(|n| self.remove_node(n))
            .collect()
    }

    fn replace_subgraph(
        &mut self,
        subg: BoundedSubgraph<Ix>,
        replacement: OpenGraph<N, E, Ix>,
    ) -> Result<Vec<Option<N>>, &str> {
        if subg.subg.nodes.is_empty() {
            return Err("Cannot replace empty subgraph.");
        }
        let [incoming_edges, outgoing_edges] = &subg.edges;

        if incoming_edges.len() != replacement.in_ports.len()
            || outgoing_edges.len() != replacement.out_ports.len()
        {
            // TODO type check.
            return Err("Boundary size mismatch.");
        }

        // insert new graph and update edge references accordingly
        let (node_map, _) = self.insert_graph(replacement.graph);

        for (e, mut np) in incoming_edges.iter().zip(replacement.in_ports) {
            let [source, _] = self.edge_endpoints(*e).expect("missing edge.");
            np.node = node_map[&np.node];
            self.update_edge(*e, source, np);
        }
        for (e, mut np) in outgoing_edges.iter().zip(replacement.out_ports) {
            let [_, target] = self.edge_endpoints(*e).expect("missing edge.");
            np.node = node_map[&np.node];
            self.update_edge(*e, np, target);
        }
        Ok(self.remove_subgraph(subg))
    }

    pub fn apply_rewrite(&mut self, rewrite: Rewrite<N, E, Ix>) -> Result<(), String> {
        self.replace_subgraph(rewrite.subg, rewrite.replacement)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use crate::graph::substitute::{BoundedSubgraph, OpenGraph};

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
            HashSet::from_iter(new_g.node_weights().copied()),
            correct_weights
        );

        let correct_weights: HashSet<_> = HashSet::from_iter([5].into_iter());
        assert_eq!(
            HashSet::from_iter(new_g.edge_weights().copied()),
            correct_weights
        );

        assert_eq!(new_g.edge_count(), 1);
        assert_eq!(new_g.node_count(), 2);
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
            HashSet::from_iter(g.node_weights().copied()),
            correct_weights
        );

        let correct_weights: HashSet<_> = HashSet::from_iter([-1, -2, -3].into_iter());
        assert_eq!(
            HashSet::from_iter(g.edge_weights().copied()),
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
        // node to be inserted
        let g2n = g2.add_node(4);

        let rem_nodes = g
            .replace_subgraph(
                BoundedSubgraph::new([n1].into_iter().into(), [vec![e1], vec![e2]]),
                OpenGraph::new(g2, vec![(g2n, 0).into()], vec![(g2n, 0).into()]),
            )
            .unwrap();

        assert_eq!(rem_nodes, vec![Some(1)]);

        let correct_weights: HashSet<_> = HashSet::from_iter([0, 2, 4].into_iter());
        assert_eq!(
            HashSet::from_iter(g.node_weights().copied()),
            correct_weights
        );

        let correct_weights: HashSet<_> = HashSet::from_iter([-1, -2, -3].into_iter());
        assert_eq!(
            HashSet::from_iter(g.edge_weights().copied()),
            correct_weights
        );

        // assert_eq!(g.node_count(), 3);
        assert_eq!(g.edge_count(), 3);
    }
}
