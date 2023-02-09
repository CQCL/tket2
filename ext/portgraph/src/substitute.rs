use crate::graph::ConnectError;

use super::graph::{Direction, EdgeIndex, Graph, NodeIndex, DIRECTIONS};
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fmt::{Debug, Display};
use thiserror::Error;

#[derive(Debug, Clone)]
pub struct SubgraphRef {
    pub nodes: HashSet<NodeIndex>,
}

impl SubgraphRef {
    pub fn new(nodes: HashSet<NodeIndex>) -> Self {
        Self { nodes }
    }
}

impl FromIterator<NodeIndex> for SubgraphRef {
    fn from_iter<T: IntoIterator<Item = NodeIndex>>(iter: T) -> Self {
        Self::new(iter.into_iter().collect())
    }
}

#[derive(Debug, Clone)]
pub struct BoundedSubgraph {
    pub subgraph: SubgraphRef,
    pub edges: [Vec<EdgeIndex>; 2],
}

impl BoundedSubgraph {
    pub fn new(subgraph: SubgraphRef, edges: [Vec<EdgeIndex>; 2]) -> Self {
        Self { subgraph, edges }
    }

    pub fn from_node<N, E>(graph: &Graph<N, E>, node: NodeIndex) -> Self {
        Self {
            subgraph: [node].into_iter().collect(),
            edges: [
                graph.node_edges(node, Direction::Incoming).collect(),
                graph.node_edges(node, Direction::Outgoing).collect(),
            ],
        }
    }
}

#[derive(Clone)]
pub struct OpenGraph<N, E> {
    pub graph: Graph<N, E>,
    pub dangling: [Vec<EdgeIndex>; 2],
}

impl<N, E> OpenGraph<N, E> {
    pub fn new(graph: Graph<N, E>, in_ports: Vec<EdgeIndex>, out_ports: Vec<EdgeIndex>) -> Self {
        Self {
            graph,
            dangling: [in_ports, out_ports],
        }
    }
}

impl<N: Debug, E: Debug> Debug for OpenGraph<N, E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenGraph")
            .field("graph", &self.graph)
            .field("in_edges", &self.dangling[0])
            .field("out_edges", &self.dangling[1])
            .finish()
    }
}

#[derive(Debug, Clone)]
pub struct Rewrite<N, E> {
    pub subg: BoundedSubgraph,
    pub replacement: OpenGraph<N, E>,
}

impl<N, E> Rewrite<N, E> {
    pub fn new(subg: BoundedSubgraph, replacement: OpenGraph<N, E>) -> Self {
        Self { subg, replacement }
    }
}

impl<N: Default + Debug + Display, E: Debug + Display + Clone> Graph<N, E> {
    fn break_edge(&mut self, edge: EdgeIndex) -> Result<[EdgeIndex; 2], RewriteError> {
        let wt = self
            .edge_weight(edge)
            .ok_or(RewriteError::EdgeMissing)?
            .clone();

        let new_e = self.add_edge(wt);
        self.replace_connection(edge, new_e, Direction::Incoming)
            .map_err(RewriteError::Connect)
            .unwrap();

        Ok([edge, new_e])
    }
    /// Remove subgraph formed by subg and return weights of nodes inside subg
    fn remove_subgraph(&mut self, subgraph: &BoundedSubgraph) -> Vec<Option<N>> {
        let boundary_edges =
            BTreeSet::from_iter(subgraph.edges.iter().flat_map(|x| x.iter().copied()));
        subgraph
            .subgraph
            .nodes
            .iter()
            .map(|n| {
                let edges: Vec<_> = DIRECTIONS
                    .iter()
                    .flat_map(|d| self.node_edges(*n, *d))
                    .filter(|e| !boundary_edges.contains(e))
                    .collect();

                for edge in edges {
                    self.remove_edge(edge);
                }

                self.remove_node(*n)
            })
            .collect()
    }

    fn replace_subgraph(
        &mut self,
        subgraph: BoundedSubgraph,
        replacement: OpenGraph<N, E>,
    ) -> Result<Vec<Option<N>>, RewriteError> {
        // if subgraph.subgraph.nodes.is_empty() {
        //     return Err(RewriteError::EmptySubgraph);
        // }

        // TODO type check.
        for direction in DIRECTIONS {
            let edges = &subgraph.edges[direction.index()];
            let ports = &replacement.dangling[direction.index()];

            if edges.len() != ports.len() {
                return Err(RewriteError::BoundarySize);
            }
        }

        let removed = self.remove_subgraph(&subgraph);
        // insert new graph and update edge references accordingly
        let (_, mut edge_map) = self.insert_graph(replacement.graph);

        // edges in the subgraph may be split in to two edges if they remain
        // connected at both ends
        // this breaks the guarantee that interface edges are always left unchanged
        let mut updated_subg_edges: BTreeMap<EdgeIndex, [EdgeIndex; 2]> = BTreeMap::new();

        for direction in DIRECTIONS {
            let subg_edges = &subgraph.edges[direction.index()];
            let repl_edges = &replacement.dangling[direction.index()];

            for (sub_edge, repl_edge) in subg_edges.iter().zip(repl_edges) {
                let sub_edge = updated_subg_edges
                    .get(sub_edge)
                    .unwrap_or(&[*sub_edge, *sub_edge])[direction.index()];
                let from = edge_map[repl_edge];
                // TODO: There should be a check to make sure this can not fail
                // before we merge the first edge to avoid leaving the graph in an
                // invalid state.
                if self.merge_edges(from, sub_edge).is_err() {
                    // if sub_edge is still connected at both ends, break in to two
                    let newes = self.break_edge(sub_edge)?;

                    updated_subg_edges.insert(sub_edge, newes);
                    // assumes the other edge in newes will be fully connected
                    // later - graph could be left in invalid state if not
                    self.merge_edges(from, newes[direction.index()]).unwrap();
                }
                // Update edge_map to point to new merged edge
                if let Some(e) = edge_map.get_mut(repl_edge) {
                    *e = sub_edge;
                }
            }
        }

        Ok(removed)
    }

    pub fn apply_rewrite(&mut self, rewrite: Rewrite<N, E>) -> Result<(), RewriteError> {
        self.replace_subgraph(rewrite.subg, rewrite.replacement)?;
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum RewriteError {
    #[error("cannot replace empty subgraph")]
    EmptySubgraph,
    #[error("boundary size mismatch")]
    BoundarySize,
    #[error("Edge missing")]
    EdgeMissing,
    #[error("Connect Error")]
    Connect(ConnectError),
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::error::Error;

    use crate::substitute::{BoundedSubgraph, OpenGraph};

    use super::Graph;

    #[test]
    fn test_remove_subgraph() -> Result<(), Box<dyn Error>> {
        let mut g = Graph::<i8, i8>::with_capacity(3, 2);

        let e1 = g.add_edge(-1);
        let e2 = g.add_edge(-2);
        let e3 = g.add_edge(-3);

        let _ = g.add_node_with_edges(0, [], [e1, e3])?;
        let n1 = g.add_node_with_edges(1, [e1], [e2])?;
        let _ = g.add_node_with_edges(2, [e2, e3], [])?;

        assert_eq!(g.node_count(), 3);
        assert_eq!(g.edge_count(), 3);

        let mut new_g = g.clone();

        let rem_nodes = new_g.remove_subgraph(&BoundedSubgraph::new(
            [n1].into_iter().collect(),
            [vec![e1], vec![e2]],
        ));

        assert_eq!(rem_nodes, vec![Some(1)]);

        let correct_weights: HashSet<_> = HashSet::from_iter([0, 2]);
        assert_eq!(
            HashSet::from_iter(new_g.node_weights().copied()),
            correct_weights
        );

        let correct_weights: HashSet<_> = HashSet::from_iter([-1, -2, -3]);
        assert_eq!(
            HashSet::from_iter(new_g.edge_weights().copied()),
            correct_weights
        );

        assert_eq!(new_g.edge_count(), 3);
        assert_eq!(new_g.node_count(), 2);

        Ok(())
    }

    #[test]
    fn test_insert_graph() -> Result<(), Box<dyn Error>> {
        let mut g = {
            let mut g = Graph::<i8, i8>::with_capacity(3, 2);

            let e1 = g.add_edge(-1);
            let e2 = g.add_edge(-2);

            let _ = g.add_node_with_edges(0, [], [e1])?;
            let _ = g.add_node_with_edges(1, [e1], [e2])?;
            let _ = g.add_node_with_edges(2, [e2], [])?;

            g
        };

        let g2 = {
            let mut g2 = Graph::<i8, i8>::with_capacity(2, 1);

            let e3 = g2.add_edge(-3);

            let _ = g2.add_node_with_edges(3, [], [e3])?;
            let _ = g2.add_node_with_edges(4, [e3], [])?;

            g2
        };

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

        Ok(())
    }

    #[test]
    fn test_replace_subgraph() -> Result<(), Box<dyn Error>> {
        let mut g = Graph::<i8, i8>::with_capacity(3, 2);

        let e1 = g.add_edge(-1);
        let e2 = g.add_edge(-2);
        let e3 = g.add_edge(-3);

        let _ = g.add_node_with_edges(0, [], [e1, e3])?;
        let n1 = g.add_node_with_edges(1, [e1], [e2])?;
        let _ = g.add_node_with_edges(2, [e2, e3], [])?;

        let mut g2 = Graph::<i8, i8>::with_capacity(2, 1);
        // node to be inserted
        let e4 = g2.add_edge(-4);
        let e5 = g2.add_edge(-5);
        let _ = g2.add_node_with_edges(3, [e4], [e5])?;

        let rem_nodes = g
            .replace_subgraph(
                BoundedSubgraph::new([n1].into_iter().collect(), [vec![e1], vec![e2]]),
                OpenGraph::new(g2, vec![e4], vec![e5]),
            )
            .unwrap();

        assert_eq!(rem_nodes, vec![Some(1)]);

        let correct_weights: HashSet<_> = HashSet::from_iter([0, 2, 3]);
        assert_eq!(
            HashSet::from_iter(g.node_weights().copied()),
            correct_weights
        );

        let correct_weights: HashSet<_> = HashSet::from_iter([-1, -2, -3]);
        assert_eq!(
            HashSet::from_iter(g.edge_weights().copied()),
            correct_weights
        );

        Ok(())
    }
}
