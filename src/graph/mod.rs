pub mod graph;
pub mod substitute;
pub mod toposort;
pub mod dot;

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use crate::graph::substitute::{BoundedGraph, Cut};

    use super::graph::Graph;

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

        let rem_nodes = g.remove_subgraph(Cut::new(vec![e1], vec![e2]));

        assert_eq!(rem_nodes, vec![Some(1)]);

        let correct_weights: HashSet<_> = HashSet::from_iter([0, 2].into_iter());
        assert_eq!(
            HashSet::from_iter(g.nodes().map(|n| *g.node_weight(n).unwrap())),
            correct_weights
        );

        let correct_weights: HashSet<_> = HashSet::from_iter([5].into_iter());
        assert_eq!(
            HashSet::from_iter(g.edges().map(|e| *g.edge_weight(e).unwrap())),
            correct_weights
        );
        dbg!(g
            .nodes()
            .map(|n| *g.node_weight(n).unwrap())
            .collect::<Vec<_>>());
        assert_eq!(g.edge_count(), 1);
        assert_eq!(g.node_count(), 2);
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

        g.replace_with_identity(Cut::new(vec![e1], vec![e2]), vec![6])
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
                Cut::new(vec![e1], vec![e2]),
                BoundedGraph::new(g2, g20, g22),
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
