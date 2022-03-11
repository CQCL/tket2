use super::{
    graph::{Direction, EdgeIndex, Graph, IndexType, NodeIndex, NodePort, PortIndex},
    toposort::TopSortWalker,
};

pub struct Cut<Ix> {
    pub in_edges: Vec<EdgeIndex<Ix>>,
    pub out_edges: Vec<EdgeIndex<Ix>>,
}

impl<Ix> Cut<Ix> {
    pub fn new(in_edges: Vec<EdgeIndex<Ix>>, out_edges: Vec<EdgeIndex<Ix>>) -> Self {
        Self {
            in_edges,
            out_edges,
        }
    }
}

pub struct BoundedGraph<N, E, Ix> {
    pub graph: Graph<N, E, Ix>,
    pub entry: NodeIndex<Ix>,
    pub exit: NodeIndex<Ix>,
}

impl<N, E, Ix> BoundedGraph<N, E, Ix> {
    pub fn new(graph: Graph<N, E, Ix>, entry: NodeIndex<Ix>, exit: NodeIndex<Ix>) -> Self {
        Self { graph, entry, exit }
    }
}

impl<N: Default, E, Ix: IndexType> Graph<N, E, Ix> {
    /// Remove provided edges, replace with edges to new placeholder entry and
    /// exit nodes, and return references to those nodes.
    fn make_cut(&mut self, cut: Cut<Ix>) -> (NodeIndex<Ix>, NodeIndex<Ix>) {
        let entry = self.add_node_with_capacity(cut.in_edges.len(), N::default());
        let exit = self.add_node_with_capacity(cut.out_edges.len(), N::default());

        for (i, e) in cut.in_edges.into_iter().enumerate() {
            let target = self.edges[e.index()].node_ports[1];
            let weight = self.remove_edge(e).expect("Invalid edge.");
            self.add_edge(NodePort::new(entry, PortIndex::new(i)), target, weight);
        }

        for (i, e) in cut.out_edges.into_iter().enumerate() {
            let source = self.edges[e.index()].node_ports[0];
            let weight = self.remove_edge(e).expect("Invalid edge.");
            self.add_edge(source, NodePort::new(exit, PortIndex::new(i)), weight);
        }

        (entry, exit)
    }

    /**
    Remove subgraph formed by cut and remove weights of nodes inside cut in
    TopoSort order
    */
    pub fn remove_subgraph(&mut self, cut: Cut<Ix>) -> Vec<Option<N>> {
        let (entry, exit) = self.make_cut(cut);
        let removed_nodes: Vec<_> = TopSortWalker::new(self, [entry].into()).collect();
        removed_nodes
            .into_iter()
            .filter_map(|node| {
                let weight = self.remove_node(node);
                if [entry, exit].contains(&node) {
                    None
                } else {
                    Some(weight)
                }
            })
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
    pub fn replace_with_identity(
        &mut self,
        cut: Cut<Ix>,
        new_weights: Vec<E>,
    ) -> Result<Vec<EdgeIndex<Ix>>, &str> {
        if cut.in_edges.len() != cut.out_edges.len() {
            return Err("Boundary size mismatch.");
        }

        let new_edges = self.merge_edgelists(&cut.in_edges, &cut.out_edges)?;

        self.remove_subgraph(cut);

        Ok(new_edges
            .into_iter()
            .zip(new_weights)
            .map(|((a, b), weight)| self.add_edge(a, b, weight))
            .collect())
    }

    pub fn replace_subgraph(
        &mut self,
        cut: Cut<Ix>,
        replacement: BoundedGraph<N, E, Ix>,
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
        if cut.in_edges.len() != new_in_edges.len() || cut.out_edges.len() != new_out_edges.len() {
            return Err("Boundary size mismatch.");
        }

        // insert new graph and update edge references accordingly
        let (node_map, edge_map) = self.insert_graph(replacement.graph);
        let new_in_edges: Vec<_> = new_in_edges.into_iter().map(|e| edge_map[&e]).collect();
        let new_out_edges: Vec<_> = new_out_edges.into_iter().map(|e| edge_map[&e]).collect();

        // get references to nodeports that need wiring up in self before
        // removing cut
        let left_nodeports = cut
            .in_edges
            .iter()
            .map(|e| self.edge_endpoints(*e).map(|(x, _)| x))
            .collect::<Option<Vec<NodePort<Ix>>>>()
            .ok_or("Invalid edge at cut in_edges")?;

        let right_nodeports = cut
            .out_edges
            .iter()
            .map(|e| self.edge_endpoints(*e).map(|(_, x)| x))
            .collect::<Option<Vec<NodePort<Ix>>>>()
            .ok_or("Invalid edge at cut out_edges")?;

        // remove according to cut (including the cut edges themselves)
        let removed_node_weights = self.remove_subgraph(cut);

        // rewire replacement I/O edges from their entry/exit nodes to the
        // appropriate nodeports in self
        for (e, np) in new_in_edges.into_iter().zip(left_nodeports) {
            let (_, target) = self.edge_endpoints(e).unwrap();
            let weight = self.remove_edge(e).unwrap();

            self.add_edge(np, target, weight);
        }

        for (e, np) in new_out_edges.into_iter().zip(right_nodeports) {
            let (source, _) = self.edge_endpoints(e).unwrap();
            let weight = self.remove_edge(e).unwrap();

            self.add_edge(source, np, weight);
        }

        // all edges from entry/exit should already have been removed, these
        // nodes are safe to delete

        self.remove_node(node_map[&replacement.entry]);
        self.remove_node(node_map[&replacement.exit]);
        Ok(removed_node_weights)
    }
}
