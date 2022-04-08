use super::{
    graph::{DefaultIx, Direction, EdgeIndex, Graph, IndexType, NodeIndex, NodePort, PortIndex},
    toposort::TopSortWalker,
};

pub struct Cut<Ix> {
    pub in_nodes: Vec<NodeIndex<Ix>>,
    pub out_nodes: Vec<NodeIndex<Ix>>,
}

impl<Ix> Cut<Ix> {
    pub fn new(in_nodes: Vec<NodeIndex<Ix>>, out_nodes: Vec<NodeIndex<Ix>>) -> Self {
        Self {
            in_nodes,
            out_nodes,
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
pub struct Rewrite<N, E, Ix = DefaultIx> {
    pub cut: Cut<Ix>,
    pub replacement: BoundedGraph<N, E, Ix>,
}

impl<N, E, Ix> Rewrite<N, E, Ix> {
    pub fn new(cut: Cut<Ix>, replacement: BoundedGraph<N, E, Ix>) -> Self {
        Self { cut, replacement }
    }
}

impl<N: Default, E, Ix: IndexType> Graph<N, E, Ix> {
    /// Remove all edges going in and out of the cut
    fn make_cut(&mut self, cut: &Cut<Ix>) {
        for edgevec in self.cut_boundary(cut) {
            for e in edgevec {
                self.remove_edge(e).expect("Invalid edge.");
            }
        }
        // for (i, e) in cut.in_nodes.into_iter().enumerate() {
        //     let target = self.edges[e.index()].node_ports[1];
        //     let weight = self.remove_edge(e).expect("Invalid edge.");
        //     self.add_edge(NodePort::new(entry, PortIndex::new(i)), target, weight);
        // }

        // for (i, e) in cut.out_nodes.into_iter().enumerate() {
        //     let source = self.edges[e.index()].node_ports[0];
        //     let weight = self.remove_edge(e).expect("Invalid edge.");
        //     self.add_edge(source, NodePort::new(exit, PortIndex::new(i)), weight);
        // }
    }

    /**
    Remove subgraph formed by cut and remove weights of nodes inside cut in
    TopoSort order
    */
    pub fn remove_subgraph_directed(
        &mut self,
        cut: Cut<Ix>,
        direction: Direction,
    ) -> Vec<Option<N>> {
        self.make_cut(&cut);
        match direction {
            Direction::Incoming => TopSortWalker::new(self, cut.out_nodes.into()).reversed(),
            Direction::Outgoing => TopSortWalker::new(self, cut.in_nodes.into()),
        }
        .collect::<Vec<_>>()
        .iter()
        .map(|n| self.remove_node(*n))
        .collect()
        // let removed_nodes: Vec<_> = walker.collect();
        // walker
    }

    pub fn remove_subgraph(&mut self, cut: Cut<Ix>) -> Vec<Option<N>> {
        self.remove_subgraph_directed(cut, Direction::Outgoing)
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

    fn cut_boundary(&self, cut: &Cut<Ix>) -> [Vec<EdgeIndex<Ix>>; 2] {
        [
            (&cut.in_nodes, Direction::Incoming),
            (&cut.out_nodes, Direction::Outgoing),
        ]
        .into_iter()
        .map(|(nodes, dir)| {
            nodes
                .iter()
                .map(|n| self.node_edges(*n, dir).cloned())
                .flatten()
                .collect()
        })
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
    }

    pub fn replace_with_identity(
        &mut self,
        cut: Cut<Ix>,
        new_weights: Vec<E>,
    ) -> Result<Vec<EdgeIndex<Ix>>, &str> {
        let [incoming_edges, outgoing_edges] = self.cut_boundary(&cut);

        if cut.in_nodes.len() != cut.out_nodes.len() {
            return Err("Boundary size mismatch.");
        }

        let new_edges = self.merge_edgelists(&incoming_edges, &outgoing_edges)?;

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

        let [incoming_edges, outgoing_edges] = self.cut_boundary(&cut);

        if incoming_edges.len() != new_in_edges.len() || outgoing_edges.len() != new_out_edges.len()
        {
            return Err("Boundary size mismatch.");
        }

        // insert new graph and update edge references accordingly
        let (node_map, edge_map) = self.insert_graph(replacement.graph);
        let new_in_edges: Vec<_> = new_in_edges.into_iter().map(|e| edge_map[&e]).collect();
        let new_out_edges: Vec<_> = new_out_edges.into_iter().map(|e| edge_map[&e]).collect();

        // get references to nodeports that need wiring up in self before
        // removing cut
        let left_nodeports = incoming_edges
            .iter()
            .map(|e| self.edge_endpoints(*e).map(|x| x[0]))
            .collect::<Option<Vec<NodePort<Ix>>>>()
            .ok_or("Invalid edge at cut in_edges")?;

        let right_nodeports = outgoing_edges
            .iter()
            .map(|e| self.edge_endpoints(*e).map(|x| x[1]))
            .collect::<Option<Vec<NodePort<Ix>>>>()
            .ok_or("Invalid edge at cut out_edges")?;

        // remove according to cut (including the cut edges themselves)
        let removed_node_weights = self.remove_subgraph(cut);

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
        self.replace_subgraph(rewrite.cut, rewrite.replacement)?;
        Ok(())
    }
}
