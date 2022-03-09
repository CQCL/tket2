use std::{
    collections::{HashSet, VecDeque},
    fmt::Debug,
};

use super::graph::{EdgeIndex, Graph, IndexType, NodePort};

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
    pub in_edges: Vec<EdgeIndex<Ix>>,
    pub out_edges: Vec<EdgeIndex<Ix>>,
}

impl<N, E, Ix> BoundedGraph<N, E, Ix> {
    pub fn new(
        graph: Graph<N, E, Ix>,
        in_edges: Vec<EdgeIndex<Ix>>,
        out_edges: Vec<EdgeIndex<Ix>>,
    ) -> Self {
        Self {
            graph,
            in_edges,
            out_edges,
        }
    }
}

impl<N, E, Ix: IndexType> Graph<N, E, Ix> {
    pub fn remove_subgraph(&mut self, cut: Cut<Ix>) {
        // remove existing subgraph
        let out_edges: HashSet<_> = cut.out_edges.iter().collect();
        let mut candidate_edges: VecDeque<_> = cut.in_edges.into();

        while let Some(e) = candidate_edges.pop_front() {
            if let Some(next_e) = self.next_edge(&e) {
                if !out_edges.contains(&next_e) {
                    candidate_edges.push_back(next_e);
                }
            }

            self.remove_node(self.edges[e.index()].node_ports[1].node);
        }
    }

    fn add_edges(&mut self, edges: Vec<(NodePort<Ix>, NodePort<Ix>, E)>) -> Vec<EdgeIndex<Ix>> {
        edges
            .into_iter()
            .map(|(a, b, weight)| self.add_edge(a, b, weight))
            .collect()
    }

    fn merge_edges(
        &mut self,
        left_edges: &Vec<EdgeIndex<Ix>>,
        right_edges: &Vec<EdgeIndex<Ix>>,
    ) -> Result<Vec<(NodePort<Ix>, NodePort<Ix>)>, &'static str> {
        left_edges
            .iter()
            .zip(right_edges.iter())
            .map(|(e_l, e_r)| -> Result<(NodePort<Ix>, NodePort<Ix>), &str> {
                // let left_edge = ;
                // let weight = left_edge.weight.take().ok_or("Left edge invalid.")?;
                Ok((
                    self.edges
                        .get(e_l.index())
                        .ok_or("Edge not found")?
                        .node_ports[0],
                    self.edges
                        .get(e_r.index())
                        .ok_or("Edge not found")?
                        .node_ports[1],
                    // weight,
                ))
            })
            .collect()
    }

    pub fn replace_with_identity(&mut self, cut: Cut<Ix>) -> Result<Vec<EdgeIndex<Ix>>, &str>
    where
        E: Debug,
    {
        if cut.in_edges.len() != cut.out_edges.len() {
            return Err("Boundary size mismatch.");
        }

        let new_edges = self.merge_edges(&cut.in_edges, &cut.out_edges)?;
        // dbg!(&new_edges);
        let new_edges = new_edges
            .iter()
            .zip(cut.out_edges.iter().map(|e| self.remove_edge(*e).unwrap()))
            .map(|((l, r), w)| (*l, *r, w))
            .collect();
        self.remove_subgraph(cut);

        let out = Ok(self.add_edges(new_edges));
        out
    }
    pub fn replace_subgraph(
        &mut self,
        cut: Cut<Ix>,
        mut replacement: BoundedGraph<N, E, Ix>,
    ) -> Result<Cut<Ix>, &str> {
        if cut.in_edges.len() != replacement.in_edges.len()
            || cut.out_edges.len() != replacement.out_edges.len()
        {
            return Err("Boundary size mismatch.");
        }

        let (_, edge_map) = self.insert_graph(replacement.graph);

        replacement.in_edges = replacement
            .in_edges
            .into_iter()
            .map(|e| edge_map[&e])
            .collect();
        replacement.out_edges = replacement
            .out_edges
            .into_iter()
            .map(|e| edge_map[&e])
            .collect();

        // let mut find_new_edges = |left_edges: &Vec<EdgeIndex<Ix>>,
        //                           right_edges: &Vec<EdgeIndex<Ix>>|
        //  -> Result<Vec<(NodePort<Ix>, NodePort<Ix>, E)>, &str> {
        //     left_edges
        //         .iter()
        //         .zip(right_edges.iter())
        //         .map(
        //             |(e_l, e_r)| -> Result<(NodePort<Ix>, NodePort<Ix>, E), &str> {
        //                 let left_edge = self.edges.get_mut(e_l.index()).ok_or("Edge not found")?;
        //                 let weight = left_edge.weight.take().ok_or("Left edge invalid.")?;
        //                 Ok((
        //                     left_edge.node_ports[0],
        //                     self.edges
        //                         .get(e_r.index())
        //                         .ok_or("Edge not found")?
        //                         .node_ports[1],
        //                     weight,
        //                 ))
        //             },
        //         )
        //         .collect()
        // };
        let new_in_edges = self.merge_edges(&cut.in_edges, &replacement.in_edges)?;
        let new_out_edges = self.merge_edges(&replacement.out_edges, &cut.out_edges)?;

        let new_in_edges = new_in_edges
            .iter()
            .zip(
                replacement
                    .in_edges
                    .iter()
                    .map(|e| self.remove_edge(*e).unwrap()),
            )
            .map(|((l, r), w)| (*l, *r, w))
            .collect();

        let new_out_edges = new_out_edges
            .iter()
            .zip(
                replacement
                    .out_edges
                    .iter()
                    .map(|e| self.remove_edge(*e).unwrap()),
            )
            .map(|((l, r), w)| (*l, *r, w))
            .collect();
        // let new_in_edges = cut
        //     .in_edges
        //     .iter()
        //     .zip(replacement.in_edges.iter())
        //     .map(
        //         |(e1, e2)| -> Result<(NodePort<Ix>, NodePort<Ix>, E), &str> {
        //             let e2 = edge_map[e2];
        //             let self_edge = self.edges.get_mut(e1.index()).ok_or("Edge not found")?;
        //             let weight = self_edge.weight.take().ok_or("Incoming edge invalid.")?;
        //             Ok((
        //                 self_edge.node_ports[0],
        //                 self.edges
        //                     .get(e2.index())
        //                     .ok_or("Edge not found")?
        //                     .node_ports[1],
        //                 weight,
        //             ))
        //         },
        //     )
        //     .collect::<Result<Vec<(NodePort<Ix>, NodePort<Ix>, E)>, &str>>()?;

        // let new_out_edges = cut
        //     .out_edges
        //     .iter()
        //     .zip(replacement.out_edges.iter())
        //     .map(
        //         |(e1, e2)| -> Result<(NodePort<Ix>, NodePort<Ix>, E), &str> {
        //             let e2 = edge_map[e2];
        //             let self_edge = self.edges.get_mut(e1.index()).ok_or("Edge not found")?;
        //             let weight = self_edge.weight.take().ok_or("Incoming edge invalid.")?;
        //             let target_port = self_edge.node_ports[1];
        //             Ok((
        //                 self.edges
        //                     .get(e2.index())
        //                     .ok_or("Edge not found")?
        //                     .node_ports[0],
        //                 target_port,
        //                 weight,
        //             ))
        //         },
        //     )
        //     .collect::<Result<Vec<(NodePort<Ix>, NodePort<Ix>, E)>, &str>>()?;

        // remove existing subgraph
        self.remove_subgraph(cut);

        // let mut add_edges = |edges: Vec<(NodePort<Ix>, NodePort<Ix>, E)>| -> Vec<EdgeIndex<Ix>> {
        //     edges
        //         .into_iter()
        //         .map(|(a, b, weight)| self.add_edge(a, b, weight))
        //         .collect()
        // };

        Ok(Cut::new(
            self.add_edges(new_in_edges),
            self.add_edges(new_out_edges),
        ))
    }
}
