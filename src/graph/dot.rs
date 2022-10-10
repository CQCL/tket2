use std::fmt::Display;

use super::graph::{Direction, Graph};

pub fn dot_string<N: Display, E: Display>(graph: &Graph<N, E>) -> String {
    let mut s = String::new();

    s.push_str("digraph {\n");

    for n in graph.node_indices() {
        let node = graph.node_weight(n).unwrap();
        s.push_str(&format!("{} [label=\"{:}\"]\n", n.index(), node)[..]);
    }

    // TODO: Needs to handle dangling edges

    for e in graph.edge_indices() {
        let a = graph.edge_endpoint(e, Direction::Outgoing).unwrap();
        let b = graph.edge_endpoint(e, Direction::Incoming).unwrap();

        let edge = graph.edge_weight(e).unwrap();
        s.push_str(
            &format!(
                "{} -> {} [label=\"({}, {}); {}\"]\n",
                a.index(),
                b.index(),
                graph
                    .node_edges(a, Direction::Outgoing)
                    .enumerate()
                    .find(|(_, oe)| *oe == e)
                    .unwrap()
                    .0,
                graph
                    .node_edges(b, Direction::Incoming)
                    .enumerate()
                    .find(|(_, oe)| *oe == e)
                    .unwrap()
                    .0,
                // 0,
                // 0,
                // b.port.index(),
                edge,
            )[..],
        );
    }

    s.push_str("}\n");
    s
}
