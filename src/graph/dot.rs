use std::fmt::Display;

use super::graph::{Graph, IndexType};

pub fn dot_string<N: Display, E: Display, Ix: IndexType>(graph: &Graph<N, E, Ix>) -> String {
    let mut s = String::new();

    s.push_str("digraph {\n");

    for n in graph.nodes() {
        let node = graph.node_weight(n).unwrap();
        s.push_str(&format!("{} [label=\"{:}\"]\n", n.index(), node)[..]);
    }

    for e in graph.edges() {
        let [a, b] = graph.edge_endpoints(e).unwrap();
        let edge = graph.edge_weight(e).unwrap();
        s.push_str(
            &format!(
                "{} -> {} [label=\"({}, {}); {}\"]\n",
                a.node.index(),
                b.node.index(),
                a.port.index(),
                b.port.index(),
                edge,
            )[..],
        );
    }
    s.push_str("}\n");
    s
}
