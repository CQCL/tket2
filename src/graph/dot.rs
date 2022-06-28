use std::fmt::Display;

use super::graph::Graph;

pub fn dot_string<N: Display, E: Display>(graph: &Graph<N, E>) -> String {
    let mut s = String::new();

    s.push_str("digraph {\n");

    for n in graph.node_indices() {
        let node = graph.node_weight(n).unwrap();
        s.push_str(&format!("{} [label=\"{:}\"]\n", n.index(), node)[..]);
    }

    for e in graph.edge_indices() {
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
