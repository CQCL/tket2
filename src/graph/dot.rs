use std::fmt::Display;

use super::graph::{Graph, DIRECTIONS};

pub fn dot_string<N: Display, E: Display>(graph: &Graph<N, E>) -> String {
    let mut s = String::new();

    s.push_str("digraph {\n");

    for n in graph.node_indices() {
        let node = graph.node_weight(n).unwrap();
        s.push_str(&format!("{} [label=\"{:}\"]\n", n.index(), node)[..]);
    }

    let mut dangle_node_index = 0;
    for e in graph.edge_indices() {
        add_edge_str(graph, e, &mut s, &mut dangle_node_index);
    }

    s.push_str("}\n");
    s
}

fn add_edge_str<N: Display, E: Display>(
    graph: &Graph<N, E>,
    e: super::graph::EdgeIndex,
    dot_s: &mut String,
    node_count: &mut usize,
) {
    let [(b, bp), (a, ap)] = DIRECTIONS.map(|dir| {
        if let Some(n) = graph.edge_endpoint(e, dir) {
            (
                format!("{}", n.index()),
                format!(
                    "{}",
                    graph
                        .node_edges(n, dir)
                        .enumerate()
                        .find(|(_, oe)| *oe == e)
                        .unwrap()
                        .0
                ),
            )
        } else {
            *node_count += 1;
            let node_id = format!("_{}", *node_count - 1);
            dot_s.push_str(&format!("{} [shape=point label=\"\"]\n", &node_id)[..]);

            (node_id, "".into())
        }
    });

    let edge = graph.edge_weight(e).unwrap();
    let edge_s = format!("{} -> {} [label=\"({}, {}); {}\"]\n", a, b, ap, bp, edge);
    dot_s.push_str(&edge_s[..])
}
