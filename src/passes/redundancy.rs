use std::collections::HashSet;

use crate::circuit::circuit::{Circuit, CircuitRewrite};
use crate::circuit::dag::{Edge, EdgeProperties, Vertex, VertexProperties, DAG};
use crate::circuit::operation::{Op, Param};
use portgraph::graph::{Direction, EdgeIndex};
use portgraph::substitute::BoundedSubgraph;

fn get_boundary(dag: &DAG, node: Vertex, direction: Direction) -> Vec<Edge> {
    dag.node_edges(node, direction).cloned().collect()
}
fn get_weights(dag: &DAG, edges: &Vec<EdgeIndex>) -> Vec<EdgeProperties> {
    edges
        .iter()
        .map(|e| dag.edge_weight(*e).unwrap().clone())
        .collect()
}

fn add_neighbours(dag: &DAG, preds: &Vec<Edge>, succs: &Vec<Edge>, set: &mut HashSet<Vertex>) {
    for (it, i) in [(preds.iter(), 0), (succs.iter(), 1)] {
        for e in it {
            set.insert(dag.edge_endpoints(*e).unwrap()[i].node);
        }
    }
}

// assumes no wire swaps
fn identity(edge_weights: Vec<EdgeProperties>) -> Circuit {
    let mut circ = Circuit::new();
    let [i, o] = circ.boundary();
    for (p, w) in edge_weights.into_iter().enumerate() {
        let noop = circ.add_vertex(Op::Noop);
        circ.add_edge((i, p as u8), (noop, 0), w.edge_type);
        circ.add_edge((noop, 0), (o, p as u8), w.edge_type);
    }

    circ
}

// A version of the redundancy removal in TKET but with only identity and dagger removal
pub fn remove_redundancies(mut circ: Circuit) -> Circuit {
    let mut candidate_nodes: HashSet<_> = circ.dag.nodes().collect();

    while !candidate_nodes.is_empty() {
        let candidate = candidate_nodes
            .take(&candidate_nodes.iter().next().cloned().unwrap())
            .unwrap();
        let op = match circ.dag.node_weight(candidate) {
            None => continue,
            Some(VertexProperties { op }) => match op {
                Op::Input | Op::Output => continue,
                _ => op,
            },
        };

        if !matches!(op, Op::Noop) {
            if let Some(phase) = op.identity_up_to_phase() {
                let preds: Vec<_> = get_boundary(&circ.dag, candidate, Direction::Incoming);
                let succs: Vec<_> = get_boundary(&circ.dag, candidate, Direction::Outgoing);

                add_neighbours(&circ.dag, &preds, &succs, &mut candidate_nodes);

                let new_weights = get_weights(&circ.dag, &preds);
                circ.apply_rewrite(CircuitRewrite::new(
                    BoundedSubgraph::from_node(&circ.dag, candidate),
                    identity(new_weights).into(),
                    Param::from(phase),
                ))
                .unwrap();
                continue;
            }
        }

        let kids: HashSet<_> = circ
            .dag
            .node_edges(candidate, Direction::Outgoing)
            .filter_map(|e| {
                let [start, end] = circ.dag.edge_endpoints(*e).unwrap();
                if start.port == end.port {
                    Some(end.node)
                } else {
                    None
                }
            })
            .collect();

        if kids.len() != 1 {
            continue;
        }

        let kid = *kids.iter().next().unwrap();

        if let Some(dagged) = circ.dag.node_weight(kid).unwrap().op.dagger() {
            if op != &dagged {
                continue;
            }

            let preds: Vec<_> = get_boundary(&circ.dag, candidate, Direction::Incoming);
            let succs: Vec<_> = get_boundary(&circ.dag, kid, Direction::Outgoing);

            let new_weights = get_weights(&circ.dag, &preds);
            add_neighbours(&circ.dag, &preds, &succs, &mut candidate_nodes);
            circ.apply_rewrite(CircuitRewrite::new(
                BoundedSubgraph::new([candidate, kid].into_iter().into(), [preds, succs]),
                identity(new_weights).into(),
                Param::from(0.0),
            ))
            .unwrap();
            continue;
        }
    }

    circ.remove_noop()
}
