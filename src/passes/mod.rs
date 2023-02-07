pub mod classical;
// pub mod redundancy;
pub mod mcts;
pub mod pattern;
pub mod squash;
pub mod taso;
// use rayon::prelude::*;

use crate::circuit::{
    circuit::{Circuit, CircuitRewrite},
    dag::{EdgeProperties, VertexProperties},
    operation::{Op, Param},
};

use self::pattern::{FixedStructPattern, Match, NodeCompClosure, PatternMatcher};
use portgraph::{
    graph::{NodeIndex, DIRECTIONS},
    substitute::{BoundedSubgraph, RewriteError, SubgraphRef},
};

pub trait RewriteGenerator<'s, T: Iterator<Item = CircuitRewrite> + 's> {
    fn rewrites<'a: 's>(&'s self, base_circ: &'a Circuit) -> T;
    fn into_rewrites(self, base_circ: &'s Circuit) -> T;
}

/// Repeatedly apply all available rewrites reported by finder closure until no more are found.
///
/// # Errors
///
/// This function will return an error if rewrite application fails.
pub fn apply_exhaustive<F>(mut circ: Circuit, finder: F) -> Result<(Circuit, bool), RewriteError>
where
    F: Fn(&Circuit) -> Vec<CircuitRewrite>,
{
    let mut success = false;
    loop {
        // assuming all the returned rewrites are non-overlapping
        // or filter to make them non-overlapping
        // then in theory, they can all be applied in parallel
        let rewrites = finder(&circ);
        if rewrites.is_empty() {
            break;
        }
        success = true;
        for rewrite in rewrites {
            circ.apply_rewrite(rewrite)?;
        }
    }

    Ok((circ, success))
}

/// Repeatedly apply first reported rewrite
///
/// # Errors
///
/// This function will return an error if rewrite application fails.
pub fn apply_greedy<F>(mut circ: Circuit, finder: F) -> Result<(Circuit, bool), RewriteError>
where
    F: Fn(&Circuit) -> Option<CircuitRewrite>,
{
    let mut success = false;
    while let Some(rewrite) = finder(&circ) {
        success |= true;
        circ.apply_rewrite(rewrite)?;
    }

    Ok((circ, success))
}

pub type CircFixedStructPattern<F> = FixedStructPattern<VertexProperties, EdgeProperties, F>;

impl<F> CircFixedStructPattern<F> {
    pub fn from_circ(pattern_circ: Circuit, node_comp_closure: F) -> Self {
        Self {
            boundary: pattern_circ.boundary(),
            graph: pattern_circ.dag,
            node_comp_closure,
        }
    }
}

pub struct PatternRewriter<F, G> {
    pattern: CircFixedStructPattern<F>,
    rewrite_closure: G,
}

impl<F, G> PatternRewriter<F, G> {
    pub fn new(pattern: CircFixedStructPattern<F>, rewrite_closure: G) -> Self {
        Self {
            pattern,
            rewrite_closure,
        }
    }
}

impl<'s, 'f: 's, F, G> RewriteGenerator<'s, CircRewriteIter<'s, F, G>> for PatternRewriter<F, G>
where
    F: NodeCompClosure<VertexProperties, EdgeProperties> + Clone + Send + Sync + 'f,
    G: Fn(Match) -> (Circuit, Param) + 's + Clone,
{
    fn into_rewrites(self, base_circ: &'s Circuit) -> CircRewriteIter<'s, F, G> {
        let ports = pattern_ports(&self.pattern);
        let matcher = PatternMatcher::new(self.pattern, base_circ.dag_ref());

        RewriteIter {
            match_iter: matcher.into_iter(),
            ports,
            rewrite_closure: self.rewrite_closure,
            circ: base_circ,
        }
    }

    fn rewrites<'a: 's>(&'s self, base_circ: &'a Circuit) -> CircRewriteIter<'s, F, G> {
        let ports = pattern_ports(&self.pattern);
        let matcher = PatternMatcher::new(self.pattern.clone(), base_circ.dag_ref());

        RewriteIter {
            match_iter: matcher.into_iter(),
            ports,
            rewrite_closure: self.rewrite_closure.clone(),
            circ: base_circ,
        }
    }
}

pub type CircRewriteIter<'a, F, G> = RewriteIter<'a, VertexProperties, EdgeProperties, F, G>;

// pub fn pattern_rewriter<'a, 'f: 'a, 'g: 'a, F, G>(
//     pattern: CircFixedStructPattern<F>,
//     circ: &'a Circuit,
//     rewrite_closure: G,
// ) -> RewriteIter<'a, VertexProperties, EdgeProperties, F, G>
// where
//     F: NodeCompClosure<VertexProperties, EdgeProperties> + Clone + 'f + Send + Sync,
//     G: Fn(Match) -> (Circuit, Param) + 'g,
// {
//     let pr = PatternRewriter {
//         pattern,
//         rewrite_closure,
//     };
//     pr.rewrites(circ)
// }
// pub fn pattern_rewriter<'a, 'f: 'a, 'g: 'a, F, G>(
//     pattern: CircFixedStructPattern<F>,
//     circ: &'a Circuit,
//     rewrite_closure: G,
// ) -> impl Iterator<Item = CircuitRewrite> + 'a
// where
//     F: NodeCompClosure<VertexProperties, EdgeProperties> + Clone + 'f + Send + Sync,
//     G: Fn(Match) -> (Circuit, Param) + 'g,
// {
//     // TODO when applying rewrites greedily, all of this construction needs to
//     // every time a match is found. Find a way to update the target of the match
//     // and restart matching without doing all this again.
//     let ports = pattern_ports(&pattern);
//     // let in_ports: Vec<_> = pattern
//     //     .graph
//     //     .neighbours(pattern.boundary[0], Direction::Outgoing)
//     //     .collect();
//     // let out_ports: Vec<_> = pattern
//     //     .graph
//     //     .neighbours(pattern.boundary[1], Direction::Incoming)
//     //     .collect();
//     let matcher = PatternMatcher::new(pattern, circ.dag_ref());

//     // matcher
//     // .into_iter()
//     // .map(match_to_rewrite(ports, circ, rewrite_closure))

//     RewriteIter {
//         match_iter: matcher.into_iter(),
//         ports,
//         rewrite_closure: &rewrite_closure,
//         circ,
//     }
// }

pub struct RewriteIter<'a, N, E, F, G> {
    match_iter: pattern::PatternMatchIter<'a, N, E, F>,
    ports: [Vec<(usize, NodeIndex)>; 2],
    rewrite_closure: G,
    circ: &'a Circuit,
}

impl<'a, N, E, F, G> Iterator for RewriteIter<'a, N, E, F, G>
where
    N: PartialEq,
    E: PartialEq,
    F: NodeCompClosure<N, E>,
    G: Fn(Match) -> (Circuit, Param),
{
    type Item = CircuitRewrite;

    fn next(&mut self) -> Option<Self::Item> {
        let pmatch = self.match_iter.next()?;

        let edges = DIRECTIONS.map(|direction| {
            self.ports[direction.index()]
                .iter()
                .map(|(p, n)| {
                    let mapped_n = *pmatch.get(n).unwrap();

                    self.circ
                        .edge_at_port(mapped_n, *p, direction)
                        .expect("Missing edge")
                })
                .collect()
        });

        let subg = BoundedSubgraph::new(SubgraphRef::from_iter(pmatch.values().copied()), edges);

        let (newcirc, phase) = (self.rewrite_closure)(pmatch);

        Some(CircuitRewrite::new(subg, newcirc.into(), phase))
    }
}

// fn match_to_rewrite<'a, 'g: 'a, G>(
//     ports: [Vec<(usize, NodeIndex)>; 2],
//     circ: &'a Circuit,
//     rewrite_closure: G,
// ) -> impl Fn(Match) -> CircuitRewrite + 'a
// where
//     G: Fn(Match) -> (Circuit, Param) + 'g,
// {
//     move |pmatch: Match| {
//         let edges = DIRECTIONS.map(|direction| {
//             ports[direction.index()]
//                 .iter()
//                 .map(|(p, n)| {
//                     let mapped_n = *pmatch.get(n).unwrap();

//                     circ.edge_at_port(mapped_n, *p, direction)
//                         .expect("Missing edge")
//                 })
//                 .collect()
//         });
//         // let in_edges: Vec<_> = in_ports
//         //     .iter()
//         //     .map(|np| {
//         //         circ.dag
//         //             .edge_at_port(
//         //                 NodePort::new(*pmatch.get(&np.node).unwrap(), np.port),
//         //                 Direction::Incoming,
//         //             )
//         //             .unwrap()
//         //     })
//         //     .collect();
//         // let out_edges: Vec<_> = out_ports
//         //     .iter()
//         //     .map(|np| {
//         //         circ.dag
//         //             .edge_at_port(
//         //                 NodePort::new(*pmatch.get(&np.node).unwrap(), np.port),
//         //                 Direction::Outgoing,
//         //             )
//         //             .unwrap()
//         //     })
//         // .collect();
//         let subg = BoundedSubgraph::new(SubgraphRef::from_iter(pmatch.values().copied()), edges);

//         let (newcirc, phase) = (rewrite_closure)(pmatch);

//         CircuitRewrite::new(subg, newcirc.into(), phase)
//     }
// }

// pub fn pattern_rewriter_parallel<'a, 'f: 'a, 'g: 'a, F, G>(
//     pattern: CircFixedStructPattern<F>,
//     circ: &'a Circuit,
//     rewrite_closure: G,
// ) -> impl ParallelIterator<Item = CircuitRewrite> + 'a
// where
//     F: NodeCompClosure<VertexProperties, EdgeProperties> + Clone + 'f + Send + Sync,
//     G: Fn(Match) -> (Circuit, Param) + 'g + Send + Sync,
// {
//     // TODO when applying rewrites greedily, all of this construction needs to
//     // every time a match is found. Find a way to update the target of the match
//     // and restart matching without doing all this again.
//     let ports = pattern_ports(&pattern);
//     // let in_ports: Vec<_> = pattern
//     //     .graph
//     //     .neighbours(pattern.boundary[0], Direction::Outgoing)
//     //     .collect();
//     // let out_ports: Vec<_> = pattern
//     //     .graph
//     //     .neighbours(pattern.boundary[1], Direction::Incoming)
//     //     .collect();
//     let matcher = PatternMatcher::new(pattern, circ.dag_ref());

//     matcher
//         .into_par_matches()
//         .map(match_to_rewrite(ports, circ, rewrite_closure))
// }

fn pattern_ports<'f, F>(pattern: &CircFixedStructPattern<F>) -> [Vec<(usize, NodeIndex)>; 2]
where
    F: NodeCompClosure<VertexProperties, EdgeProperties> + Clone + 'f + Send + Sync,
{
    DIRECTIONS.map(|direction| {
        pattern
            .graph
            .node_edges(pattern.boundary[direction.index()], direction.reverse())
            .map(|e| {
                let target = pattern
                    .graph
                    .edge_endpoint(e, direction)
                    .expect("missing edge");
                let port = pattern
                    .graph
                    .node_edges(target, direction)
                    .enumerate()
                    .find_map(|(i, e2)| (e == e2).then_some(i))
                    .expect("missing edge");

                (port, target)
            })
            // .map(|e| {
            //     pattern
            //         .graph
            //         .edge_endpoint(e, direction)
            //         .expect("dangling edge")
            // })
            .collect()
    })
}

// pub struct PatternRewrites<'p, I, F> {
//     match_iter: I,
//     pattern: CircFixedStructPattern<'p, F>,
//     boundary_ports: [Vec<NodePort>; 2],
// }

// impl<'p, I, F> PatternRewrites<'p, I, F> {
//     pub fn new(
//         match_iter: I,
//         pattern: CircFixedStructPattern<'p, F>,
//         boundary_ports: [Vec<NodePort>; 2],
//     ) -> Self {
//         Self {
//             match_iter,
//             pattern,
//             boundary_ports,
//         }
//     }
// }

// impl<I: Iterator<Item = Match<DefaultIx>>, F: Fn(NodeIndex, &VertexProperties) -> bool> Iterator
//     for PatternRewrites<'_, I, F>
// {
//     type Item = CircuitRewrite;

//     fn next(&mut self) -> Option<Self::Item> {
//         self.match_iter.next().map(|pmatch| {
//             let in_edges: Vec<_> = self.boundary_ports[0]
//                 .iter()
//                 .map(|np| {
//                     self.pattern
//                         .graph
//                         .edge_at_port(
//                             NodePort::new(*pmatch.get(&np.node).unwrap(), np.port),
//                             Direction::Incoming,
//                         )
//                         .unwrap()
//                 })
//                 .collect();
//             let out_edges: Vec<_> = self.boundary_ports[1]
//                 .iter()
//                 .map(|np| {
//                     self.pattern
//                         .graph
//                         .edge_at_port(
//                             NodePort::new(*pmatch.get(&np.node).unwrap(), np.port),
//                             Direction::Outgoing,
//                         )
//                         .unwrap()
//                 })
//                 .collect();
//             let subg = BoundedSubgraph::new(pmatch.values().cloned().into(), [in_edges, out_edges]);

//             let (newcirc, phase) = (rewrite_closure)(circ, pmatch);

//             CircuitRewrite::new(subg, newcirc.into(), phase)
//         })
//     }
// }
// #[cfg(test)]
// mod tests {
//     use symengine::Expression;

//     use crate::{
//         circuit::{
//             circuit::{Circuit, UnitID},
//             operation::{Op, Param},
//         },
//         graph::graph::PortIndex,
//     };
//     use tket_json_rs::circuit_json::SerialCircuit;

//     use super::redundancy::remove_redundancies;

//     #[test]
//     fn test_remove_redundancies() {
//         // circuit with only redundant gates; identity unitary
//         //[Rz(a) q[0];, Rz(-a) q[0];, CX q[0], q[1];, CX q[0], q[1];, Rx(2) q[1];]
//         let qubits = vec![
//             UnitID::Qubit {
//                 name: "q".into(),
//                 index: vec![0],
//             },
//             UnitID::Qubit {
//                 name: "q".into(),
//                 index: vec![0],
//             },
//         ];
//         let mut circ = Circuit::with_uids(qubits);

//         circ.append_op(Op::Rz(Param::from_str("a")), &vec![PortIndex::new(0)])
//             .unwrap();
//         circ.append_op(Op::Rz(Param::new("-a")), &vec![PortIndex::new(0)])
//             .unwrap();
//         circ.append_op(Op::CX, &vec![PortIndex::new(0), PortIndex::new(1)])
//             .unwrap();
//         circ.append_op(Op::CX, &vec![PortIndex::new(0), PortIndex::new(1)])
//             .unwrap();
//         circ.append_op(Op::Rx(Param::new("2.0")), &vec![PortIndex::new(1)])
//             .unwrap();

//         let circ2 = remove_redundancies(circ);

//         let _reser: SerialCircuit<Param> = circ2.into();

//         assert_eq!(_reser.commands.len(), 0);
//         // Rx(2pi) introduces a phase
//         assert_eq!(_reser.phase, Expression::new("1.0"));
//     }
// }

pub fn decompose_custom(circ: &Circuit) -> impl Iterator<Item = CircuitRewrite> + '_ {
    circ.dag.node_indices().filter_map(|n| {
        let op = &circ.dag.node_weight(n).unwrap().op;
        if let Op::Custom(x) = op {
            Some(CircuitRewrite::new(
                BoundedSubgraph::from_node(&circ.dag, n),
                x.to_circuit().expect("Circuit generation failed.").into(),
                0.0,
            ))
        } else {
            None
        }
    })
}

#[cfg(feature = "pyo3")]
use pyo3::prelude::pyfunction;

#[cfg_attr(feature = "pyo3", pyfunction)]
pub fn decompose_custom_pass(circ: Circuit) -> (Circuit, bool) {
    let (circ, suc) = apply_exhaustive(circ, |c| decompose_custom(c).collect()).unwrap();
    (circ, suc)
}
