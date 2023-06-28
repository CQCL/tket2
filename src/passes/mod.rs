// pub mod classical;
// pub mod redundancy;
// pub mod pattern;
// pub mod squash;
pub mod taso;
// use rayon::prelude::*;

// use crate::circuit::{
//     circuit::{Circuit, CircuitRewrite},
//     dag::{EdgeProperties, VertexProperties},
//     operation::{Op, Param},
// };

// use self::pattern::{FixedStructPattern, Match, NodeCompClosure, PatternMatcher};
// use portgraph::{
//     graph::{NodeIndex, DIRECTIONS},
//     substitute::{BoundedSubgraph, RewriteError, SubgraphRef},
// };

// pub trait RewriteGenerator<'s, T: Iterator<Item = CircuitRewrite> + 's> {
//     fn rewrites<'a: 's>(&'s self, base_circ: &'a Circuit) -> T;
//     fn into_rewrites(self, base_circ: &'s Circuit) -> T;
// }

// /// Repeatedly apply all available rewrites reported by finder closure until no more are found.
// ///
// /// # Errors
// ///
// /// This function will return an error if rewrite application fails.
// pub fn apply_exhaustive<F>(mut circ: Circuit, finder: F) -> Result<(Circuit, bool), RewriteError>
// where
//     F: Fn(&Circuit) -> Vec<CircuitRewrite>,
// {
//     let mut success = false;
//     loop {
//         // assuming all the returned rewrites are non-overlapping
//         // or filter to make them non-overlapping
//         // then in theory, they can all be applied in parallel
//         let rewrites = finder(&circ);
//         if rewrites.is_empty() {
//             break;
//         }
//         success = true;
//         for rewrite in rewrites {
//             circ.apply_rewrite(rewrite)?;
//         }
//     }

//     Ok((circ, success))
// }

// /// Repeatedly apply first reported rewrite
// ///
// /// # Errors
// ///
// /// This function will return an error if rewrite application fails.
// pub fn apply_greedy<F>(mut circ: Circuit, finder: F) -> Result<(Circuit, bool), RewriteError>
// where
//     F: Fn(&Circuit) -> Option<CircuitRewrite>,
// {
//     let mut success = false;
//     while let Some(rewrite) = finder(&circ) {
//         success |= true;
//         circ.apply_rewrite(rewrite)?;
//     }

//     Ok((circ, success))
// }

// pub type CircFixedStructPattern<F> = FixedStructPattern<VertexProperties, EdgeProperties, F>;

// impl<F> CircFixedStructPattern<F> {
//     pub fn from_circ(pattern_circ: Circuit, node_comp_closure: F) -> Self {
//         Self {
//             boundary: pattern_circ.boundary(),
//             graph: pattern_circ.dag,
//             node_comp_closure,
//         }
//     }
// }

// pub struct PatternRewriter<F, G> {
//     pattern: CircFixedStructPattern<F>,
//     rewrite_closure: G,
// }

// impl<F, G> PatternRewriter<F, G> {
//     pub fn new(pattern: CircFixedStructPattern<F>, rewrite_closure: G) -> Self {
//         Self {
//             pattern,
//             rewrite_closure,
//         }
//     }
// }

// impl<'s, 'f: 's, F, G> RewriteGenerator<'s, CircRewriteIter<'s, F, G>> for PatternRewriter<F, G>
// where
//     F: NodeCompClosure<VertexProperties, EdgeProperties> + Clone + Send + Sync + 'f,
//     G: Fn(Match) -> (Circuit, Param) + 's + Clone,
// {
//     fn into_rewrites(self, base_circ: &'s Circuit) -> CircRewriteIter<'s, F, G> {
//         let ports = pattern_ports(&self.pattern);
//         let matcher = PatternMatcher::new(self.pattern, base_circ.dag_ref());

//         RewriteIter {
//             match_iter: matcher.into_iter(),
//             ports,
//             rewrite_closure: self.rewrite_closure,
//             circ: base_circ,
//         }
//     }

//     fn rewrites<'a: 's>(&'s self, base_circ: &'a Circuit) -> CircRewriteIter<'s, F, G> {
//         let ports = pattern_ports(&self.pattern);
//         let matcher = PatternMatcher::new(self.pattern.clone(), base_circ.dag_ref());

//         RewriteIter {
//             match_iter: matcher.into_iter(),
//             ports,
//             rewrite_closure: self.rewrite_closure.clone(),
//             circ: base_circ,
//         }
//     }
// }

// pub type CircRewriteIter<'a, F, G> = RewriteIter<'a, VertexProperties, EdgeProperties, F, G>;



// pub fn decompose_custom(circ: &Circuit) -> impl Iterator<Item = CircuitRewrite> + '_ {
//     circ.dag.node_indices().filter_map(|n| {
//         let op = &circ.dag.node_weight(n).unwrap().op;
//         if let Op::Custom(x) = op {
//             Some(CircuitRewrite::new(
//                 BoundedSubgraph::from_node(&circ.dag, n),
//                 x.to_circuit().expect("Circuit generation failed.").into(),
//                 0.0,
//             ))
//         } else {
//             None
//         }
//     })
// }

// #[cfg(feature = "pyo3")]
// use pyo3::prelude::pyfunction;

// #[cfg_attr(feature = "pyo3", pyfunction)]
// pub fn decompose_custom_pass(circ: Circuit) -> (Circuit, bool) {
//     let (circ, suc) = apply_exhaustive(circ, |c| decompose_custom(c).collect()).unwrap();
//     (circ, suc)
// }
