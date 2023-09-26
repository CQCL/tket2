//! Rewriting strategies for circuit optimisation.
//!
//! This module contains the [`RewriteStrategy`] trait, which is currently
//! implemented by
//!  - [`GreedyRewriteStrategy`], which applies as many rewrites as possible
//!   on one circuit, and
//! - [`ExhaustiveRewriteStrategy`], which clones the original circuit as many
//!   times as there are possible rewrites and applies a different rewrite
//!   to every circuit.

use std::collections::HashSet;

use hugr::{ops::OpType, Hugr, HugrView, Node};
use itertools::Itertools;

use crate::{ops::op_matches, T2Op};

use super::CircuitRewrite;

/// Rewriting strategies for circuit optimisation.
///
/// A rewrite strategy takes a set of possible rewrites and applies them
/// to a circuit according to a strategy. It returns a list of new circuits,
/// each obtained by applying one or several non-overlapping rewrites to the
/// original circuit.
pub trait RewriteStrategy {
    /// Apply a set of rewrites to a circuit.
    fn apply_rewrites(
        &self,
        rewrites: impl IntoIterator<Item = CircuitRewrite>,
        circ: &Hugr,
    ) -> Vec<Hugr>;
}

/// A rewrite strategy applying as many non-overlapping rewrites as possible.
///
/// All possible rewrites are sorted by the number of gates they remove from
/// the circuit and are applied in order. If a rewrite overlaps with a rewrite
/// that has already been applied, it is skipped.
///
/// This strategy will always return exactly one circuit: the original circuit
/// with as many rewrites applied as possible.
///
/// Rewrites are only applied if they strictly decrease gate count.
pub struct GreedyRewriteStrategy;

impl RewriteStrategy for GreedyRewriteStrategy {
    #[tracing::instrument(skip_all)]
    fn apply_rewrites(
        &self,
        rewrites: impl IntoIterator<Item = CircuitRewrite>,
        circ: &Hugr,
    ) -> Vec<Hugr> {
        let rewrites = rewrites
            .into_iter()
            .sorted_by_key(|rw| rw.node_count_delta())
            .take_while(|rw| rw.node_count_delta() < 0);
        let mut changed_nodes = HashSet::new();
        let mut circ = circ.clone();
        for rewrite in rewrites {
            if rewrite
                .subcircuit()
                .nodes()
                .iter()
                .any(|n| changed_nodes.contains(n))
            {
                continue;
            }
            changed_nodes.extend(rewrite.subcircuit().nodes().iter().copied());
            rewrite
                .apply(&mut circ)
                .expect("Could not perform rewrite in greedy strategy");
        }
        vec![circ]
    }
}

/// A rewrite strategy that explores applying each rewrite to copies of the
/// circuit.
///
/// The parameter gamma controls how greedy the algorithm should be. It allows
/// a rewrite C1 -> C2 if C2 has at most gamma times the cost of C1:
///
/// $cost(C2) < gamma * cost(C1)$
///
/// The cost function is given by the number of operations in the circuit that
/// satisfy a given Op predicate. This allows for instance to use the total
/// number of gates (true predicate) or the number of CX gates as cost function.
///
/// gamma = 1 is the greedy strategy where a rewrite is only allowed if it
/// strictly reduces the gate count. The default is gamma = 1.0001 (as set in
/// the Quartz paper) and the number of CX gates. This essentially allows
/// rewrites that improve or leave the number of CX unchanged.
#[derive(Debug, Clone)]
pub struct ExhaustiveRewriteStrategy<P> {
    /// The gamma parameter.
    pub gamma: f64,
    /// Ops to count for cost function.
    pub op_predicate: P,
}

impl<P> ExhaustiveRewriteStrategy<P> {
    /// New exhaustive rewrite strategy with provided predicate.
    ///
    /// The gamma parameter is set to the default 1.0001.
    pub fn with_predicate(op_predicate: P) -> Self {
        Self {
            gamma: 1.0001,
            op_predicate,
        }
    }

    /// New exhaustive rewrite strategy with provided gamma and predicate.
    pub fn new(gamma: f64, op_predicate: P) -> Self {
        Self {
            gamma,
            op_predicate,
        }
    }
}

/// Exhaustive rewrite strategy with CX count cost function.
///
/// The gamma parameter is set to the default 1.0001. This is a good default
/// choice for NISQ-y circuits, where CX gates are the most expensive.
pub fn exhaustive_cx() -> ExhaustiveRewriteStrategy<fn(&OpType) -> bool> {
    ExhaustiveRewriteStrategy::with_predicate(is_cx)
}

/// Exhaustive rewrite strategy with CX count cost function and provided gamma.
pub fn exhaustive_cx_with_gamma(gamma: f64) -> ExhaustiveRewriteStrategy<fn(&OpType) -> bool> {
    ExhaustiveRewriteStrategy::new(gamma, is_cx)
}

impl<P: Fn(&OpType) -> bool> RewriteStrategy for ExhaustiveRewriteStrategy<P> {
    #[tracing::instrument(skip_all)]
    fn apply_rewrites(
        &self,
        rewrites: impl IntoIterator<Item = CircuitRewrite>,
        circ: &Hugr,
    ) -> Vec<Hugr> {
        rewrites
            .into_iter()
            .filter(|rw| {
                let old_count = pre_rewrite_cost(rw, circ, &self.op_predicate) as f64;
                let new_count = post_rewrite_cost(rw, circ, &self.op_predicate) as f64;
                new_count < old_count * self.gamma
            })
            .map(|rw| {
                let mut circ = circ.clone();
                rw.apply(&mut circ).expect("invalid pattern match");
                circ
            })
            .collect()
    }
}

fn is_cx(op: &OpType) -> bool {
    op_matches(op, T2Op::CX)
}

fn cost(
    nodes: impl IntoIterator<Item = Node>,
    circ: &Hugr,
    pred: impl Fn(&OpType) -> bool,
) -> usize {
    nodes
        .into_iter()
        .filter(|n| {
            let op = circ.get_optype(*n);
            pred(op)
        })
        .count()
}

fn pre_rewrite_cost(rw: &CircuitRewrite, circ: &Hugr, pred: impl Fn(&OpType) -> bool) -> usize {
    cost(rw.subcircuit().nodes().iter().copied(), circ, pred)
}

fn post_rewrite_cost(rw: &CircuitRewrite, circ: &Hugr, pred: impl Fn(&OpType) -> bool) -> usize {
    cost(rw.replacement().nodes(), circ, pred)
}

#[cfg(test)]
mod tests {
    use super::*;
    use hugr::{
        ops::{OpTag, OpTrait},
        Hugr, HugrView, Node,
    };
    use itertools::Itertools;

    use crate::{
        circuit::Circuit,
        rewrite::{CircuitRewrite, Subcircuit},
        utils::build_simple_circuit,
        T2Op,
    };

    fn n_cx(n_gates: usize) -> Hugr {
        let qbs = [0, 1];
        build_simple_circuit(2, |circ| {
            for _ in 0..n_gates {
                circ.append(T2Op::CX, qbs).unwrap();
            }
            Ok(())
        })
        .unwrap()
    }

    /// Rewrite cx_nodes -> empty
    fn rw_to_empty(hugr: &Hugr, cx_nodes: impl Into<Vec<Node>>) -> CircuitRewrite {
        let subcirc = Subcircuit::try_from_nodes(cx_nodes, hugr).unwrap();
        subcirc.create_rewrite(hugr, n_cx(0)).unwrap()
    }

    /// Rewrite cx_nodes -> 10x CX
    fn rw_to_full(hugr: &Hugr, cx_nodes: impl Into<Vec<Node>>) -> CircuitRewrite {
        let subcirc = Subcircuit::try_from_nodes(cx_nodes, hugr).unwrap();
        subcirc.create_rewrite(hugr, n_cx(10)).unwrap()
    }

    #[test]
    fn test_greedy_strategy() {
        let circ = n_cx(10);
        let cx_gates = circ
            .nodes()
            .filter(|&n| OpTag::Leaf.is_superset(circ.get_optype(n).tag()))
            .collect_vec();

        let rws = [
            rw_to_empty(&circ, cx_gates[0..2].to_vec()),
            rw_to_full(&circ, cx_gates[4..7].to_vec()),
            rw_to_empty(&circ, cx_gates[4..6].to_vec()),
            rw_to_empty(&circ, cx_gates[9..10].to_vec()),
        ];

        let strategy = GreedyRewriteStrategy;
        let rewritten = strategy.apply_rewrites(rws, &circ);
        assert_eq!(rewritten.len(), 1);
        assert_eq!(rewritten[0].num_gates(), 5);
    }

    #[test]
    fn test_exhaustive_default_strategy() {
        let circ = n_cx(10);
        let cx_gates = circ
            .nodes()
            .filter(|&n| OpTag::Leaf.is_superset(circ.get_optype(n).tag()))
            .collect_vec();

        let rws = [
            rw_to_empty(&circ, cx_gates[0..2].to_vec()),
            rw_to_full(&circ, cx_gates[4..7].to_vec()),
            rw_to_empty(&circ, cx_gates[4..8].to_vec()),
            rw_to_empty(&circ, cx_gates[9..10].to_vec()),
        ];

        let strategy = exhaustive_cx();
        let rewritten = strategy.apply_rewrites(rws, &circ);
        let exp_circ_lens = HashSet::from_iter([8, 6, 9]);
        let circ_lens: HashSet<_> = rewritten.iter().map(|c| c.num_gates()).collect();
        assert_eq!(circ_lens, exp_circ_lens);
    }

    #[test]
    fn test_exhaustive_generous_strategy() {
        let circ = n_cx(10);
        let cx_gates = circ
            .nodes()
            .filter(|&n| OpTag::Leaf.is_superset(circ.get_optype(n).tag()))
            .collect_vec();

        let rws = [
            rw_to_empty(&circ, cx_gates[0..2].to_vec()),
            rw_to_full(&circ, cx_gates[4..7].to_vec()),
            rw_to_empty(&circ, cx_gates[4..8].to_vec()),
            rw_to_empty(&circ, cx_gates[9..10].to_vec()),
        ];

        let strategy = exhaustive_cx_with_gamma(10.);
        let rewritten = strategy.apply_rewrites(rws, &circ);
        let exp_circ_lens = HashSet::from_iter([8, 17, 6, 9]);
        let circ_lens: HashSet<_> = rewritten.iter().map(|c| c.num_gates()).collect();
        assert_eq!(circ_lens, exp_circ_lens);
    }
}
