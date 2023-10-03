//! Rewriting strategies for circuit optimisation.
//!
//! This module contains the [`RewriteStrategy`] trait, which is currently
//! implemented by
//! - [`GreedyRewriteStrategy`], which applies as many rewrites as possible
//!   on one circuit, and
//! - exhaustive strategies, which clone the original circuit and explore every
//!   possible rewrite (with some pruning strategy):
//!    - [`NonIncreasingGateCountStrategy`], which only considers rewrites that
//!      do not increase some cost function (e.g. cx gate count, implemented as
//!      [`NonIncreasingCXCountStrategy`]), and
//!    - [`ExhaustiveGammaStrategy`], which ignores rewrites that increase the
//!      cost function beyond a threshold given by a f64 parameter gamma.

use std::{collections::HashSet, fmt::Debug, iter::Sum};

use derive_more::From;
use hugr::{ops::OpType, Hugr, HugrView, Node};
use itertools::Itertools;

use crate::{ops::op_matches, Circuit, T2Op};

use super::CircuitRewrite;

/// Rewriting strategies for circuit optimisation.
///
/// A rewrite strategy takes a set of possible rewrites and applies them
/// to a circuit according to a strategy. It returns a list of new circuits,
/// each obtained by applying one or several non-overlapping rewrites to the
/// original circuit.
///
/// It also assign every circuit a totally ordered cost that can be used when
/// using rewrites for circuit optimisation.
pub trait RewriteStrategy {
    /// The circuit cost to be minised.
    type Cost: Ord;

    /// Apply a set of rewrites to a circuit.
    fn apply_rewrites(
        &self,
        rewrites: impl IntoIterator<Item = CircuitRewrite>,
        circ: &Hugr,
    ) -> Vec<Hugr>;

    /// The cost of a circuit.
    fn circuit_cost(&self, circ: &Hugr) -> Self::Cost;
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
    type Cost = usize;

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

    fn circuit_cost(&self, circ: &Hugr) -> Self::Cost {
        circ.num_gates()
    }
}

/// Exhaustive rewrite strategy allowing smaller or equal cost rewrites.
///
/// Rewrites are permitted based on a cost function called the major cost: if
/// the major cost of the target of the rewrite is smaller or equal to the major
/// cost of the pattern, the rewrite is allowed.
///
/// A second cost function, the minor cost, is used as a tie breaker: within
/// circuits with the same major cost, the circuit ordering prioritises circuits
/// with a smaller minor cost.
///
/// An example would be to use the number of CX gates as major cost and the
/// total number of gates as minor cost. Compared to a [`ExhaustiveGammaStrategy`],
/// that would only order circuits based on the number of CX gates, this creates
/// a less flat optimisation landscape.
#[derive(Debug, Clone)]
pub struct NonIncreasingGateCountStrategy<C1, C2> {
    major_cost: C1,
    minor_cost: C2,
}

impl<C1, C2> ExhaustiveThresholdStrategy for NonIncreasingGateCountStrategy<C1, C2>
where
    C1: Fn(&OpType) -> usize,
    C2: Fn(&OpType) -> usize,
{
    type OpCost = MajorMinorCost;
    type SumOpCost = MajorMinorCost;

    fn threshold(&self, pattern_cost: &Self::SumOpCost, target_cost: &Self::SumOpCost) -> bool {
        target_cost.major <= pattern_cost.major
    }

    fn op_cost(&self, op: &OpType) -> Self::OpCost {
        ((self.major_cost)(op), (self.minor_cost)(op)).into()
    }
}

/// Non-increasing rewrite strategy based on CX count.
///
/// The minor cost to break ties between equal CX counts is the number of
/// quantum gates.
pub type NonIncreasingCXCountStrategy =
    NonIncreasingGateCountStrategy<fn(&OpType) -> usize, fn(&OpType) -> usize>;

impl NonIncreasingCXCountStrategy {
    /// Create rewrite strategy based on non-increasing CX count.
    pub fn default_cx() -> Self {
        Self {
            major_cost: |op| is_cx(op) as usize,
            minor_cost: |op| is_quantum(op) as usize,
        }
    }
}

/// Exhaustive rewrite strategy allowing rewrites with bounded cost increase.
///
/// The parameter gamma controls how greedy the algorithm should be. It allows
/// a rewrite C1 -> C2 if C2 has at most gamma times the cost of C1:
///
/// $cost(C2) < gamma * cost(C1)$
///
/// The cost function is given by the sum of the cost of each operation in the
/// circuit. This allows for instance to use of the total number of gates (true
/// predicate), the number of CX gates or a weighted sum of gate types as cost
/// functions.
///
/// gamma = 1 is the greedy strategy where a rewrite is only allowed if it
/// strictly reduces the gate count. The default is gamma = 1.0001 (as set in
/// the Quartz paper) and the number of CX gates. This essentially allows
/// rewrites that improve or leave the number of CX unchanged.
#[derive(Debug, Clone)]
pub struct ExhaustiveGammaStrategy<C> {
    /// The gamma parameter.
    pub gamma: f64,
    /// A cost function for each operation.
    pub op_cost: C,
}

impl<C: Fn(&OpType) -> usize> ExhaustiveThresholdStrategy for ExhaustiveGammaStrategy<C> {
    type OpCost = usize;
    type SumOpCost = usize;

    fn threshold(&self, &pattern_cost: &Self::SumOpCost, &target_cost: &Self::SumOpCost) -> bool {
        (target_cost as f64) < self.gamma * (pattern_cost as f64)
    }

    fn op_cost(&self, op: &OpType) -> Self::OpCost {
        (self.op_cost)(op)
    }
}

impl<C> ExhaustiveGammaStrategy<C> {
    /// New exhaustive rewrite strategy with provided predicate.
    ///
    /// The gamma parameter is set to the default 1.0001.
    pub fn with_cost(op_cost: C) -> Self {
        Self {
            gamma: 1.0001,
            op_cost,
        }
    }

    /// New exhaustive rewrite strategy with provided gamma and predicate.
    pub fn new(gamma: f64, op_cost: C) -> Self {
        Self { gamma, op_cost }
    }
}

impl ExhaustiveGammaStrategy<fn(&OpType) -> usize> {
    /// Exhaustive rewrite strategy with CX count cost function.
    ///
    /// The gamma parameter is set to the default 1.0001. This is a good default
    /// choice for NISQ-y circuits, where CX gates are the most expensive.
    pub fn exhaustive_cx() -> Self {
        ExhaustiveGammaStrategy::with_cost(|op| is_cx(op) as usize)
    }

    /// Exhaustive rewrite strategy with CX count cost function and provided gamma.
    pub fn exhaustive_cx_with_gamma(gamma: f64) -> Self {
        ExhaustiveGammaStrategy::new(gamma, |op| is_cx(op) as usize)
    }
}

/// Exhaustive strategies based on cost functions and thresholds.
///
/// Every possible rewrite is applied to a copy of the input circuit. Thus for
/// one circuit, up to `n` rewritten circuits will be returned, each obtained
/// by applying one of the `n` rewrites to the original circuit.
///
/// Whether a rewrite is allowed or not is determined by a cost function and a
/// threshold function: if the cost of the target of the rewrite is below the
/// threshold given by the cost of the original circuit, the rewrite is
/// performed.
///
/// The cost function must return a value of type `Self::OpCost`. All op costs
/// are summed up to obtain a total cost that is then compared using the
/// threshold function.
pub trait ExhaustiveThresholdStrategy {
    /// The cost of a single operation.
    type OpCost;
    /// The sum of the cost of all operations in a circuit.
    type SumOpCost;

    /// Whether the rewrite is allowed or not, based on the cost of the pattern and target.
    fn threshold(&self, pattern_cost: &Self::SumOpCost, target_cost: &Self::SumOpCost) -> bool;

    /// The cost of a single operation.
    fn op_cost(&self, op: &OpType) -> Self::OpCost;
}

impl<T: ExhaustiveThresholdStrategy> RewriteStrategy for T
where
    T::SumOpCost: Sum<T::OpCost> + Ord,
{
    type Cost = T::SumOpCost;

    #[tracing::instrument(skip_all)]
    fn apply_rewrites(
        &self,
        rewrites: impl IntoIterator<Item = CircuitRewrite>,
        circ: &Hugr,
    ) -> Vec<Hugr> {
        rewrites
            .into_iter()
            .filter(|rw| {
                let pattern_cost = pre_rewrite_cost(rw, circ, |op| self.op_cost(op));
                let target_cost = post_rewrite_cost(rw, circ, |op| self.op_cost(op));
                self.threshold(&pattern_cost, &target_cost)
            })
            .map(|rw| {
                let mut circ = circ.clone();
                rw.apply(&mut circ).expect("invalid pattern match");
                circ
            })
            .collect()
    }

    fn circuit_cost(&self, circ: &Hugr) -> Self::Cost {
        cost(circ.nodes(), circ, |op| self.op_cost(op))
    }
}

/// A pair of major and minor cost.
///
/// This is used to order circuits based on major cost first, then minor cost.
/// A typical example would be CX count as major cost and total gate count as
/// minor cost.
#[derive(Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, From)]
pub struct MajorMinorCost {
    major: usize,
    minor: usize,
}

// Serialise as string so that it is easy to write to CSV
impl serde::Serialize for MajorMinorCost {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&format!("{:?}", self))
    }
}

impl Debug for MajorMinorCost {
    // TODO: A nicer print for the logs
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "(major={}, minor={})", self.major, self.minor)
    }
}

impl Sum for MajorMinorCost {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|a, b| (a.major + b.major, a.minor + b.minor).into())
            .unwrap_or_default()
    }
}

fn is_cx(op: &OpType) -> bool {
    op_matches(op, T2Op::CX)
}

fn is_quantum(op: &OpType) -> bool {
    let Ok(op): Result<T2Op, _> = op.try_into() else {
        return false;
    };
    op.is_quantum()
}

fn cost<C, T, S>(nodes: impl IntoIterator<Item = Node>, circ: &Hugr, op_cost: C) -> S
where
    C: Fn(&OpType) -> T,
    S: Sum<T>,
{
    nodes
        .into_iter()
        .map(|n| {
            let op = circ.get_optype(n);
            op_cost(op)
        })
        .sum()
}

fn pre_rewrite_cost<C, T, S>(rw: &CircuitRewrite, circ: &Hugr, pred: C) -> S
where
    C: Fn(&OpType) -> T,
    S: Sum<T>,
{
    cost(rw.subcircuit().nodes().iter().copied(), circ, pred)
}

fn post_rewrite_cost<C, T, S>(rw: &CircuitRewrite, circ: &Hugr, pred: C) -> S
where
    C: Fn(&OpType) -> T,
    S: Sum<T>,
{
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

        let strategy = ExhaustiveGammaStrategy::exhaustive_cx();
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

        let strategy = ExhaustiveGammaStrategy::exhaustive_cx_with_gamma(10.);
        let rewritten = strategy.apply_rewrites(rws, &circ);
        let exp_circ_lens = HashSet::from_iter([8, 17, 6, 9]);
        let circ_lens: HashSet<_> = rewritten.iter().map(|c| c.num_gates()).collect();
        assert_eq!(circ_lens, exp_circ_lens);
    }

    #[test]
    fn test_exhaustive_default_cx_cost() {
        let strat = NonIncreasingCXCountStrategy::default_cx();
        let circ = n_cx(3);
        assert_eq!(strat.circuit_cost(&circ), (3, 3).into());
        let circ = build_simple_circuit(2, |circ| {
            circ.append(T2Op::CX, [0, 1])?;
            circ.append(T2Op::X, [0])?;
            circ.append(T2Op::X, [1])?;
            Ok(())
        })
        .unwrap();
        assert_eq!(strat.circuit_cost(&circ), (1, 3).into());
    }

    #[test]
    fn test_exhaustive_default_cx_threshold() {
        let strat = NonIncreasingCXCountStrategy::default_cx();
        assert!(strat.threshold(&(3, 0).into(), &(3, 0).into()));
        assert!(strat.threshold(&(3, 0).into(), &(3, 5).into()));
        assert!(!strat.threshold(&(3, 10).into(), &(4, 0).into()));
        assert!(strat.threshold(&(3, 0).into(), &(1, 5).into()));
    }
}
