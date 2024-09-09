//! Rewriting strategies for circuit optimisation.
//!
//! This module contains the [`RewriteStrategy`] trait, which is currently
//! implemented by

use std::iter::{self, Sum};
use std::ops::{Add, AddAssign, SubAssign};
use std::{collections::HashSet, fmt::Debug};

use derive_more::From;
use hugr::hugr::hugrmut::HugrMut;
use hugr::ops::OpType;
use hugr::HugrView;
use itertools::Itertools;

use crate::circuit::cost::{is_cx, CircuitCost, CostDelta, LexicographicCost};
use crate::circuit::ToTk2OpIter;
#[cfg(feature = "portmatching")]
use crate::static_circ::{BoxedStaticRewrite, StaticSizeCircuit};
use crate::{Circuit, Tk2Op};

use super::trace::RewriteTrace;
use super::CircuitRewrite;

/// A possible rewrite result returned by a rewrite strategy.
#[derive(Debug, Clone)]
pub struct RewriteResult<Circuit, C: CircuitCost> {
    /// The rewritten circuit.
    pub circ: Circuit,
    /// The cost delta of the rewrite.
    pub cost_delta: C::CostDelta,
}

impl<Circuit: Clone, C: CircuitCost> From<(Circuit, C::CostDelta)> for RewriteResult<Circuit, C> {
    #[inline]
    fn from((circ, cost_delta): (Circuit, C::CostDelta)) -> Self {
        Self {
            circ: circ.to_owned(),
            cost_delta,
        }
    }
}

/// Cost function definitions required in exhaustive strategies.
///
/// See [`ExhaustiveThresholdStrategy`], [`ExhaustiveGreedyStrategy`].
pub trait StrategyCost {
    /// The cost of a single operation.
    type OpCost: Copy + Ord + AddAssign + SubAssign + Sum + Default + std::fmt::Debug;

    /// An estimate rewrite's value given its cost.
    ///
    /// Return `None` if the rewrite should not be considered valuable (salient),
    /// otherwise return its value as a positive integer.
    fn value(&self, cost: &Self::OpCost) -> Option<usize>;

    /// The cost of a single operation.
    fn op_cost(&self, op: Tk2Op) -> Self::OpCost;

    /// The cost of a circuit using this strategy's cost function.
    fn circuit_cost(&self, circ: &impl ToTk2OpIter) -> Self::OpCost {
        circ.tk2_ops().map(|op| self.op_cost(op)).sum()
    }
}

/// Rewrite strategy cost allowing smaller or equal cost rewrites.
///
/// Rewrites are permitted based on a coarse cost function: if
/// the coarse cost of the target of the rewrite is smaller or equal to the
/// coarse cost of the pattern, the rewrite is allowed.
///
/// Further more fine-grained cost functions can be used as tie breakers: within
/// circuits with the same coarse cost, circuits are ranked according to a
/// lexicographic ordering of their cost functions.
///
/// An example would be to use the number of CX gates as coarse cost and the
/// total number of gates as the only fine grained cost function.
///
/// Lexicographic orderings may be useful to add relief to an otherwise flat
/// optimisation landscape.
#[derive(Debug, Clone)]
pub struct LexicographicCostFunction<F, const N: usize> {
    cost_fns: [F; N],
}

impl<F, const N: usize> StrategyCost for LexicographicCostFunction<F, N>
where
    F: Fn(Tk2Op) -> usize,
{
    type OpCost = LexicographicCost<isize, N>;

    #[inline]
    fn op_cost(&self, op: Tk2Op) -> Self::OpCost {
        let mut costs = [0; N];
        for (cost_fn, cost_mut) in self.cost_fns.iter().zip(&mut costs) {
            *cost_mut = cost_fn(op) as isize;
        }
        costs.into()
    }

    fn value(&self, cost: &Self::OpCost) -> Option<usize> {
        if cost.msb() < &0 {
            return Some(-cost.msb() as usize);
        } else {
            None
        }
    }
}

impl LexicographicCostFunction<fn(Tk2Op) -> usize, 2> {
    /// Non-increasing rewrite strategy based on CX count.
    ///
    /// A fine-grained cost function given by the total number of quantum gates
    /// is used to rank circuits with equal CX count.
    ///
    /// This is probably a good default for NISQ-y circuit optimisation.
    #[inline]
    pub fn default_cx() -> Self {
        Self {
            cost_fns: [|op| is_cx(op) as usize, |op| op.is_quantum() as usize],
        }
        .into()
    }
}

/// Rewrite strategy cost allowing rewrites with bounded cost increase.
///
/// The parameter gamma controls how greedy the algorithm should be. It allows a
/// rewrite C1 -> C2 if C2 has at most gamma times the cost of C1:
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
pub struct GammaStrategyCost<C> {
    /// The gamma parameter.
    pub gamma: f64,
    /// A cost function for each operation.
    pub op_cost: C,
}

impl<C: Fn(Tk2Op) -> usize> StrategyCost for GammaStrategyCost<C> {
    type OpCost = isize;

    #[inline]
    fn op_cost(&self, op: Tk2Op) -> Self::OpCost {
        (self.op_cost)(op) as isize
    }

    fn value(&self, cost: &Self::OpCost) -> Option<usize> {
        if cost < &0 {
            Some(-cost as usize)
        } else {
            None
        }
    }
}

impl<C> GammaStrategyCost<C> {
    /// New exhaustive rewrite strategy with provided predicate.
    ///
    /// The gamma parameter is set to the default 1.0001.
    #[inline]
    pub fn with_cost(op_cost: C) -> Self {
        Self {
            gamma: 1.0001,
            op_cost,
        }
    }

    /// New exhaustive rewrite strategy with provided gamma and predicate.
    #[inline]
    pub fn new(gamma: f64, op_cost: C) -> Self {
        Self { gamma, op_cost }
    }
}

impl GammaStrategyCost<fn(Tk2Op) -> usize> {
    /// Exhaustive rewrite strategy with CX count cost function.
    ///
    /// The gamma parameter is set to the default 1.0001. This is a good default
    /// choice for NISQ-y circuits, where CX gates are the most expensive.
    #[inline]
    pub fn exhaustive_cx() -> Self {
        GammaStrategyCost::with_cost(|op| is_cx(op) as usize)
    }

    /// Exhaustive rewrite strategy with CX count cost function and provided gamma.
    #[inline]
    pub fn exhaustive_cx_with_gamma(gamma: f64) -> Self {
        GammaStrategyCost::new(gamma, |op| is_cx(op) as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hugr::Node;
    use itertools::Itertools;

    use crate::rewrite::trace::REWRITE_TRACING_ENABLED;
    use crate::{
        circuit::Circuit,
        rewrite::{CircuitRewrite, Subcircuit},
        utils::build_simple_circuit,
        Tk2Op,
    };

    fn n_cx(n_gates: usize) -> Circuit {
        let qbs = [0, 1];
        build_simple_circuit(2, |circ| {
            for _ in 0..n_gates {
                circ.append(Tk2Op::CX, qbs).unwrap();
            }
            Ok(())
        })
        .unwrap_or_else(|e| panic!("{}", e))
    }

    /// Rewrite cx_nodes -> empty
    fn rw_to_empty(circ: &Circuit, cx_nodes: impl Into<Vec<Node>>) -> CircuitRewrite {
        let subcirc = Subcircuit::try_from_nodes(cx_nodes, circ).unwrap();
        subcirc
            .create_rewrite(circ, n_cx(0))
            .unwrap_or_else(|e| panic!("{}", e))
    }

    /// Rewrite cx_nodes -> 10x CX
    fn rw_to_full(circ: &Circuit, cx_nodes: impl Into<Vec<Node>>) -> CircuitRewrite {
        let subcirc = Subcircuit::try_from_nodes(cx_nodes, circ).unwrap();
        subcirc
            .create_rewrite(circ, n_cx(10))
            .unwrap_or_else(|e| panic!("{}", e))
    }

    // #[test]
    // fn test_greedy_strategy() {
    //     let mut circ = n_cx(10);
    //     let cx_gates = circ.commands().map(|cmd| cmd.node()).collect_vec();

    //     assert_eq!(circ.rewrite_trace(), None);
    //     circ.enable_rewrite_tracing();
    //     match REWRITE_TRACING_ENABLED {
    //         true => assert_eq!(circ.rewrite_trace(), Some(vec![])),
    //         false => assert_eq!(circ.rewrite_trace(), None),
    //     }

    //     let rws = [
    //         rw_to_empty(&circ, cx_gates[0..2].to_vec()),
    //         rw_to_full(&circ, cx_gates[4..7].to_vec()),
    //         rw_to_empty(&circ, cx_gates[4..6].to_vec()),
    //         rw_to_empty(&circ, cx_gates[9..10].to_vec()),
    //     ];

    //     let strategy = GreedyRewriteStrategy;
    //     let rewritten = strategy.apply_rewrites(rws, &circ).collect_vec();
    //     assert_eq!(rewritten.len(), 1);
    //     assert_eq!(rewritten[0].circ.num_operations(), 5);

    //     if REWRITE_TRACING_ENABLED {
    //         assert_eq!(rewritten[0].circ.rewrite_trace().unwrap().len(), 3);
    //     }
    // }

    // #[test]
    // fn test_exhaustive_default_strategy() {
    //     let mut circ = n_cx(10);
    //     let cx_gates = circ.commands().map(|cmd| cmd.node()).collect_vec();
    //     circ.enable_rewrite_tracing();

    //     let rws = [
    //         rw_to_empty(&circ, cx_gates[0..2].to_vec()),
    //         rw_to_full(&circ, cx_gates[4..7].to_vec()),
    //         rw_to_empty(&circ, cx_gates[4..8].to_vec()),
    //         rw_to_empty(&circ, cx_gates[9..10].to_vec()),
    //     ];

    //     let strategy = LexicographicCostFunction::default_cx();
    //     let rewritten = strategy.apply_rewrites(rws, &circ).collect_vec();
    //     let exp_circ_lens = HashSet::from_iter([3, 7, 9]);
    //     let circ_lens: HashSet<_> = rewritten.iter().map(|r| r.circ.num_operations()).collect();
    //     assert_eq!(circ_lens, exp_circ_lens);

    //     if REWRITE_TRACING_ENABLED {
    //         // Each strategy branch applies a single rewrite, composed of
    //         // multiple individual elements from `rws`.
    //         assert_eq!(
    //             rewritten[0].circ.rewrite_trace().unwrap(),
    //             vec![RewriteTrace::new(3)]
    //         );
    //         assert_eq!(
    //             rewritten[1].circ.rewrite_trace().unwrap(),
    //             vec![RewriteTrace::new(2)]
    //         );
    //         assert_eq!(
    //             rewritten[2].circ.rewrite_trace().unwrap(),
    //             vec![RewriteTrace::new(1)]
    //         );
    //     }
    // }

    // #[test]
    // fn test_exhaustive_gamma_strategy() {
    //     let circ = n_cx(10);
    //     let cx_gates = circ.commands().map(|cmd| cmd.node()).collect_vec();

    //     let rws = [
    //         rw_to_empty(&circ, cx_gates[0..2].to_vec()),
    //         rw_to_full(&circ, cx_gates[4..7].to_vec()),
    //         rw_to_empty(&circ, cx_gates[4..8].to_vec()),
    //         rw_to_empty(&circ, cx_gates[9..10].to_vec()),
    //     ];

    //     let strategy = GammaStrategyCost::exhaustive_cx_with_gamma(10.);
    //     let rewritten = strategy.apply_rewrites(rws, &circ);
    //     let exp_circ_lens = HashSet::from_iter([8, 17, 6, 9]);
    //     let circ_lens: HashSet<_> = rewritten.map(|r| r.circ.num_operations()).collect();
    //     assert_eq!(circ_lens, exp_circ_lens);
    // }

    #[test]
    fn test_exhaustive_default_cx_cost() {
        let strat = LexicographicCostFunction::default_cx();
        let circ = n_cx(3);
        assert_eq!(strat.circuit_cost(&circ), (3, 3).into());
        let circ = build_simple_circuit(2, |circ| {
            circ.append(Tk2Op::CX, [0, 1])?;
            circ.append(Tk2Op::X, [0])?;
            circ.append(Tk2Op::X, [1])?;
            Ok(())
        })
        .unwrap();
        assert_eq!(strat.circuit_cost(&circ), (1, 3).into());
    }

    // #[test]
    // fn test_exhaustive_default_cx_threshold() {
    //     let strat = LexicographicCostFunction::default_cx();
    //     assert!(strat.under_threshold(&(3, 0).into(), &(3, 0).into()));
    //     assert!(strat.under_threshold(&(3, 0).into(), &(3, 5).into()));
    //     assert!(!strat.under_threshold(&(3, 10).into(), &(4, 0).into()));
    //     assert!(strat.under_threshold(&(3, 0).into(), &(1, 5).into()));
    // }
}
