//! Constraints for port matching.
use std::collections::BTreeSet;

use super::{indexing::PatternOpPosition, predicate::Predicate};

use itertools::Itertools;
use portmatching as pm;

pub type Constraint = pm::Constraint<PatternOpPosition, Predicate>;

pub(super) fn constraint_key(c: &Constraint) -> (&PatternOpPosition, &Predicate) {
    let arg = match c.predicate() {
        Predicate::Link { .. } => c.required_bindings().iter().max().unwrap(),
        Predicate::IsOp { .. } => c.required_bindings().first().unwrap(),
        Predicate::SameOp { .. } => c.required_bindings().first().unwrap(),
        Predicate::DistinctQubits { .. } => c.required_bindings().first().unwrap(),
    };
    (arg, c.predicate())
}

impl pm::ToConstraintsTree<PatternOpPosition> for Predicate {
    fn to_constraints_tree(constraints: Vec<Constraint>) -> pm::MutuallyExclusiveTree<Constraint> {
        let constraints = constraints
            .into_iter()
            .enumerate()
            .map(|(i, c)| (c, i))
            .sorted_by(|(c1, _), (c2, _)| constraint_key(c1).cmp(&constraint_key(c2)))
            .collect_vec();
        let Some((first, _)) = constraints.first().cloned() else {
            return pm::MutuallyExclusiveTree::new();
        };
        match first.predicate() {
            Predicate::Link { .. } | Predicate::IsOp { .. } => {
                pm::MutuallyExclusiveTree::with_transitive_mutex(constraints, |a, b| {
                    match (a.predicate(), b.predicate()) {
                        (Predicate::IsOp { .. }, Predicate::IsOp { .. }) => {
                            fst_required_binding_eq(a, b)
                        }
                        (
                            Predicate::Link { out_port: lp_a, .. },
                            Predicate::Link { out_port: lp_b, .. },
                        ) => lp_a == lp_b && fst_required_binding_eq(a, b),
                        _ => false,
                    }
                })
            }
            Predicate::SameOp { .. } => {
                pm::MutuallyExclusiveTree::with_pairwise_mutex(constraints, |a, b| {
                    if !matches!(b.predicate(), Predicate::SameOp { .. }) {
                        return false;
                    }
                    // a and b are mutually exclusive if they share an argument
                    let a_args: BTreeSet<_> = a.required_bindings().iter().copied().collect();
                    let b_args: BTreeSet<_> = b.required_bindings().iter().copied().collect();
                    assert_ne!(a_args, b_args);
                    !a_args.is_disjoint(&b_args)
                })
            }
            Predicate::DistinctQubits { .. } => {
                let constraints = constraints.into_iter().filter(|(c, _)| {
                    // We can only turn DistinctQubits constraints into mutex predicates
                    // if they act on the same variable
                    matches!(c.predicate(), Predicate::DistinctQubits { .. })
                        && fst_required_binding_eq(c, &first)
                });
                pm::MutuallyExclusiveTree::with_powerset(constraints.collect())
            }
        }
    }
}

impl pm::ConditionedPredicate<PatternOpPosition> for Predicate {
    fn conditioned(constraint: &Constraint, satisfied: &[&Constraint]) -> Option<Constraint> {
        if !matches!(constraint.predicate(), Predicate::DistinctQubits { .. }) {
            return Some(constraint.clone());
        }
        let first_key = constraint.required_bindings()[0];
        let mut keys: BTreeSet<_> = constraint.required_bindings()[1..]
            .iter()
            .copied()
            .collect();
        for s in satisfied
            .iter()
            .filter(|s| s.required_bindings()[0] == first_key)
        {
            for k in s.required_bindings()[1..].iter() {
                keys.remove(k);
            }
        }
        if keys.is_empty() {
            return None;
        }
        let mut args = vec![first_key];
        args.extend(keys);
        let n_qubits = args.len();
        Some(Constraint::try_new(Predicate::DistinctQubits { n_qubits }, args).unwrap())
    }
}

fn fst_required_binding_eq(a: &Constraint, b: &Constraint) -> bool {
    a.required_bindings()[0] == b.required_bindings()[0]
}
