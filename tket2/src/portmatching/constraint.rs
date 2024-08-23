use std::collections::BTreeSet;

use super::{indexing::PatternOpLocation, predicate::Predicate};

use itertools::Itertools;
use portmatching as pm;

pub type Constraint = pm::Constraint<PatternOpLocation, Predicate>;

pub(super) fn constraint_key(c: &Constraint) -> (&PatternOpLocation, &Predicate) {
    let arg = match c.predicate() {
        Predicate::Link { .. } => c.required_bindings().iter().max().unwrap(),
        Predicate::IsOp { .. } => c.required_bindings().first().unwrap(),
        Predicate::NotEq { .. } => c.required_bindings().first().unwrap(),
    };
    (arg, c.predicate())
}

impl pm::ToConstraintsTree<PatternOpLocation> for Predicate {
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
            Predicate::NotEq { .. } => {
                let constraints = constraints.into_iter().filter(|(c, _)| {
                    // We can only turn IsNotEqual constraints into mutex predicates
                    // if they act on the same variable
                    matches!(c.predicate(), Predicate::NotEq { .. })
                        && fst_required_binding_eq(c, &first)
                });
                pm::MutuallyExclusiveTree::with_powerset(constraints.collect())
            }
        }
    }
}

impl pm::ConditionedPredicate<PatternOpLocation> for Predicate {
    fn conditioned(constraint: &Constraint, satisfied: &[&Constraint]) -> Option<Constraint> {
        if !matches!(constraint.predicate(), Predicate::NotEq { .. }) {
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
            for k in s.required_bindings()[1..].iter().copied() {
                keys.remove(&k);
            }
        }
        if keys.is_empty() {
            return None;
        }
        let mut args = vec![first_key];
        let n_other = keys.len();
        args.extend(keys);
        Some(Constraint::try_new(Predicate::NotEq { n_other }, args).unwrap())
    }
}

fn fst_required_binding_eq(a: &Constraint, b: &Constraint) -> bool {
    a.required_bindings()[0] == b.required_bindings()[0]
}
