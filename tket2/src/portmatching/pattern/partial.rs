use std::collections::BTreeSet;

use itertools::Itertools;
use portmatching::{
    self as pm,
    pattern::{ClassRank, Satisfiable},
};

use crate::portmatching::{
    branch::find_shared_class,
    indexing::HugrPortID,
    pattern::{compute_class_rank, get_distinct_from_classes},
    Constraint, ConstraintClass, HugrVariableID, Predicate,
};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct PartialPattern {
    /// The set of constraints that must still be satisfied
    pattern_constraints: BTreeSet<Constraint>,
    /// All linear wires in the pattern
    all_linear_wires: BTreeSet<HugrPortID>,
    /// The subset of `all_linear_wires` that are known to be distinct
    known_distinct_wires: BTreeSet<HugrPortID>,
}

impl PartialPattern {
    pub fn new(
        pattern_constraints: BTreeSet<Constraint>,
        all_linear_wires: BTreeSet<HugrPortID>,
    ) -> Self {
        Self {
            pattern_constraints,
            known_distinct_wires: BTreeSet::new(),
            all_linear_wires,
        }
    }

    /// Whether all constraints of the pattern are satisfied
    fn is_satisifed(&self) -> bool {
        self.pattern_constraints.is_empty() && self.are_all_wires_known()
    }

    /// Whether all wires in the pattern are known to be pairwise distinct
    fn are_all_wires_known(&self) -> bool {
        self.all_wires()
            .all(|w| self.known_distinct_wires.contains(&w))
    }

    fn all_wires(&self) -> impl Iterator<Item = HugrPortID> + '_ {
        self.all_linear_wires
            .iter()
            .filter_map(|&v| HugrPortID::try_from(v).ok())
    }

    fn with_removed_constraint(
        &self,
        constraint: &portmatching::Constraint<HugrVariableID, Predicate>,
    ) -> Self {
        let mut ret = self.clone();
        let success = ret.pattern_constraints.remove(constraint);
        debug_assert!(success);
        ret
    }

    fn with_known_distinct_wire(&self, wire: HugrPortID) -> Self {
        let mut ret = self.clone();
        ret.known_distinct_wires.insert(wire);
        ret
    }

    fn to_satisfiable(&self) -> Satisfiable<Self> {
        self.clone().into_satisfiable()
    }

    fn into_satisfiable(self) -> Satisfiable<Self> {
        if self.is_satisifed() {
            Satisfiable::Tautology
        } else {
            Satisfiable::Yes(self)
        }
    }
}

impl pm::PartialPattern for PartialPattern {
    type Constraint = Constraint;
    type ConstraintClass = ConstraintClass;
    type Key = HugrVariableID;

    fn rank_classes(
        &self,
        known_bindings: &[Self::Key],
    ) -> impl Iterator<Item = (Self::ConstraintClass, ClassRank)> {
        // Class rank * number of new bindings required for constraint
        let pattern_classes = self.pattern_constraints.iter().flat_map(|c| {
            let reqs = c.required_bindings();
            let n_new_bindings = reqs.iter().filter(|k| !known_bindings.contains(k)).count() as i32;
            c.predicate()
                .get_classes(c.required_bindings())
                .into_iter()
                .map(move |cls| (cls, compute_class_rank(cls, n_new_bindings)))
        });
        let distinct_from_classes =
            get_distinct_from_classes(known_bindings, &self.known_distinct_wires, self.all_wires())
                .collect_vec();

        pattern_classes.chain(distinct_from_classes)
    }

    fn nominate(&self, cls: &Self::BranchClass) -> BTreeSet<Self::Constraint> {
        let mut constraints: BTreeSet<_> = self
            .pattern_constraints
            .iter()
            .filter(|c| {
                let classes = c.predicate().get_classes(c.required_bindings());
                classes.iter().any(|c_cls| c_cls == cls)
            })
            .cloned()
            .collect();

        // Nominate an IsDistinctFrom constraint if applicable
        if let &BranchClass::IsDistinctFromClass(w) = cls {
            if self.all_linear_wires.contains(&w) && !self.known_distinct_wires.contains(&w) {
                let pred = Predicate::new_is_distinct_from(self.known_distinct_wires.len());
                let all_wires = [w]
                    .into_iter()
                    .chain(self.known_distinct_wires.iter().copied());
                let args = all_wires.map(HugrVariableID::LinearWire).collect();
                let constraint = Constraint::try_new(pred, args).unwrap();
                constraints.insert(constraint);
            }
        }

        constraints
    }

    fn apply_transitions(&self, transitions: &[Self::Constraint]) -> Vec<Satisfiable<Self>> {
        let class = find_shared_class(transitions)
            .expect("transitions applied simultaneously must share a common class");

        use ConstraintClass::*;
        match class {
            IsOpEqualClass(_)
            | OccupyOutgoingPortClass(_, _)
            | OccupyIncomingPortClass(_, _)
            | IsWireSourceClass(_)
            | IsLinearWireSinkClass(_) => {
                let constraint_pos = transitions
                    .iter()
                    .position(|c| self.pattern_constraints.contains(c))
                    .expect("no nominated transition found");
                let constraint = &transitions[constraint_pos];
                let new_pattern = self.with_removed_constraint(constraint);
                one_hot_vec(
                    constraint_pos,
                    transitions.len(),
                    new_pattern.into_satisfiable(),
                )
            }
            IsDistinctFromClass(wire) => {
                assert!(
                    !self.known_distinct_wires.contains(&wire),
                    "no transition nominated for this branch class"
                );

                // Find the transitions to apply
                let pos_transitions = transitions.iter().enumerate().filter(|(_, c)| {
                    let other_wires = BTreeSet::<HugrPortID>::from_iter(
                        c.required_bindings()[1..]
                            .iter()
                            .map(|&w| w.try_into().unwrap()),
                    );
                    self.known_distinct_wires
                        .iter()
                        .all(|w| other_wires.contains(w))
                });

                // Sanity check: at least one transition will be applied
                let mut pos_transitions = pos_transitions.peekable();
                assert!(
                    pos_transitions.peek().is_some(),
                    "no nominated transition found"
                );

                // Build the vector of new patterns
                let mut ret = vec![Satisfiable::No; transitions.len()];
                let new_pattern = self.with_known_distinct_wire(wire);
                for (pos, _) in pos_transitions {
                    ret[pos] = new_pattern.to_satisfiable();
                }
                ret
            }
        }
    }

    fn is_satisfiable(&self) -> portmatching::pattern::Satisfiable {
        if self.is_satisifed() {
            Satisfiable::Tautology
        } else {
            Satisfiable::Yes(())
        }
    }
}

fn one_hot_vec<V: Clone>(
    constraint_pos: usize,
    len: usize,
    val: Satisfiable<V>,
) -> Vec<Satisfiable<V>> {
    let mut ret = vec![Satisfiable::No; len];
    ret[constraint_pos] = val;
    ret
}
