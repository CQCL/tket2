use std::collections::BTreeSet;

use itertools::Itertools;
use portmatching::{
    self as pm,
    pattern::{ClassRank, Satisfiable},
};

use crate::portmatching::{
    indexing::{HugrNodeID, HugrPortID},
    pattern::{compute_class_rank, get_distinct_from_classes},
    BranchClass, Constraint, HugrVariableID, MatchOp, Predicate,
};

use super::Uf;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct PatternLogic {
    known_constraints: Uf<HugrNodeID, HugrPortID, MatchOp>,
    pattern_constraints: BTreeSet<Constraint>,

    /// All linear wires in the pattern
    all_linear_wires: BTreeSet<HugrPortID>,
    /// The subset of `all_linear_wires` that are known to be distinct
    known_distinct_wires: BTreeSet<HugrPortID>,
}

impl PatternLogic {
    pub fn new(
        pattern_constraints: BTreeSet<Constraint>,
        all_linear_wires: BTreeSet<HugrPortID>,
    ) -> Self {
        Self {
            known_constraints: Uf::new(),
            pattern_constraints,
            known_distinct_wires: BTreeSet::new(),
            all_linear_wires,
        }
    }

    /// Add constraint to the pattern's knowledge
    fn add_known_constraint(&mut self, constraint: &Constraint) -> Satisfiable {
        match constraint.predicate() {
            Predicate::IsOpEqual(op) => {
                let HugrVariableID::Op(node) = constraint.required_bindings()[0] else {
                    panic!("invalid key type");
                };
                self.known_constraints.set_op(node, op.clone())
            }
            &Predicate::IsWireSource(out_port) => {
                let HugrVariableID::Op(node) = constraint.required_bindings()[0] else {
                    panic!("invalid key type");
                };
                let (wire, is_linear) = match constraint.required_bindings()[1] {
                    HugrVariableID::CopyableWire(w) => (w, false),
                    HugrVariableID::LinearWire(w) => (w, true),
                    _ => panic!("invalid key type"),
                };
                self.known_constraints
                    .set_link(node, wire, out_port, is_linear)
            }
            &Predicate::IsWireSink(in_port) => {
                let HugrVariableID::Op(node) = constraint.required_bindings()[0] else {
                    panic!("invalid key type");
                };
                let (wire, is_linear) = match constraint.required_bindings()[1] {
                    HugrVariableID::CopyableWire(w) => (w, false),
                    HugrVariableID::LinearWire(w) => (w, true),
                    _ => panic!("invalid key type"),
                };
                self.known_constraints
                    .set_link(node, wire, in_port, is_linear)
            }
            Predicate::IsDistinctFrom { .. } => {
                let is_linear = matches!(
                    constraint.required_bindings()[0],
                    HugrVariableID::LinearWire(_)
                );
                let ports = constraint
                    .required_bindings()
                    .iter()
                    .map(|&key| {
                        let Ok(port) = key.try_into() else {
                            panic!("invalid key type");
                        };
                        port
                    })
                    .collect_vec();
                let fst_port = ports[0];

                let mut all_tautology = true;
                for &port in &ports[1..] {
                    match self
                        .known_constraints
                        .set_wires_not_equal(fst_port, port, is_linear)
                    {
                        Satisfiable::Yes(_) => all_tautology = false,
                        Satisfiable::No => return Satisfiable::No,
                        Satisfiable::Tautology => (),
                    }
                }

                // Update known distinct wires
                if !self.known_distinct_wires.contains(&fst_port) {
                    let all_not_equal = self
                        .known_distinct_wires
                        .iter()
                        .all(|&w| self.known_constraints.check_wires_not_equal(fst_port, w));
                    if all_not_equal {
                        self.known_distinct_wires.insert(fst_port);
                    }
                }

                if all_tautology {
                    return Satisfiable::Tautology;
                } else {
                    Satisfiable::Yes(())
                }
            }
        }
    }

    /// Simplify pattern constraints based on known constraints.
    ///
    /// Call this after modifying known constraints.
    ///
    fn update_pattern_constraints(&mut self) -> Satisfiable {
        let mut clone = self.clone();
        let mut satisfiable = true;
        self.pattern_constraints.retain(|constraint| {
            match clone.add_known_constraint(constraint) {
                Satisfiable::Yes(()) => true,
                Satisfiable::No => {
                    satisfiable = false;
                    false
                }
                Satisfiable::Tautology => false,
            }
        });

        if !satisfiable {
            return Satisfiable::No;
        }
        if self.is_satisifed() {
            return Satisfiable::Tautology;
        }

        Satisfiable::Yes(())
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

    /// A "partial distinct from" constraint is a constraint with [Predicate::IsDistinctFrom]
    /// that does not contain enough information for us to conclude that all
    /// wires we are interested in are distinct.
    ///
    /// For example, consider the following constraint:
    /// ```text
    /// IsDistinctFrom(w1, w2, w3)
    /// ```
    /// asserting that w1 != w2 and w1 != w3. If `self.known_distinct_wires`
    /// is [w2, w3, w4], then we cannot conclude that w1 is distinct from all
    /// other wires as w1 ?= w4. In this case we ignore the constraint (and
    /// count on the fact that a stricter constraint is being added, too).
    ///
    /// We ignore such constraints to avoid blowing up the state space we keep
    /// track of.
    fn is_partial_distinct_from(&self, constraint: &Constraint) -> bool {
        if !matches!(constraint.predicate(), Predicate::IsDistinctFrom { .. }) {
            return false;
        }
        let target: HugrPortID = constraint.required_bindings()[0].try_into().unwrap();
        if self.known_distinct_wires.contains(&target) {
            return false;
        }
        if !self.all_linear_wires.contains(&target) {
            return false;
        }
        let other_wires: BTreeSet<HugrPortID> = constraint.required_bindings()[1..]
            .iter()
            .map(|&w| w.try_into().unwrap())
            .collect();
        self.known_distinct_wires
            .iter()
            .any(|&w| !other_wires.contains(&w))
    }
}

impl pm::PatternLogic for PatternLogic {
    type Constraint = Constraint;

    type BranchClass = BranchClass;

    type Key = HugrVariableID;

    fn rank_classes(
        &self,
        known_bindings: &[Self::Key],
    ) -> impl Iterator<Item = (Self::BranchClass, ClassRank)> {
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
        let ret = transitions
            .iter()
            .map(|constraint| {
                let mut clone = self.clone();
                if matches!(clone.add_known_constraint(constraint), Satisfiable::No) {
                    return Satisfiable::No;
                }
                // Do not propagate to DistinctFrom predicates that only match partially
                if self.is_partial_distinct_from(constraint) {
                    return Satisfiable::No;
                }
                match clone.update_pattern_constraints() {
                    Satisfiable::Yes(()) => Satisfiable::Yes(clone),
                    Satisfiable::No => Satisfiable::No,
                    Satisfiable::Tautology => Satisfiable::Tautology,
                }
            })
            .collect_vec();
        let cnt = ret
            .iter()
            .filter(|sat| matches!(sat, Satisfiable::Yes(_)))
            .count();
        if cnt > 1 {
            dbg!(&transitions);
            println!("increased to {cnt}");
        }

        ret
    }

    fn is_satisfiable(&self) -> portmatching::pattern::Satisfiable {
        if self.is_satisifed() {
            Satisfiable::Tautology
        } else {
            Satisfiable::Yes(())
        }
    }
}
