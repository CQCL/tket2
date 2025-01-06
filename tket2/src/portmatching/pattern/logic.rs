use std::collections::{BTreeSet, VecDeque};

use itertools::Itertools;
use portmatching::{
    self as pm,
    pattern::{self, ClassRank, Satisfiable},
};

use crate::portmatching::{
    indexing::{HugrNodeID, HugrPortID},
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
        transitions
            .iter()
            .map(|constraint| {
                let mut clone = self.clone();
                if matches!(clone.add_known_constraint(constraint), Satisfiable::No) {
                    return Satisfiable::No;
                }
                match clone.update_pattern_constraints() {
                    Satisfiable::Yes(()) => Satisfiable::Yes(clone),
                    Satisfiable::No => Satisfiable::No,
                    Satisfiable::Tautology => Satisfiable::Tautology,
                }
            })
            .collect()
    }

    fn is_satisfiable(&self) -> portmatching::pattern::Satisfiable {
        if self.is_satisifed() {
            Satisfiable::Tautology
        } else {
            Satisfiable::Yes(())
        }
    }
}

fn compute_class_rank(cls: BranchClass, n_new_bindings: i32) -> f64 {
    cls.get_rank() * (2_f64.powi(n_new_bindings))
}

/// Find wires that we'd like to check are distinct from all known distinct
/// wires
fn get_distinct_from_classes<'a>(
    known_bindings: &'a [HugrVariableID],
    known_distinct_wires: &'a BTreeSet<HugrPortID>,
    all_wires: impl Iterator<Item = HugrPortID> + 'a,
) -> impl Iterator<Item = (BranchClass, ClassRank)> + 'a {
    // The ports already bound
    let known_ports: BTreeSet<_> = known_bindings
        .iter()
        .filter_map(|&k| HugrPortID::try_from(k).ok())
        .collect();

    all_wires
        .filter(|w| !known_distinct_wires.contains(w))
        .map(move |w| {
            let cls = BranchClass::IsDistinctFromClass(w);

            let args = known_distinct_wires.iter().chain([&w]);
            let n_new_bindings = args.filter(|w| !known_ports.contains(w)).count() as i32;

            let rank = compute_class_rank(cls, n_new_bindings);
            (cls, rank)
        })
}
