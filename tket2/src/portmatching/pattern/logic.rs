use std::collections::BTreeSet;

use itertools::Itertools;
use portmatching::{self as pm, pattern::Satisfiable};

use crate::portmatching::{
    indexing::{HugrNodeID, HugrPortID},
    BranchClass, Constraint, HugrVariableID, MatchOp, Predicate,
};

use super::Uf;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct PatternLogic {
    known_constraints: Uf<HugrNodeID, HugrPortID, MatchOp>,
    pattern_constraints: BTreeSet<Constraint>,
}

impl PatternLogic {
    pub fn from_constraints(pattern_constraints: BTreeSet<Constraint>) -> Self {
        Self {
            known_constraints: Uf::new(),
            pattern_constraints,
        }
    }
}

impl pm::PatternLogic for PatternLogic {
    type Constraint = Constraint;

    type BranchClass = BranchClass;

    fn rank_classes(
        &self,
    ) -> impl Iterator<Item = (Self::BranchClass, portmatching::pattern::ClassRank)> {
        let branches = self
            .pattern_constraints
            .iter()
            .flat_map(|c| c.predicate().get_classes(c.required_bindings()))
            .unique();
        branches.map(|cls| (cls, cls.get_rank()))
    }

    fn nominate(&self, cls: &Self::BranchClass) -> BTreeSet<Self::Constraint> {
        self.pattern_constraints
            .iter()
            .filter(|c| {
                let classes = c.predicate().get_classes(c.required_bindings());
                classes.iter().any(|c_cls| c_cls == cls)
            })
            .cloned()
            .collect()
    }

    fn apply_transitions(&self, transitions: &[Self::Constraint]) -> Vec<Satisfiable<Self>> {
        transitions
            .iter()
            .map(|constraint| {
                let mut clone = self.clone();
                let sat = match constraint.predicate() {
                    Predicate::IsOpEqual(op) => {
                        let HugrVariableID::Op(node) = constraint.required_bindings()[0] else {
                            panic!("invalid key type");
                        };
                        clone.known_constraints.set_op(node, op.clone())
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
                        clone
                            .known_constraints
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
                        clone
                            .known_constraints
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
                            match clone
                                .known_constraints
                                .set_wires_not_equal(fst_port, port, is_linear)
                            {
                                Satisfiable::Yes(_) => all_tautology = false,
                                Satisfiable::No => return Satisfiable::No,
                                Satisfiable::Tautology => (),
                            }
                        }
                        if all_tautology {
                            return Satisfiable::Tautology;
                        } else {
                            Satisfiable::Yes(())
                        }
                    }
                };
                match sat {
                    Satisfiable::Yes(()) => Satisfiable::Yes(clone),
                    Satisfiable::No => Satisfiable::No,
                    Satisfiable::Tautology => Satisfiable::Tautology,
                }
            })
            .collect()
    }

    fn is_satisfiable(&self) -> portmatching::pattern::Satisfiable {
        if self.pattern_constraints.is_empty() {
            Satisfiable::Tautology
        } else {
            Satisfiable::Yes(())
        }
    }
}
