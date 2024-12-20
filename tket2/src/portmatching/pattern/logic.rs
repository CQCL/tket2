use std::collections::{BTreeMap, BTreeSet};

use hugr::Port;
use portmatching::{self as pm, pattern::Satisfiable};

use crate::portmatching::{
    branch::BranchClass,
    indexing::{HugrNodeID, HugrPortID},
    HugrVariableID, Predicate,
};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct PatternLogic {}

impl pm::PatternLogic for PatternLogic {
    type Constraint = pm::Constraint<HugrVariableID, Predicate>;

    type BranchClass = BranchClass;

    fn rank_classes(
        &self,
    ) -> impl Iterator<Item = (Self::BranchClass, portmatching::pattern::ClassRank)> {
        None.into_iter()
    }

    fn nominate(&self, cls: &Self::BranchClass) -> std::collections::BTreeSet<Self::Constraint> {
        todo!()
    }

    fn condition_on(
        &self,
        transitions: &[Self::Constraint],
        known_constraints: &std::collections::BTreeSet<Self::Constraint>,
    ) -> Vec<portmatching::pattern::Satisfiable<Self>> {
        todo!()
    }

    fn is_satisfiable(&self) -> portmatching::pattern::Satisfiable {
        todo!()
    }
}
