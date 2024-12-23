//! Implements predicate logic necessary to construct correct pattern matchers
//! as well as to evaluate predicates efficiently.
//!
//! Concretely, this module provides three things:
//!  1. Specifies the branch classes that cluster predicates into groups that can
//!     be evaluated together (e.g. mutually exclusive predicates)
//!  2. Implements [`pm::predicate::ConstraintLogic`] to provide the logic to
//!     simplify predicates when conditioned on other predicates
//!  3. Provides a branch selector to evaluate predicates that are in the same
//!     class efficiently in batches.
//!
//! 1. and 2. are necessary to construct a pattern matcher, while 3. is necessary
//! to evaluate predicates efficiently when traversing it.

use hugr::HugrView;
use portmatching as pm;

use crate::Circuit;

use super::{
    indexing::{HugrNodeID, HugrPortID},
    Constraint, HugrVariableID, HugrVariableValue, Predicate,
};

/// The branch classes that cluster hugr [`Predicate`]s into groups that can be
/// evaluated and constructed together.
///
/// These dictate the set of available transitions within a pattern matcher: all
/// outgoing transitions at any one state will share one branch class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BranchClass {
    /// The class of all [`Predicate::IsOpEqual`] predicates.
    ///
    /// The class is parametrised on the node the predicate applies to. Any two
    /// constraints in the same class must have the same MatchOp or are mutually
    /// exclusive.
    IsOpEqualClass(HugrNodeID),
    /// The class of all [`Predicate::IsDistinctFrom`] predicates.
    ///
    /// The class is parametrised by the first wire of the predicate -- the one
    /// that is distinct from all others. The logical relationship between two
    /// constraints of the same class is given by the relationship between the
    /// sets of all but the first predicate argument: e.g. if one's set is a
    /// subset of another, then there is a logical implication from the second
    /// to the first.
    IsDistinctFromClass(HugrPortID),
    /// One of two classes for [`Predicate::IsWireSource`].
    ///
    /// The class is parametrised by the node and port the wire is connected to.
    /// There can only be one wire at any one port. Hence any two constraints
    /// in the same class must be connected to the same wire or be mutually
    /// exclusive.
    OccupyOutgoingPortClass(HugrNodeID, hugr::OutgoingPort),
    /// One of up to two classes for [`Predicate::IsWireSink`].
    ///
    /// The class is parametrised by the node and port the wire is connected to.
    /// There can only be one wire at any one port. Hence any two constraints
    /// in the same class must be connected to the same wire or be mutually
    /// exclusive.
    OccupyIncomingPortClass(HugrNodeID, hugr::IncomingPort),
    /// The second class for [`Predicate::IsWireSource`].
    ///
    /// The class is parametrised by the wire this predicate refers to. The source
    /// of a wire is always unique. Hence any two constraints on the source of
    /// the same wire must be for the same node and port or be mutually
    /// exclusive.
    IsWireSourceClass(HugrPortID),
    /// The second class for [`Predicate::IsWireSource`].
    ///
    /// This class only applies if the wire type is linear ([`HugrVariableID::LinearWire`]).
    /// The class is parametrised by the wire this predicate refers to. The sink
    /// of a linear wire is always unique. Hence any two constraints on the sink of
    /// the same wire must be for the same node and port or be mutually
    /// exclusive.
    IsLinearWireSinkClass(HugrPortID),
}

impl BranchClass {
    pub(super) fn get_rank(&self) -> pm::pattern::ClassRank {
        use BranchClass::*;
        match self {
            IsOpEqualClass(_) => 0.3,
            IsDistinctFromClass(_) => 0.7,
            OccupyOutgoingPortClass(_, _) => 0.1,
            OccupyIncomingPortClass(_, _) => 0.1,
            IsWireSourceClass(_) => 0.1,
            IsLinearWireSinkClass(_) => 0.1,
        }
    }
}

impl Predicate {
    fn to_constraint(&self, keys: Vec<HugrVariableID>) -> pm::Constraint<HugrVariableID, Self> {
        pm::Constraint::try_new(self.clone(), keys).unwrap()
    }
}

impl Predicate {
    pub(super) fn get_classes(&self, keys: &[HugrVariableID]) -> Vec<BranchClass> {
        use HugrVariableID::*;
        use Predicate::*;

        match *self {
            IsOpEqual(_) => {
                let Ok(node) = keys[0].try_into() else {
                    panic!("invalid key type");
                };
                vec![BranchClass::IsOpEqualClass(node)]
            }
            IsWireSource(out_port) => {
                let Ok(node) = keys[0].try_into() else {
                    panic!("invalid key type");
                };
                let Ok(wire) = keys[1].try_into() else {
                    panic!("invalid key type")
                };
                vec![
                    BranchClass::IsWireSourceClass(wire),
                    BranchClass::OccupyOutgoingPortClass(node, out_port),
                ]
            }
            IsWireSink(in_port) => {
                let Ok(node) = keys[0].try_into() else {
                    panic!("invalid key type");
                };
                match keys[1] {
                    Op(_) => panic!("invalid key type"),
                    CopyableWire(_) => vec![BranchClass::OccupyIncomingPortClass(node, in_port)],
                    LinearWire(wire) => vec![
                        BranchClass::IsLinearWireSinkClass(wire),
                        BranchClass::OccupyIncomingPortClass(node, in_port),
                    ],
                }
            }
            IsDistinctFrom { .. } => {
                let Ok(port) = keys[0].try_into() else {
                    panic!("invalid key type");
                };
                vec![BranchClass::IsDistinctFromClass(port)]
            }
        }
    }
}

/// A branch selector for Hugr [`Constraint`]s.
pub struct BranchSelector {
    branch_class: BranchClass,
    binding_indices: Vec<Vec<usize>>,
    predicates: Vec<Predicate>,
    all_required_bindings: Vec<HugrVariableID>,
}

impl BranchSelector {
    /// Construct a branch selector from an iterator of constraints.
    pub fn from_constraints<'c>(constraints: impl IntoIterator<Item = &'c Constraint>) -> Self {
        let transitions = constraints.into_iter();
        let size_hint = transitions.size_hint().0;

        let mut predicates = Vec::with_capacity(size_hint);
        let mut all_required_bindings = Vec::new();
        let mut binding_indices = Vec::with_capacity(size_hint);
        let mut valid_branch_classes: Option<Vec<_>> = None;

        for constraint in transitions {
            let pred = constraint.predicate().clone();

            // Populate required indices
            let mut indices = Vec::new();
            let reqs = constraint.required_bindings();
            indices.reserve(reqs.len());
            for &req in reqs {
                let pos = all_required_bindings.iter().position(|&k| k == req);
                if let Some(pos) = pos {
                    indices.push(pos);
                } else {
                    all_required_bindings.push(req);
                    indices.push(all_required_bindings.len() - 1);
                }
            }

            // Filter valid branch classes
            let c_classes = constraint
                .predicate()
                .get_classes(&constraint.required_bindings());
            if let Some(valid_branch_classes) = valid_branch_classes.as_mut() {
                valid_branch_classes.retain(|c| c_classes.contains(c));
            } else {
                valid_branch_classes = Some(
                    constraint
                        .predicate()
                        .get_classes(&constraint.required_bindings()),
                );
            }

            binding_indices.push(indices);
            predicates.push(pred);
        }

        let branch_class = valid_branch_classes
            .expect("empty branch selector")
            .pop()
            .expect("no shared branch class in selector");

        Self {
            predicates,
            all_required_bindings,
            binding_indices,
            branch_class,
        }
    }
}

impl pm::BranchSelector for BranchSelector {
    type Key = HugrVariableID;

    fn required_bindings(&self) -> &[Self::Key] {
        &self.all_required_bindings
    }
}

impl<H: HugrView> pm::EvaluateBranchSelector<Circuit<H>, HugrVariableValue> for BranchSelector {
    fn eval(&self, bindings: &[Option<HugrVariableValue>], data: &Circuit<H>) -> Vec<usize> {
        let opt_ind =
            self.predicates
                .iter()
                .zip(&self.binding_indices)
                .position(|(predicate, indices)| {
                    let Ok(bindings) = indices
                        .iter()
                        .map(|&i| bindings[i].as_ref().ok_or(()))
                        .collect::<Result<Vec<_>, _>>()
                    else {
                        return false;
                    };
                    <Predicate as pm::Predicate<_, _>>::check(predicate, &bindings, data)
                });
        opt_ind.into_iter().collect()
    }
}

impl pm::CreateBranchSelector<Constraint> for BranchSelector {
    fn create_branch_selector(constraints: Vec<Constraint>) -> Self {
        Self::from_constraints(&constraints)
    }
}
