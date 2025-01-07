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

use std::collections::BTreeSet;

use hugr::HugrView;
use itertools::Itertools;
use portmatching::{self as pm};

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
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
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
            // Always evaluate first: any two different constraints in the same
            // class are always mutually exclusive
            IsOpEqualClass(_) => 0.1,
            // Evaluate second: if ports are different, then they are mutually
            // exclusive. If nodes are different, then must still prove that the
            // two nodes are actually different
            IsWireSourceClass(_) => 0.13,
            IsLinearWireSinkClass(_) => 0.13,
            // Evaluate third: if wires are different, then must still prove that
            // the two wires are actually different
            OccupyOutgoingPortClass(_, _) => 0.16,
            OccupyIncomingPortClass(_, _) => 0.16,
            // Evaluate last, but before considering other constraints on
            // unknown keys
            IsDistinctFromClass(_) => 0.19,
        }
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
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum BranchSelector {
    Det {
        branch_class: BranchClass,
        binding_indices: Vec<Vec<usize>>,
        predicates: Vec<Predicate>,
        all_required_bindings: Vec<HugrVariableID>,
    },
    NonDet {
        branch_class: BranchClass,
        binding_indices: Vec<Vec<usize>>,
        predicates: Vec<Predicate>,
        all_required_bindings: Vec<HugrVariableID>,
    },
    DominantDistinct {
        all_required_bindings: Vec<HugrVariableID>,
        target_ind: usize,
        other_wire_sets: Vec<BTreeSet<usize>>,
    },
}

impl BranchSelector {
    pub fn new_det(constraints: &[Constraint]) -> Self {
        Self::new_det_or_non_det(constraints, true)
    }

    pub fn new_non_det(constraints: &[Constraint]) -> Self {
        Self::new_det_or_non_det(constraints, false)
    }

    pub fn new_dominant(constraints: &[Constraint]) -> Self {
        assert!(!constraints.is_empty());
        assert!(constraints
            .iter()
            .all(|c| matches!(c.predicate(), Predicate::IsDistinctFrom { .. })));

        let (all_required_bindings, binding_indices) = get_required_bindings(constraints);
        let target_ind = binding_indices[0][0];
        assert!(binding_indices.iter().all(|i| i[0] == target_ind));

        let other_wire_sets = binding_indices
            .iter()
            .map(|bindings| BTreeSet::from_iter(bindings[1..].iter().copied()))
            .collect_vec();

        Self::DominantDistinct {
            all_required_bindings,
            target_ind,
            other_wire_sets,
        }
    }

    /// Construct a det or non-det branch selector from an iterator of constraints.
    pub fn new_det_or_non_det(transitions: &[Constraint], is_det: bool) -> Self {
        let predicates = transitions.iter().map(|c| c.predicate().clone()).collect();
        let (all_required_bindings, binding_indices) = get_required_bindings(transitions);
        let branch_class =
            find_shared_class(transitions).expect("no shared branch class in selector");

        match is_det {
            true => Self::Det {
                predicates,
                all_required_bindings,
                binding_indices,
                branch_class,
            },
            false => Self::NonDet {
                predicates,
                all_required_bindings,
                binding_indices,
                branch_class,
            },
        }
    }

    /// Indices of all constraints that are satisfied by the given bindings.
    ///
    /// Panics if called on a DominantDistinct branch selector
    pub(crate) fn all_satisfied_positions(
        &self,
        bindings: &[Option<HugrVariableValue>],
        data: &Circuit<impl HugrView>,
    ) -> Vec<usize> {
        match self {
            Self::Det { .. } | Self::NonDet { .. } => {
                let ret = self.zip_predicates().positions(|(predicate, indices)| {
                    let Ok(bindings) = indices
                        .iter()
                        .map(|&i| bindings[i].as_ref().ok_or(()))
                        .collect::<Result<Vec<_>, _>>()
                    else {
                        return false;
                    };
                    <Predicate as pm::Predicate<_, _>>::check(predicate, &bindings, data)
                });
                ret.collect()
            }
            Self::DominantDistinct {
                other_wire_sets,
                target_ind,
                ..
            } => {
                let ret = other_wire_sets.iter().positions(|other_wires| {
                    let indices = [target_ind].into_iter().chain(other_wires);
                    let Ok(bindings) = indices
                        .map(|&i| bindings[i].as_ref().ok_or(()))
                        .collect::<Result<Vec<_>, _>>()
                    else {
                        return false;
                    };
                    let predicate = Predicate::new_is_distinct_from(other_wires.len());
                    <Predicate as pm::Predicate<_, _>>::check(&predicate, &bindings, data)
                });
                ret.collect()
            }
        }
    }

    fn zip_predicates(&self) -> impl Iterator<Item = (&Predicate, &Vec<usize>)> {
        let (predicates, binding_indices) = match self {
            Self::Det {
                predicates,
                binding_indices,
                ..
            } => (predicates, binding_indices),
            Self::NonDet {
                predicates,
                binding_indices,
                ..
            } => (predicates, binding_indices),
            Self::DominantDistinct { .. } => panic!("DominantDistinct has no predicate list"),
        };
        predicates.iter().zip(binding_indices)
    }

    fn branch_class(&self) -> BranchClass {
        match self {
            Self::Det { branch_class, .. } => *branch_class,
            Self::NonDet { branch_class, .. } => *branch_class,
            Self::DominantDistinct {
                all_required_bindings,
                target_ind,
                ..
            } => {
                let wire = all_required_bindings[*target_ind];
                BranchClass::IsDistinctFromClass(wire.try_into().unwrap())
            }
        }
    }

    fn predicate(&self, i: usize) -> Predicate {
        match self {
            Self::Det { predicates, .. } => predicates[i].clone(),
            Self::NonDet { predicates, .. } => predicates[i].clone(),
            Self::DominantDistinct {
                other_wire_sets, ..
            } => {
                let n_other = other_wire_sets[i].len();
                Predicate::new_is_distinct_from(n_other)
            }
        }
    }
}

impl pm::BranchSelector for BranchSelector {
    type Key = HugrVariableID;

    fn required_bindings(&self) -> &[Self::Key] {
        match self {
            Self::Det {
                all_required_bindings,
                ..
            } => all_required_bindings,
            Self::NonDet {
                all_required_bindings,
                ..
            } => all_required_bindings,
            Self::DominantDistinct {
                all_required_bindings,
                ..
            } => all_required_bindings,
        }
    }
}

impl pm::branch_selector::DisplayBranchSelector for BranchSelector {
    fn fmt_class(&self) -> String {
        format!("{:?}", self.branch_class())
    }

    fn fmt_nth_constraint(&self, n: usize) -> String {
        format!("{:?}", self.predicate(n))
    }
}

impl<H: HugrView> pm::EvaluateBranchSelector<Circuit<H>, HugrVariableValue> for BranchSelector {
    fn eval(&self, bindings: &[Option<HugrVariableValue>], data: &Circuit<H>) -> Vec<usize> {
        match self {
            Self::Det { .. } => {
                let first_satisifed = self
                    .all_satisfied_positions(bindings, data)
                    .first()
                    .copied();
                first_satisifed.into_iter().collect()
            }
            Self::NonDet { .. } => self.all_satisfied_positions(bindings, data),
            Self::DominantDistinct {
                other_wire_sets, ..
            } => {
                let all_satisfied = self.all_satisfied_positions(bindings, data);
                let mut retained = all_satisfied.clone();
                // TODO: this is asymptotically quite inefficient, would be faster
                // using a graph data structure
                retained.retain(|&i| {
                    !all_satisfied
                        .iter()
                        .any(|&j| j != i && other_wire_sets[i].is_subset(&other_wire_sets[j]))
                });
                retained
            }
        }
    }
}

pub(super) fn find_shared_class(transitions: &[Constraint]) -> Option<BranchClass> {
    let all_classes = transitions.iter().map(|c| {
        c.predicate()
            .get_classes(c.required_bindings())
            .into_iter()
            .collect::<BTreeSet<_>>()
    });
    all_classes
        .reduce(|acc, classes| acc.intersection(&classes).cloned().collect())
        .and_then(|common_classes| common_classes.into_iter().next())
}

fn get_required_bindings(transitions: &[Constraint]) -> (Vec<HugrVariableID>, Vec<Vec<usize>>) {
    let mut all_required_bindings = Vec::new();
    let mut binding_indices = Vec::with_capacity(transitions.len());

    for constraint in transitions {
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

        binding_indices.push(indices);
    }

    (all_required_bindings, binding_indices)
}
