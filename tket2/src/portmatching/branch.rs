//! Constraint classes for [`Predicate`].
//!
//! Constraint classes are labels for constraints that are used to cluster
//! constraints into groups that are related to one another (in some way), and
//! can be evaluated together.
//!
//! This is used in pattern matching: all transitions outgoing from any
//! state of the pattern matcher must belong to one constraint class. It thus
//! makes sense to use constraint classes to e.g. group constraints that are
//! mutually exclusive.
//!
//! Concretely, this module provides two things:
//!  1. Defines [`ConstraintClass`] to cluster constraints made of
//!     [`Predicate`]s. This also specifies the assignment of constraints to
//!     classes, as well as the "expansion factor" of a list of constraints
//!     (a heuristic to determine how "valuable/expensive" constraints are --
//!     see the inline comments for more details).
//!  2. Provides a branch selector [`BranchSelector`] to evaluate lists of
//!     constraints in a class. This is used at pattern matching runtime to
//!     evaluate all constraints and choose which transitions to descend into.
//!
//! The former is necessary to construct a pattern matcher, while the latter is
//! used to evaluate predicates when traversing it.

use std::{collections::BTreeSet, fmt};

use hugr::Port;
use itertools::Itertools;
use portmatching::{self as pm, indexing::IndexKey};

use super::{
    indexing::{HugrNodeID, HugrPortID},
    HugrVariableID, Predicate,
};

/// The constraint classes that cluster hugr [`Predicate`]s into groups that can be
/// evaluated and constructed together.
///
/// These dictate the set of available transitions within a pattern matcher: all
/// outgoing transitions at any one state will share one constraint class.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum ConstraintClass {
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
    /// The second class for [`Predicate::IsWireSink`].
    ///
    /// This class only applies if the wire type is linear ([`HugrVariableID::LinearWire`]).
    /// The class is parametrised by the wire this predicate refers to. The sink
    /// of a linear wire is always unique. Hence any two constraints on the sink of
    /// the same wire must be for the same node and port or be mutually
    /// exclusive.
    IsLinearWireSinkClass(HugrPortID),
}

impl pm::ConstraintClass<super::Constraint> for ConstraintClass {
    fn get_classes(constraint: &super::Constraint) -> Vec<Self> {
        use HugrVariableID::*;
        use Predicate::*;

        let keys = constraint.required_bindings();

        match *constraint.predicate() {
            IsOpEqual(_) => {
                let Ok(node) = keys[0].try_into() else {
                    panic!("invalid key type");
                };
                vec![ConstraintClass::IsOpEqualClass(node)]
            }
            IsWireSource(out_port) => {
                let Ok(node) = keys[0].try_into() else {
                    panic!("invalid key type");
                };
                let Ok(wire) = keys[1].try_into() else {
                    panic!("invalid key type")
                };
                vec![
                    ConstraintClass::IsWireSourceClass(wire),
                    ConstraintClass::OccupyOutgoingPortClass(node, out_port),
                ]
            }
            IsWireSink(in_port) => {
                let Ok(node) = keys[0].try_into() else {
                    panic!("invalid key type");
                };
                match keys[1] {
                    Op(_) => panic!("invalid key type"),
                    CopyableWire(_) => {
                        vec![ConstraintClass::OccupyIncomingPortClass(node, in_port)]
                    }
                    LinearWire(wire) => vec![
                        ConstraintClass::IsLinearWireSinkClass(wire),
                        ConstraintClass::OccupyIncomingPortClass(node, in_port),
                    ],
                }
            }
            IsDistinctFrom { .. } => {
                let Ok(port) = keys[0].try_into() else {
                    panic!("invalid key type");
                };
                vec![ConstraintClass::IsDistinctFromClass(port)]
            }
        }
    }

    fn expansion_factor<'c>(
        &self,
        constraints: impl IntoIterator<Item = &'c super::Constraint>,
    ) -> portmatching::constraint_class::ExpansionFactor
    where
        super::Constraint: 'c,
    {
        // Expansion factor is an estimate for the number of constraints that
        // we expect to be satisfied in `constraints`. This is used when building
        // the pattern matcher automaton to prioritise sets of constraints that
        // are the most selective i.e. have the lowest expansion factor.
        //
        // These are quite conservative estimates and could be fine-tuned in
        // the future

        // A guess of the probability that two node/wire keys are bound to the
        // same node/wire (i.e. a non-injective binding map)
        let not_inj_probability = 0.3;

        use ConstraintClass::*;
        match self {
            // Constraints on the same node.
            //
            // Any two different op constraints on the same node are always
            // mutually exclusive. So (an upper bound for) the expansion factor
            // is 1.
            IsOpEqualClass(_) => 1,
            // Constraints on the same wire.
            //
            // If the constraints are on different port offsets, then they are
            // mutually exclusive. (you cannot have the same wire connect to
            // different port offset at source / or linear sink). If constraints
            // are on the same port offset but different node keys, then in the
            // worst case we cannot say anything: it is still to be proven that
            // the two node keys are actually bound to different nodes.
            IsWireSourceClass(_) | IsLinearWireSinkClass(_) => {
                let groups = constraints
                    .into_iter()
                    .into_group_map_by(|c| match *c.predicate() {
                        Predicate::IsWireSource(outgoing_port) => Port::from(outgoing_port),
                        Predicate::IsWireSink(incoming_port) => Port::from(incoming_port),
                        _ => panic!("unexpected predicate in constraint class"),
                    });
                let max_group_size = groups
                    .values()
                    .map(|g| g.len() as u64)
                    .max()
                    .expect("empty constraints iterator");
                1 + ((max_group_size as f64 * not_inj_probability).ceil() as u64)
            }
            // Constraints on the same (node, port) tuple.
            //
            // The wire at a given port is unique, but we must still prove that
            // the wire keys are bound to different wires.
            OccupyOutgoingPortClass(_, _) | OccupyIncomingPortClass(_, _) => {
                let n_constraints = constraints.into_iter().count() as f64;
                1 + ((n_constraints * not_inj_probability).ceil() as u64)
            }
            // `Predicate::IsDistinctFrom` constraints checking for distinctness
            // of the same node.
            //
            // We use the DominantDistinct evaluation strategy, so the number of
            // constraints satisfied is at most the size of the subset of
            // constraints such that no constraint implies another.
            IsDistinctFromClass(_) => {
                let all_constraints = constraints.into_iter().collect_vec();
                all_constraints
                    .iter()
                    .filter(|&&self_constraint| {
                        !all_constraints.iter().any(|&other_constraint| {
                            self_constraint != other_constraint && {
                                self_constraint.predicate().implies(
                                    other_constraint.predicate(),
                                    self_constraint.required_bindings(),
                                    other_constraint.required_bindings(),
                                )
                            }
                        })
                    })
                    .count() as u64
            }
        }
    }
}

impl pm::GetConstraintClass<HugrVariableID> for Predicate {
    type ConstraintClass = ConstraintClass;
}

/// A branch selector for Hugr [`Predicate`]s.
///
/// This is used to evaluate constraints "in batches", i.e. to evaluate
/// which constraints are satisfied from a list of constraints.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BranchSelector<K, P: pm::GetConstraintClass<K>> {
    /// The branch class all constraints belong to
    constraint_class: P::ConstraintClass,
    /// The indices into `all_required_bindings` for the keys of each
    /// constraint.
    ///
    /// The i-th constraint requires the bindings `all_required_bindings[j]`
    /// for j in binding_indices[i].
    binding_indices: Vec<Vec<usize>>,
    /// The predicates of each constraint
    predicates: Vec<P>,
    /// All bindings required for the evaluation of the constraints
    all_required_bindings: Vec<K>,
    /// The evaluation strategy for the constraints
    ///
    /// This determines which subset of satisfied constraints should be
    /// returned by the branch selector. For example, in the determinstic
    /// strategy, only the first constraint that is satisfied is returned.
    evaluation_strategy: EvaluationStrategy,
}

/// An evaluation strategy for [`BranchSelector`]
///
/// There are multiple possible evaluation strategies:
///  - in "deterministic" evaluation, constraints are evaluated in order up
///    until the first constraint that is satisfied. It is called deterministic
///    because the resulting automaton is deterministic: at any state, there is
///    at most one constriant that is satisfied, therefore at most one
///    transition can be taken.
///  - in "non-deterministic" evaluation, all constraints are evaluated
///    independently of each other.
///  - "dominant distinct" evaluation is a hybrid of the above two: given
///    implication relations between predicates (P1 => P2), it will find the
///    largest set of satisfied constraints such that no two constraints imply
///    each other. It is made specifically for [`Predicate::IsDistinctFrom`]
///    predicates; the => implication relation is given by the subset relation
///    of `other_wire_sets`.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
enum EvaluationStrategy {
    /// Deterministic evaluation of constraints
    Det,
    /// Non-deterministic evaluation of constraints
    NonDet,
    /// Evaluation of constraints that do not imply each other
    DominantDistinct,
}

/// A trait to capture the implication relation between two constraints.
///
/// Required for most [`BranchSelector`] methods, as it is used in the
/// `EvaluationStrategy::DominantDistinct` evaluation strategy.
///
/// We provide an implementation for [`Predicate`], which should cover most
/// users needs.
pub trait PredicateImplication<K>: Clone + pm::ArityPredicate + pm::GetConstraintClass<K> {
    /// Check if `self` implies `other` given the keys of the constraints.
    ///
    /// It is guaranteed that for `self` and `other`, `supports_implication()`
    /// is true.
    fn implies(&self, other: &Self, self_keys: &[K], other_keys: &[K]) -> bool;

    /// Check if the predicate supports implication.
    fn supports_implication(&self) -> bool;
}

impl PredicateImplication<HugrVariableID> for Predicate {
    fn implies(
        &self,
        other: &Self,
        self_keys: &[HugrVariableID],
        other_keys: &[HugrVariableID],
    ) -> bool {
        debug_assert!(matches!(self, Predicate::IsDistinctFrom { .. }));
        debug_assert!(matches!(other, Predicate::IsDistinctFrom { .. }));
        // TODO: this is inefficient. Ideally, this would only be evaluated
        // once when a BranchSelector is constructed, and then cached.

        let self_target = self_keys[0];
        let other_target = other_keys[0];
        if self_target != other_target {
            // Both constraints must target the same key for an implication to exist
            return false;
        }
        // self => other if other_keys[1..] is a subset of self_keys[1..]
        other_keys[1..].iter().all(|&k| self_keys[1..].contains(&k))
    }

    fn supports_implication(&self) -> bool {
        matches!(self, Predicate::IsDistinctFrom { .. })
    }
}

impl<K, P> BranchSelector<K, P>
where
    K: IndexKey,
    P: PredicateImplication<K>,
    P::ConstraintClass: Copy,
{
    /// Construct a deterministic branch selector from constraints.
    pub fn new_det(constraints: &[pm::Constraint<K, P>]) -> Self {
        Self::new(constraints, EvaluationStrategy::Det)
    }

    /// Construct a non-deterministic branch selector from constraints.
    pub fn new_non_det(constraints: &[pm::Constraint<K, P>]) -> Self {
        Self::new(constraints, EvaluationStrategy::NonDet)
    }

    /// Construct a dominant distinct branch selector from constraints.
    pub fn new_dominant(constraints: &[pm::Constraint<K, P>]) -> Self {
        assert!(!constraints.is_empty());
        assert!(constraints
            .iter()
            .all(|c| c.predicate().supports_implication()));

        Self::new(constraints, EvaluationStrategy::DominantDistinct)
    }

    /// Construct a det or non-det branch selector from an iterator of constraints.
    fn new(transitions: &[pm::Constraint<K, P>], evaluation_strategy: EvaluationStrategy) -> Self {
        let predicates = transitions.iter().map(|c| c.predicate().clone()).collect();
        let (all_required_bindings, binding_indices) = get_required_bindings(transitions);
        let constraint_class =
            find_shared_class(transitions).expect("no shared branch class in selector");

        Self {
            predicates,
            all_required_bindings,
            binding_indices,
            constraint_class,
            evaluation_strategy,
        }
    }

    /// Indices of all constraints that are satisfied by the given bindings.
    pub(crate) fn all_satisfied_positions<'a, D: 'a, V: 'a>(
        &'a self,
        bindings: &'a [Option<V>],
        data: &'a D,
    ) -> impl Iterator<Item = usize> + 'a
    where
        P: pm::EvaluatePredicate<D, V>,
    {
        // TODO: this evaluation could be tailored to the specific predicates,
        // to evaluate all predicates in one go, rather than one by one.
        self.predicates_indices_iter()
            .positions(|(predicate, indices)| {
                let Ok(bindings) = indices
                    .iter()
                    .map(|&i| bindings[i].as_ref().ok_or(()))
                    .collect::<Result<Vec<_>, _>>()
                else {
                    return false;
                };
                predicate.check(&bindings, data)
            })
    }

    fn implies(&self, i: usize, j: usize) -> bool {
        let bindings_i = self.keys(i).collect_vec();
        let bindings_j = self.keys(j).collect_vec();
        self.predicates[i].implies(&self.predicates[j], &bindings_i, &bindings_j)
    }

    fn predicates_indices_iter(&self) -> impl Iterator<Item = (&P, &Vec<usize>)> + '_ {
        self.predicates.iter().zip(self.binding_indices.iter())
    }

    fn constraint_class(&self) -> P::ConstraintClass {
        self.constraint_class
    }

    fn predicate(&self, i: usize) -> &P {
        &self.predicates[i]
    }

    fn keys(&self, i: usize) -> impl Iterator<Item = K> + '_ {
        self.binding_indices[i]
            .iter()
            .map(|&i| self.all_required_bindings[i])
    }
}

impl<K: IndexKey, P: pm::GetConstraintClass<K>> pm::BranchSelector for BranchSelector<K, P> {
    type Key = K;

    fn required_bindings(&self) -> &[Self::Key] {
        &self.all_required_bindings
    }
}

impl<K: IndexKey, P: PredicateImplication<K> + fmt::Debug>
    pm::branch_selector::DisplayBranchSelector for BranchSelector<K, P>
where
    P::ConstraintClass: Copy + fmt::Debug,
{
    fn fmt_class(&self) -> String {
        format!("{:?}", self.constraint_class())
    }

    fn fmt_nth_constraint(&self, n: usize) -> String {
        format!("{:?}", self.predicate(n))
    }
}

impl<D, V, P, K> pm::EvaluateBranchSelector<D, V> for BranchSelector<K, P>
where
    P: pm::EvaluatePredicate<D, V> + PredicateImplication<K>,
    P::ConstraintClass: Copy,
    K: IndexKey,
{
    fn eval(&self, bindings: &[Option<V>], data: &D) -> Vec<usize> {
        let mut satisfied_positions = self.all_satisfied_positions(bindings, data);
        match self.evaluation_strategy {
            EvaluationStrategy::Det => {
                let first_satisifed = satisfied_positions.next();
                first_satisifed.into_iter().collect()
            }
            EvaluationStrategy::NonDet => satisfied_positions.collect(),
            EvaluationStrategy::DominantDistinct => {
                // TODO: this is inefficient, could be significantly
                // sped up using some dag storing the subset relations
                let all_satisfied = self.all_satisfied_positions(bindings, data).collect_vec();
                let mut retained = all_satisfied.clone();
                retained.retain(|&i| !all_satisfied.iter().any(|&j| i != j && self.implies(j, i)));
                retained
            }
        }
    }
}

pub(super) fn find_shared_class<K: IndexKey, P: PredicateImplication<K>>(
    transitions: &[pm::Constraint<K, P>],
) -> Option<P::ConstraintClass>
where
    P::ConstraintClass: Copy,
{
    let all_classes = transitions.iter().map(|c| {
        c.predicate()
            .try_get_classes(c.required_bindings())
            .expect("invalid constraints")
            .into_iter()
            .collect::<BTreeSet<_>>()
    });
    all_classes
        .reduce(|acc, classes| acc.intersection(&classes).cloned().collect())
        .and_then(|common_classes| common_classes.into_iter().next())
}

fn get_required_bindings<K: Eq + Copy, P>(
    transitions: &[pm::Constraint<K, P>],
) -> (Vec<K>, Vec<Vec<usize>>) {
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

#[cfg(test)]
mod test {
    use hugr::{
        ops::{Module, OpType},
        Direction, Port,
    };
    use portmatching as pm;

    use rstest::rstest;

    use crate::portmatching::{
        indexing::{HugrNodeID, HugrPortID},
        Constraint, HugrVariableID, Predicate,
    };

    use super::{BranchSelector, ConstraintClass, EvaluationStrategy, PredicateImplication};

    #[derive(Debug, Clone, Copy, Eq, PartialEq, PartialOrd, Ord)]
    enum TestPredicate {
        TrueIfDataIsZero,
        TrueIfDataIsOne,
        AlwaysTrue,
        AlwaysFalse,
    }
    #[derive(Debug, Clone, Copy, Eq, PartialEq, PartialOrd, Ord)]
    struct TestConstraintClass;
    impl pm::ConstraintClass<pm::Constraint<(), TestPredicate>> for TestConstraintClass {
        fn get_classes(_: &pm::Constraint<(), TestPredicate>) -> Vec<Self> {
            vec![TestConstraintClass]
        }

        fn expansion_factor<'c>(
            &self,
            _: impl IntoIterator<Item = &'c pm::Constraint<(), TestPredicate>>,
        ) -> portmatching::constraint_class::ExpansionFactor
        where
            pm::Constraint<(), TestPredicate>: 'c,
        {
            1
        }
    }
    impl pm::ArityPredicate for TestPredicate {
        fn arity(&self) -> usize {
            0
        }
    }
    impl pm::GetConstraintClass<()> for TestPredicate {
        type ConstraintClass = TestConstraintClass;
    }
    impl PredicateImplication<()> for TestPredicate {
        fn implies(&self, other: &Self, _: &[()], _: &[()]) -> bool {
            match (self, other) {
                _ if self == other => true,
                (TestPredicate::TrueIfDataIsZero, TestPredicate::AlwaysTrue) => true,
                (TestPredicate::TrueIfDataIsOne, TestPredicate::AlwaysTrue) => true,
                (TestPredicate::AlwaysFalse, TestPredicate::AlwaysTrue) => true,
                (TestPredicate::AlwaysFalse, TestPredicate::TrueIfDataIsOne) => true,
                (TestPredicate::AlwaysFalse, TestPredicate::TrueIfDataIsZero) => true,
                _ => false,
            }
        }

        fn supports_implication(&self) -> bool {
            true
        }
    }
    impl pm::EvaluatePredicate<usize, ()> for TestPredicate {
        fn check(&self, _: &[impl std::borrow::Borrow<()>], &data: &usize) -> bool {
            match self {
                TestPredicate::TrueIfDataIsZero if data == 0 => true,
                TestPredicate::TrueIfDataIsOne if data == 1 => true,
                TestPredicate::AlwaysTrue => true,
                _ => false,
            }
        }
    }

    #[rstest]
    #[case(
        EvaluationStrategy::Det,
        vec![0],
        vec![1]
    )]
    #[case(
        EvaluationStrategy::NonDet,
        vec![0, 2],
        vec![1, 2]
    )]
    #[case(
        EvaluationStrategy::DominantDistinct,
        vec![0],
        vec![1]
    )]
    fn test_branch_selector(
        #[case] eval_strat: EvaluationStrategy,
        #[case] expected_0: Vec<usize>,
        #[case] expected_1: Vec<usize>,
    ) {
        use portmatching::EvaluateBranchSelector;

        let constraints = vec![
            pm::Constraint::try_new(TestPredicate::TrueIfDataIsZero, vec![]).unwrap(),
            pm::Constraint::try_new(TestPredicate::TrueIfDataIsOne, vec![]).unwrap(),
            pm::Constraint::try_new(TestPredicate::AlwaysTrue, vec![]).unwrap(),
            pm::Constraint::try_new(TestPredicate::AlwaysFalse, vec![]).unwrap(),
        ];
        let selector = BranchSelector::new(&constraints, eval_strat);

        let result = selector.eval(&[], &0);
        assert_eq!(result, expected_0);

        let result = selector.eval(&[], &1);
        assert_eq!(result, expected_1);
    }

    #[rstest]
    #[case(
        Constraint::try_new(
            Predicate::IsOpEqual(OpType::Module(Module {}).into()),
            vec![HugrVariableID::Op(HugrNodeID::root())],
        )
        .unwrap(),
        vec![ConstraintClass::IsOpEqualClass(HugrNodeID::root())]
    )]
    #[case(
        Constraint::try_new(
            Predicate::IsWireSource(0.into()),
            vec![
                HugrVariableID::Op(HugrNodeID::root()),
                HugrVariableID::CopyableWire(HugrPortID::new(
                    HugrNodeID::root(),
                    Port::new(Direction::Outgoing, 0),
                )),
            ],
        )
        .unwrap(),
        vec![
            ConstraintClass::IsWireSourceClass(HugrPortID::new(
                HugrNodeID::root(),
                Port::new(Direction::Outgoing, 0),
            )),
            ConstraintClass::OccupyOutgoingPortClass(
                HugrNodeID::root(),
                0.into(),
            ),
        ]
    )]
    #[case(
        Constraint::try_new(
            Predicate::IsWireSink(0.into()),
            vec![
                HugrVariableID::Op(HugrNodeID::root()),
                HugrVariableID::CopyableWire(HugrPortID::new(
                    HugrNodeID::root(),
                    Port::new(Direction::Outgoing, 0),
                )),
            ],
        )
        .unwrap(),
        vec![
            ConstraintClass::OccupyIncomingPortClass(
                HugrNodeID::root(),
                0.into(),
            ),
        ]
    )]
    #[case(
        Constraint::try_new(
            Predicate::IsWireSink(0.into()),
            vec![
                HugrVariableID::Op(HugrNodeID::root()),
                HugrVariableID::LinearWire(HugrPortID::new(
                    HugrNodeID::root(),
                    Port::new(Direction::Outgoing, 0),
                )),
            ],
        )
        .unwrap(),
        vec![
            ConstraintClass::IsLinearWireSinkClass(HugrPortID::new(
                HugrNodeID::root(),
                Port::new(Direction::Outgoing, 0),
            )),
            ConstraintClass::OccupyIncomingPortClass(
                HugrNodeID::root(),
                0.into(),
            ),
        ]
    )]
    #[case(
        Constraint::try_new(
            Predicate::new_is_distinct_from(0),
            vec![HugrVariableID::LinearWire(HugrPortID::new(
                HugrNodeID::root(),
                Port::new(Direction::Outgoing, 0),
            ))]
        )
        .unwrap(),
        vec![ConstraintClass::IsDistinctFromClass(HugrPortID::new(
            HugrNodeID::root(),
            Port::new(Direction::Outgoing, 0),
        ))]
    )]
    fn test_get_classes(#[case] constraint: Constraint, #[case] expected: Vec<ConstraintClass>) {
        let result = constraint.get_classes();
        assert_eq!(result, expected);
    }
}
