use portgraph::PortOffset;
use portmatching as pm;

use crate::static_circ::{MatchOp, OpLocation, StaticSizeCircuit};

use super::{indexing::PatternOpLocation, Constraint};

/// Predicate for matching `StaticSizeCircuit`s.
#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum Predicate {
    /// An edge from `out_port` to `in_port`.
    Link { out_port: usize, in_port: usize },
    /// An operation of type `op`.
    IsOp { op: MatchOp },
    /// Check that the locations map is injective on the set of locations.
    NotEq { n_other: usize },
}

impl pm::ArityPredicate for Predicate {
    fn arity(&self) -> usize {
        match self {
            Predicate::Link { .. } => 2,
            Predicate::IsOp { .. } => 1,
            Predicate::NotEq { n_other } => n_other + 1,
        }
    }
}

impl pm::Predicate<StaticSizeCircuit> for Predicate {
    type Value = OpLocation;

    fn check(
        &self,
        data: &StaticSizeCircuit,
        args: &[impl std::borrow::Borrow<Self::Value>],
    ) -> bool {
        match self {
            &Predicate::Link { out_port, in_port } => {
                let &out_loc = args[0].borrow();
                let &in_loc = args[1].borrow();
                data.linked_op(out_loc, PortOffset::Outgoing(out_port as u16).into())
                    == Some((PortOffset::Incoming(in_port as u16).into(), in_loc))
            }
            Predicate::IsOp { op } => {
                let &loc = args[0].borrow();
                data.get(loc) == Some(op)
            }
            &Predicate::NotEq { n_other } => {
                let op = data.get_ptr(*args[0].borrow()).unwrap();
                for i in 0..n_other {
                    let &loc = args[i + 1].borrow();
                    if data.get_ptr(loc) == Some(op) {
                        return false;
                    }
                }
                true
            }
        }
    }
}

impl pm::DetHeuristic<PatternOpLocation> for Predicate {
    fn make_det(_constraints: &[&Constraint]) -> bool {
        true
    }
}
