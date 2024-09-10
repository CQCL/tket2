//! Pattern matching predicates.


use itertools::Itertools;
use portgraph::PortOffset;
use portmatching as pm;

use crate::{
    static_circ::{OpPosition, StaticSizeCircuit},
    Tk2Op,
};

use super::{indexing::PatternOpPosition, Constraint};

/// Predicate for matching `StaticSizeCircuit`s.
#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum Predicate {
    /// An edge from `out_port` to `in_port`.
    Link {
        /// The outgoing source port of the edge.
        out_port: usize,
        /// The incoming target port of the edge.
        in_port: usize,
    },
    /// An operation of type `op`.
    IsOp {
        /// The operation type to match.
        op: Tk2Op,
    },
    /// All locations are the same operation.
    SameOp {
        /// The arity of the predicate, i.e. the number of positions
        /// that share the same op.
        arity: usize,
    },
    /// Check that the first qubit is distinct from all others.
    DistinctQubits {
        /// The number of positions to check.
        n_qubits: usize,
    },
}

impl pm::ArityPredicate for Predicate {
    fn arity(&self) -> usize {
        match self {
            Predicate::Link { .. } => 2,
            Predicate::IsOp { .. } => 1,
            Predicate::SameOp { arity } => *arity,
            Predicate::DistinctQubits { n_qubits } => *n_qubits,
        }
    }
}

impl pm::Predicate<StaticSizeCircuit> for Predicate {
    type Value = OpPosition;

    fn check(
        &self,
        data: &StaticSizeCircuit,
        args: &[impl std::borrow::Borrow<Self::Value>],
    ) -> bool {
        let to_op_id = |pos| data.at_position(pos).unwrap();
        match *self {
            Predicate::Link { out_port, in_port } => {
                let &out_pos = args[0].borrow();
                let &in_pos = args[1].borrow();
                let out_op = to_op_id(out_pos);
                let in_op = to_op_id(in_pos);
                data.linked_op(out_op, PortOffset::Outgoing(out_port as u16).into())
                    == Some((PortOffset::Incoming(in_port as u16).into(), in_op))
            }
            Predicate::IsOp { ref op } => {
                let &pos = args[0].borrow();
                let id = to_op_id(pos);
                data.get(id).map(|op| op.op) == Some(*op)
            }
            Predicate::SameOp { .. } => args.iter().tuple_windows().all(|(a, b)| {
                let &loc_a = a.borrow();
                let &loc_b = b.borrow();
                let op_a = to_op_id(loc_a);
                let op_b = to_op_id(loc_b);
                op_a == op_b
            }),
            Predicate::DistinctQubits { n_qubits } => {
                let qubits = args.iter().map(|loc| loc.borrow().qubit);
                qubits.unique().count() == n_qubits
            }
        }
    }
}

impl pm::DetHeuristic<PatternOpPosition> for Predicate {
    fn make_det(constraints: &[&Constraint]) -> bool {
        constraints.len() > 2
    }
}
