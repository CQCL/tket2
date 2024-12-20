//! Predicates for pattern matching HUGRs.

use std::fmt::Debug;
use std::{borrow::Borrow, cmp};

use derive_more::{Display, Error};
use hugr::HugrView;
use itertools::Itertools;
use portmatching as pm;

use crate::Circuit;

use super::{
    indexing::find_source, to_hugr_values_tuple, to_hugr_values_vec, HugrVariableID,
    HugrVariableValue, MatchOp,
};

/// Predicate for pattern matching on flat hugrs.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
pub enum Predicate {
    /// Unary predicate checking the node OpType
    ///
    /// Constraint variable must be of type [`HugrVariableID::Op`].
    IsOpEqual(MatchOp),

    /// Arity-2 Predicate checking the (unique) source of a wire
    ///
    /// Given a wire and a node, check that the wire is connected to the node
    /// at the specified outgoing port.
    ///
    /// The first constraint variable must be of type [`HugrVariableID::Op`].
    /// The second constraint variable must be of type
    /// [`HugrVariableID::CopyableWire`] or [`HugrVariableID::LinearWire`].
    IsWireSource(hugr::OutgoingPort),

    /// Arity-2 Predicate checking the sink of a wire
    ///
    /// Given a wire and a node, check that the wire is connected to the node
    /// at the specified incoming port.
    ///
    /// If the wire is a linear type, then wire sinks must be unique.
    ///
    /// The first constraint variable must be of type [`HugrVariableID::Op`].
    /// The second constraint variable must be of type
    /// [`HugrVariableID::CopyableWire`] or [`HugrVariableID::LinearWire`].
    IsWireSink(hugr::IncomingPort),

    /// 1 + `n_other` arity Predicate checking injectivity of wires (i.e. wires
    /// are distinct).
    ///
    /// Given a wire and `n_other` other wires, check that the first wire is
    /// distinct from all the other wires. This is not checking all-to-all
    /// distinctness: this is so that the predicate is closed under AND (when
    /// the first argument is identical).
    ///
    /// All `n_other` + 1 constraint variables must be of the same wire type,
    /// either [`HugrVariableID::CopyableWire`] or [`HugrVariableID::LinearWire`].
    IsDistinctFrom {
        /// The number of other nodes, determining the predicate arity.
        n_other: cmp::Reverse<usize>,
    },
}

impl Predicate {
    pub fn new_is_distinct_from(n_other: usize) -> Self {
        Predicate::IsDistinctFrom {
            n_other: cmp::Reverse(n_other),
        }
    }
}

/// A constraint to define patterns on flat hugrs.
pub type Constraint = pm::Constraint<HugrVariableID, Predicate>;

impl pm::ArityPredicate for Predicate {
    fn arity(&self) -> usize {
        use Predicate::*;

        match *self {
            IsOpEqual(_) => 1,
            IsWireSource(_) | IsWireSink(_) => 2,
            IsDistinctFrom { n_other } => n_other.0 + 1,
        }
    }
}

/// Error type for invalid predicates.
#[derive(Debug, Clone, Copy, Error, Display)]
pub enum InvalidPredicateError {
    /// The variable type is invalid.
    #[display("Invalid variable type")]
    InvalidVariableType,
    /// The predicate arity is invalid.
    #[display("Invalid predicate arity")]
    InvalidArity,
}

impl<H: HugrView> pm::Predicate<Circuit<H>, HugrVariableValue> for Predicate {
    fn check(&self, args: &[impl Borrow<HugrVariableValue>], data: &Circuit<H>) -> bool {
        let hugr = data.hugr();
        match self {
            Predicate::IsOpEqual(exp_match_op) => {
                // Get the native hugr node value
                let vals = to_hugr_values_vec::<hugr::Node, _>(args).unwrap();
                let node = vals.into_iter().exactly_one().expect("one variable");

                let match_op: MatchOp = hugr.get_optype(node).into();
                exp_match_op == &match_op
            }
            &Predicate::IsWireSource(exp_out_port) => {
                let (node, (out_node, out_port)): (hugr::Node, (hugr::Node, hugr::OutgoingPort)) =
                    to_hugr_values_tuple(args).unwrap();
                out_port == exp_out_port && node == out_node
            }
            &Predicate::IsWireSink(exp_in_port) => {
                let (node, (out_node, out_port)): (hugr::Node, (hugr::Node, hugr::OutgoingPort)) =
                    to_hugr_values_tuple(args).unwrap();
                let Some((exp_out_node, exp_out_port)) =
                    find_source(node, exp_in_port, data.hugr())
                else {
                    return false;
                };
                exp_out_node == out_node && exp_out_port == out_port
            }
            Predicate::IsDistinctFrom { .. } => {
                // Get the native hugr node values
                let vals = to_hugr_values_vec::<(hugr::Node, hugr::OutgoingPort), _>(args).unwrap();
                let first_val = vals[0];
                vals[1..].iter().all(|&v| v != first_val)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{portmatching::tests::circ_with_copy, Tk2Op};

    use super::*;
    use hugr::{ops::OpType, Direction, IncomingPort, Node, OutgoingPort, Port};
    use itertools::{Either, Itertools};
    use rstest::rstest;

    fn get_nodes_by_tk2op(circ: &Circuit, t2_op: Tk2Op) -> Vec<Node> {
        let t2_op: OpType = t2_op.into();
        circ.hugr()
            .nodes()
            .filter(|n| circ.hugr().get_optype(*n) == &t2_op)
            .collect()
    }

    #[rstest]
    fn test_check_node_op(circ_with_copy: Circuit) {
        let pred = Predicate::IsOpEqual(Tk2Op::Rx.into());
        let rx_ops = get_nodes_by_tk2op(&circ_with_copy, Tk2Op::Rx);
        for rx in rx_ops {
            assert!(<Predicate as pm::Predicate<_, _>>::check(
                &pred,
                &[&HugrVariableValue::Node(rx)],
                &circ_with_copy,
            ));
        }
    }

    fn wire_pred(port: hugr::Port) -> Predicate {
        match port.as_directed() {
            Either::Left(in_port) => Predicate::IsWireSink(in_port),
            Either::Right(out_port) => Predicate::IsWireSource(out_port),
        }
    }

    #[rstest]
    fn test_check_shared_edge(circ_with_copy: Circuit) {
        let rx_ops = get_nodes_by_tk2op(&circ_with_copy, Tk2Op::Rx);

        // valid edges
        let edges: [(Port, Port, bool); 4] = [
            (
                OutgoingPort::from(0).into(),
                IncomingPort::from(0).into(),
                true,
            ),
            (
                IncomingPort::from(1).into(),
                IncomingPort::from(1).into(),
                true,
            ),
            (
                OutgoingPort::from(1).into(),
                IncomingPort::from(0).into(),
                false,
            ),
            (
                OutgoingPort::from(1).into(),
                IncomingPort::from(10).into(),
                false,
            ),
        ];
        for (p1, p2, is_valid) in edges {
            let pred1 = wire_pred(p1);
            let pred2 = wire_pred(p2);

            let node1: HugrVariableValue = rx_ops[0].into();
            let node2: HugrVariableValue = rx_ops[1].into();
            let wire = match p1.as_directed() {
                Either::Left(in_port) => {
                    HugrVariableValue::new_wire_from_sink(rx_ops[0], in_port, circ_with_copy.hugr())
                }
                Either::Right(out_port) => {
                    HugrVariableValue::new_wire_from_source(rx_ops[0], out_port)
                }
            };

            assert!(<Predicate as pm::Predicate::<_, _>>::check(
                &pred1,
                &[&node1, &wire],
                &circ_with_copy
            ));
            assert_eq!(
                <Predicate as pm::Predicate::<_, _>>::check(
                    &pred2,
                    &[&node2, &wire],
                    &circ_with_copy
                ),
                is_valid
            );
        }
    }
}
