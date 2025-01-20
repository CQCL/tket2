//! Predicates for pattern matching HUGRs.

use std::fmt::Debug;
use std::{borrow::Borrow, cmp};

use derive_more::{Display, Error};
use hugr::HugrView;
use itertools::Itertools;
use portmatching as pm;

use crate::Circuit;

use super::{HugrVariableID, HugrVariableValue, MatchOp};

/// A constraint to define patterns on flat hugrs.
///
/// See [`Predicate`] for the type of constraints that can be defined.
pub type Constraint = pm::Constraint<HugrVariableID, Predicate>;

/// Predicate for pattern matching on flat hugrs.
///
/// Predicates are boolean-valued functions that can be evaluated given
/// a tuple of [`HugrVariableValue`]s.
///
/// When combined together with a tuple of constraint variables, a predicate
/// defines a constraint. Given a binding from constraint variables to
/// constraint values, a constraint can thus be evaluated by evaluating
/// the predicate on the bound values.
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

/// The valid "signatures" of predicates, useful for argument parsing.
enum PredicateArguments {
    /// Argument for a [`Predicate::IsOpEqual`] predicate.
    Node(hugr::Node),
    /// Arguments for a [`Predicate::IsWireSource`] or [`Predicate::IsWireSink`] predicate.
    NodeAndWire(hugr::Node, hugr::Wire),
    /// Arguments for a [`Predicate::IsDistinctFrom`] predicate.
    ManyNodes(Vec<hugr::Node>),
}

impl Predicate {
    /// Create a new [`Predicate::IsDistinctFrom`] predicate.
    pub fn new_is_distinct_from(n_other: usize) -> Self {
        Predicate::IsDistinctFrom {
            n_other: cmp::Reverse(n_other),
        }
    }

    /// Parse predicate arguments into constraint values of the expected types
    fn parse_arguments<'b, B>(
        &self,
        args: impl IntoIterator<Item = &'b B>,
    ) -> Result<PredicateArguments, InvalidPredicateError>
    where
        B: Borrow<HugrVariableValue> + 'b,
    {
        let args = args.into_iter();
        match self {
            Predicate::IsOpEqual(..) => {
                let arg0 = args
                    .exactly_one()
                    .map_err(|_| InvalidPredicateError::InvalidArity)?;
                let node = (*arg0.borrow())
                    .try_into()
                    .map_err(|_| InvalidPredicateError::InvalidVariableType)?;
                Ok(PredicateArguments::Node(node))
            }
            Predicate::IsWireSource(..) | Predicate::IsWireSink(..) => {
                let (arg0, arg1) = args
                    .collect_tuple()
                    .ok_or(InvalidPredicateError::InvalidArity)?;
                let node = (*arg0.borrow())
                    .try_into()
                    .map_err(|_| InvalidPredicateError::InvalidVariableType)?;
                let (out_node, out_port) = (*arg1.borrow())
                    .try_into()
                    .map_err(|_| InvalidPredicateError::InvalidVariableType)?;
                Ok(PredicateArguments::NodeAndWire(
                    node,
                    hugr::Wire::new(out_node, out_port),
                ))
            }
            Predicate::IsDistinctFrom { .. } => {
                let nodes = args
                    .into_iter()
                    .map(|arg| {
                        (*arg.borrow())
                            .try_into()
                            .map_err(|_| InvalidPredicateError::InvalidVariableType)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(PredicateArguments::ManyNodes(nodes))
            }
        }
    }
}

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
                let Ok(PredicateArguments::Node(node)) = self.parse_arguments(args) else {
                    panic!("ill-formed constraint");
                };

                let match_op: MatchOp = hugr.get_optype(node).into();
                exp_match_op == &match_op
            }
            &Predicate::IsWireSource(exp_out_port) => {
                let Ok(PredicateArguments::NodeAndWire(node, wire)) = self.parse_arguments(args)
                else {
                    panic!("ill-formed constraint");
                };
                wire.source() == exp_out_port && node == wire.node()
            }
            &Predicate::IsWireSink(exp_in_port) => {
                let Ok(PredicateArguments::NodeAndWire(node, wire)) = self.parse_arguments(args)
                else {
                    panic!("ill-formed constraint");
                };
                let Some((exp_out_node, exp_out_port)) =
                    data.hugr().single_linked_output(node, exp_in_port)
                else {
                    return false;
                };
                exp_out_node == wire.node() && exp_out_port == wire.source()
            }
            Predicate::IsDistinctFrom { .. } => {
                let Ok(PredicateArguments::ManyNodes(nodes)) = self.parse_arguments(args) else {
                    panic!("ill-formed constraint");
                };
                let first_node = nodes[0];
                nodes[1..].iter().all(|&n| n != first_node)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{portmatching::tests::circ_with_copy, Tk2Op};

    use super::*;
    use hugr::{ops::OpType, IncomingPort, Node, OutgoingPort, Port};
    use itertools::Either;
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
                Either::Left(in_port) => HugrVariableValue::new_wire_from_incoming(
                    rx_ops[0],
                    in_port,
                    circ_with_copy.hugr(),
                ),
                Either::Right(out_port) => {
                    HugrVariableValue::new_wire_from_outgoing(rx_ops[0], out_port)
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
