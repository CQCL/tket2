//! Constraint definitions for pattern matching on flat hugrs.

use std::borrow::Borrow;
use std::fmt::Debug;

use derive_more::{Display, Error};
use hugr::HugrView;
use itertools::Either;
use itertools::Itertools;
use portmatching as pm;

use crate::Circuit;

use super::HugrVariableID;
use super::HugrVariableValue;
use super::MatchOp;

/// Predicate for pattern matching on flat hugrs.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
pub enum Predicate {
    /// Unary predicate checking the node OpType
    ///
    /// Constraint variable must be of type [`HugrVariableID::Node`].
    NodeOp(MatchOp),
    /// Predicate checking the ports of a wire, corresponding to a SSA value.
    ///
    /// Every wire has one outgoing and one or more incoming ports. The outgoing
    /// port is optional, for the case where it is an input value. The incoming
    /// ports must be an exhaustive list of all uses of the wire within the
    /// pattern.
    ///
    /// Constraint variables must be of type [`HugrVariableID::Port`].
    Wire {
        /// Whether the wire outgoing port is specified.
        has_out_port: bool,
        /// The number of incoming ports of the wire.
        n_in_ports: usize,
    },
    /// The node is not equal to any of the other `n_other` nodes
    ///
    /// Constraint variables must be of type [`HugrVariableID::Node`].
    IsNotEqual {
        /// The number of other nodes, determining the predicate arity.
        n_other: usize,
    },
}

/// A constraint to define patterns on flat hugrs.
pub type Constraint = pm::Constraint<HugrVariableID, Predicate>;

impl pm::ArityPredicate for Predicate {
    fn arity(&self) -> usize {
        match *self {
            Predicate::NodeOp(_) => 1,
            Predicate::Wire {
                has_out_port,
                n_in_ports,
            } => {
                let n_out_ports = has_out_port as usize;
                n_out_ports + n_in_ports
            }
            Predicate::IsNotEqual { n_other } => n_other + 1,
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

impl<H: HugrView> pm::Predicate<Circuit<H>> for Predicate {
    type InvalidPredicateError = InvalidPredicateError;

    fn check(
        &self,
        data: &Circuit<H>,
        args: &[impl Borrow<HugrVariableValue>],
    ) -> Result<bool, InvalidPredicateError> {
        let hugr = data.hugr();
        match self {
            Predicate::NodeOp(exp_match_op) => {
                // Get the native hugr node value
                let vals = to_hugr_values::<hugr::Node, _>(args)?;
                let node = vals.into_iter().exactly_one().expect("one variable");

                let match_op: MatchOp = hugr.get_optype(node).into();
                Ok(exp_match_op == &match_op)
            }
            &Predicate::Wire { has_out_port, .. } => {
                // Get the native hugr values
                let vals = to_hugr_values::<(hugr::Node, hugr::Port), _>(args)?;

                if has_out_port {
                    // In this case, the wire in the pattern must match exactly
                    // with the wire in the circuit.

                    // Get the outgoing port
                    let &(out_node, out_port) =
                        vals.first().ok_or(InvalidPredicateError::InvalidArity)?;
                    let out_port = out_port
                        .as_outgoing()
                        .map_err(|_| InvalidPredicateError::InvalidVariableType)?;

                    // Get the incoming ports
                    let exp_in_ports = vals[1..]
                        .iter()
                        .map(|&(n, p)| (n, p.as_incoming().expect("wrong port type")))
                        .sorted();

                    // Get the actual incoming ports in the hugr
                    let actual_in_ports = hugr.linked_inputs(out_node, out_port).sorted();

                    Ok(exp_in_ports.eq(actual_in_ports))
                } else {
                    // In this case we just need to check that the incoming
                    // ports are all connected to the same wire.

                    // nothing to do if there is just one port
                    if vals.len() <= 1 {
                        return Ok(true);
                    }

                    // The outgoing port of a wire is unique, so check that it
                    // is the same for all the incoming ports.
                    let mut first_out_port = None;
                    for (node, port) in vals {
                        let in_port = port
                            .as_incoming()
                            .map_err(|_| InvalidPredicateError::InvalidVariableType)?;
                        let Some(out_port) = hugr.single_linked_output(node, in_port) else {
                            return Ok(false);
                        };
                        if first_out_port.is_none() {
                            first_out_port = Some(out_port);
                        } else if out_port != first_out_port.unwrap() {
                            return Ok(false);
                        }
                    }
                    Ok(true)
                }
            }
            Predicate::IsNotEqual { .. } => {
                // Get the native hugr node values
                let vals = to_hugr_values::<hugr::Node, _>(args)?;
                let first_val = vals[0];
                Ok(vals[1..].iter().all(|&v| v != first_val))
            }
        }
    }
}

/// Convert a port to its unique outgoing port.
///
/// If the port is outgoing, leave it unchanged, otherwise find the unique
/// outgoing port that it is attached to.
fn to_out_port(
    hugr: &impl HugrView,
    (node, port): (hugr::Node, hugr::Port),
) -> (hugr::Node, hugr::OutgoingPort) {
    match port.as_directed() {
        Either::Left(in_port) => hugr
            .single_linked_output(node, in_port)
            .expect("not a single outgoing port"),
        Either::Right(out_port) => (node, out_port),
    }
}

fn to_hugr_values<'b, V, B>(
    args: impl IntoIterator<Item = &'b B>,
) -> Result<Vec<V>, InvalidPredicateError>
where
    B: Borrow<HugrVariableValue> + 'b,
    V: TryFrom<HugrVariableValue>,
    V::Error: Debug,
{
    args.into_iter()
        .map(|arg| {
            let var = arg.borrow().clone();
            var.try_into()
                .map_err(|_| InvalidPredicateError::InvalidVariableType)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use crate::{portmatching::tests::circ_with_copy, Tk2Op};

    use super::*;
    use hugr::{ops::OpType, Direction, IncomingPort, Node, OutgoingPort, Port};
    use itertools::Itertools;
    use portmatching::Predicate;
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
        let pred = super::Predicate::NodeOp(Tk2Op::Rx.into());
        let rx_ops = get_nodes_by_tk2op(&circ_with_copy, Tk2Op::Rx);
        for rx in rx_ops {
            assert!(pred
                .check(&circ_with_copy, &[&HugrVariableValue::Node(rx)])
                .unwrap());
        }
    }

    fn binary_wire_pred(dir: Direction) -> super::Predicate {
        let has_out_port = dir == Direction::Outgoing;
        super::Predicate::Wire {
            has_out_port,
            n_in_ports: 1 + !has_out_port as usize,
        }
    }

    #[rstest]
    fn test_check_shared_edge(circ_with_copy: Circuit) {
        let rx_ops = get_nodes_by_tk2op(&circ_with_copy, Tk2Op::Rx);

        // valid edges
        let edges: [(Port, Port); 2] = [
            (OutgoingPort::from(0).into(), IncomingPort::from(0).into()),
            (IncomingPort::from(1).into(), IncomingPort::from(1).into()),
        ];
        for (p1, p2) in edges {
            let pred = binary_wire_pred(p1.direction());
            let vals = rx_ops
                .iter()
                .copied()
                .zip([p1, p2])
                .map(HugrVariableValue::from)
                .collect_vec();
            assert!(pred.check(&circ_with_copy, vals.as_slice()).unwrap());
        }

        // invalid edges
        let edges: [(Port, Port); 2] = [
            (OutgoingPort::from(1).into(), IncomingPort::from(0).into()),
            (OutgoingPort::from(1).into(), IncomingPort::from(10).into()),
        ];
        for (p1, p2) in edges {
            let pred = binary_wire_pred(p1.direction());
            let vals = rx_ops
                .iter()
                .copied()
                .zip([p1, p2])
                .map(HugrVariableValue::from)
                .collect_vec();
            assert!(!pred.check(&circ_with_copy, vals.as_slice()).unwrap());
        }
    }
}
