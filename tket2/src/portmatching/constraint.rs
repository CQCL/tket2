//! Constraint definitions for pattern matching on flat hugrs.

use std::borrow::Borrow;
use std::fmt::Debug;

use hugr::HugrView;
use itertools::Either;
use portmatching as pm;

use crate::Circuit;

use super::HugrVariableID;
use super::HugrVariableValue;
use super::MatchOp;

/// Predicate for pattern matching on flat hugrs.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Predicate {
    /// Unary predicate checking the node OpType
    NodeOp(MatchOp),
    /// Binary predicate checking that two ports are on the same edge.
    ShareEdge(hugr::Port, hugr::Port),
    /// The node is not equal to any of the other `n_other` nodes
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
            Predicate::ShareEdge(..) => 2,
            Predicate::IsNotEqual { n_other } => n_other + 1,
        }
    }
}

impl<H: HugrView> pm::Predicate<Circuit<H>> for Predicate {
    fn check(&self, data: &Circuit<H>, args: &[impl Borrow<HugrVariableValue>]) -> bool {
        match self {
            Predicate::NodeOp(exp_match_op) => {
                let vals = unwrap_to_hugr_values::<hugr::Node, _>(args);
                let node = vals[0];
                let match_op: MatchOp = data.hugr().get_optype(node).into();
                exp_match_op == &match_op
            }
            Predicate::ShareEdge(exp_port1, exp_port2) => {
                // Get the native hugr values
                let vals = unwrap_to_hugr_values::<(hugr::Node, hugr::Port), _>(args);

                // Check that the port offsets are correct
                if &vals[0].1 != exp_port1 || &vals[1].1 != exp_port2 {
                    return false;
                }

                // Convert all ports to Outgoing ports, as they are unique
                let to_out_port = |(node, port): (hugr::Node, hugr::Port)| match port.as_directed()
                {
                    Either::Left(in_port) => data
                        .hugr()
                        .single_linked_output(node, in_port)
                        .expect("not a single outgoing port"),
                    Either::Right(out_port) => (node, out_port),
                };

                to_out_port(vals[0]) == to_out_port(vals[1])
            }
            Predicate::IsNotEqual { .. } => {
                // Get the native hugr values
                let vals = unwrap_to_hugr_values::<hugr::Node, _>(args);
                let first_val = vals[0];
                vals[1..].iter().all(|&v| v != first_val)
            }
        }
    }
}

fn unwrap_to_hugr_values<'b, V, B>(args: impl IntoIterator<Item = &'b B>) -> Vec<V>
where
    B: Borrow<HugrVariableValue> + 'b,
    V: TryFrom<HugrVariableValue>,
    V::Error: Debug,
{
    args.into_iter()
        .map(|arg| {
            let var = arg.borrow().clone();
            var.try_into().expect("invalid variable binding type")
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use crate::{portmatching::tests::circ_with_copy, Tk2Op};

    use super::*;
    use hugr::{ops::OpType, IncomingPort, Node, OutgoingPort};
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
            assert!(pred.check(&circ_with_copy, &[&HugrVariableValue::Node(rx)]));
        }
    }

    #[rstest]
    fn test_check_shared_edge(circ_with_copy: Circuit) {
        let rx_ops = get_nodes_by_tk2op(&circ_with_copy, Tk2Op::Rx);

        // valid edges
        let edges = [
            (OutgoingPort::from(0).into(), IncomingPort::from(0).into()),
            (IncomingPort::from(1).into(), IncomingPort::from(1).into()),
        ];
        for (p1, p2) in edges {
            let pred = super::Predicate::ShareEdge(p1, p2);
            let vals = rx_ops
                .iter()
                .copied()
                .zip([p1, p2])
                .map(HugrVariableValue::from)
                .collect_vec();
            assert!(pred.check(&circ_with_copy, vals.as_slice()));
        }

        // invalid edges
        let edges = [
            (OutgoingPort::from(1).into(), IncomingPort::from(0).into()),
            (OutgoingPort::from(1).into(), OutgoingPort::from(10).into()),
        ];
        for (p1, p2) in edges {
            let pred = super::Predicate::ShareEdge(p1, p2);
            let vals = rx_ops
                .iter()
                .copied()
                .zip([p1, p2])
                .map(HugrVariableValue::from)
                .collect_vec();
            assert!(!pred.check(&circ_with_copy, vals.as_slice()));
        }
    }
}
