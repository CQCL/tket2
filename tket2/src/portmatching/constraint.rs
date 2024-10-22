//! Constraint definitions for pattern matching on flat hugrs.

use std::borrow::Borrow;
use std::fmt::Debug;

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
    /// Unary predicate checking the port offset.
    ///
    /// Constraint variable must be of type [`HugrVariableID::Port`].
    PortOffset(hugr::Port),
    /// Binary predicate checking that two ports are attached to the same wire.
    ///
    /// Constraint variables must be of type [`HugrVariableID::Port`].
    /// Either port may be incoming or outgoing, (but every wire has at most one
    /// outgoing port)
    CopyableWire,
    /// Binary predicate checking the two ports of a linear wire.
    ///
    /// Every linear wire has two ports, one outgoing and one incoming.
    /// Hence one argument must be the outgoing port of the linear wire, the
    /// other the incoming port.
    ///
    /// Constraint variables must be of type [`HugrVariableID::Port`].
    LinearWire,
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
            Predicate::NodeOp(_) | Predicate::PortOffset(_) => 1,
            Predicate::CopyableWire | Predicate::LinearWire => 2,
            Predicate::IsNotEqual { n_other } => n_other + 1,
        }
    }
}

impl<H: HugrView> pm::Predicate<Circuit<H>> for Predicate {
    fn check(&self, data: &Circuit<H>, args: &[impl Borrow<HugrVariableValue>]) -> bool {
        let hugr = data.hugr();
        match self {
            Predicate::NodeOp(exp_match_op) => {
                // Get the native hugr node value
                let vals = unwrap_to_hugr_values::<hugr::Node, _>(args);
                let node = vals.into_iter().exactly_one().expect("one variable");

                let match_op: MatchOp = hugr.get_optype(node).into();
                exp_match_op == &match_op
            }
            Predicate::PortOffset(exp_port) => {
                // Get the native hugr port value
                let vals = unwrap_to_hugr_values::<(hugr::Node, hugr::Port), _>(args);
                let (_, port) = vals.into_iter().exactly_one().expect("one variable");

                // Check that the port offset is correct
                exp_port == &port
            }
            Predicate::CopyableWire | Predicate::LinearWire => {
                // Get the native hugr values
                let (arg1, arg2) = unwrap_to_hugr_values::<(hugr::Node, hugr::Port), _>(args)
                    .into_iter()
                    .collect_tuple()
                    .expect("two variables");

                // Convert all ports to Outgoing ports, as they are unique
                let out1 = to_out_port(hugr, arg1);
                let out2 = to_out_port(hugr, arg2);

                out1 == out2
            }
            Predicate::IsNotEqual { .. } => {
                // Get the native hugr node values
                let vals = unwrap_to_hugr_values::<hugr::Node, _>(args);
                let first_val = vals[0];
                vals[1..].iter().all(|&v| v != first_val)
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
    use hugr::{ops::OpType, IncomingPort, Node, OutgoingPort, Port};
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
        let edges: [(Port, Port); 2] = [
            (OutgoingPort::from(0).into(), IncomingPort::from(0).into()),
            (IncomingPort::from(1).into(), IncomingPort::from(1).into()),
        ];
        for (p1, p2) in edges {
            let pred = super::Predicate::CopyableWire;
            let vals = rx_ops
                .iter()
                .copied()
                .zip([p1, p2])
                .map(HugrVariableValue::from)
                .collect_vec();
            assert!(pred.check(&circ_with_copy, vals.as_slice()));
        }

        // Check port offset predicate
        for (n, &p) in edges
            .iter()
            .flat_map(|(p1, p2)| [(rx_ops[0], p1), (rx_ops[1], p2)])
        {
            let pred = super::Predicate::PortOffset(p);
            let val = HugrVariableValue::from((n, p));
            assert!(pred.check(&circ_with_copy, &[&val]));
        }

        // invalid edges
        let edges: [(Port, Port); 2] = [
            (OutgoingPort::from(1).into(), IncomingPort::from(0).into()),
            (OutgoingPort::from(1).into(), OutgoingPort::from(10).into()),
        ];
        for (p1, p2) in edges {
            let pred = super::Predicate::CopyableWire;
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
