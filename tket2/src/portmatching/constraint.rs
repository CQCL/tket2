//! Constraint definitions for pattern matching on flat hugrs.

use std::borrow::Borrow;
use std::fmt::Debug;

use hugr::HugrView;
use itertools::Either;
use itertools::Itertools;

use crate::Circuit;

use super::HugrVariableID;
use super::HugrVariableValue;
use super::MatchOp;

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
