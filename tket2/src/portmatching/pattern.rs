//! Circuit Patterns for pattern matching

use std::collections::BTreeSet;

use hugr::Port;
use hugr::{Direction, PortIndex};
use itertools::Itertools;
use portmatching::Pattern;
use thiserror::Error;

use super::constraint::constraint_key;
use super::indexing::{DisconnectedCircuit, PatternOpPosition};
use super::predicate::Predicate;
use super::Constraint;
use crate::static_circ::{OpPosition, StaticQubitIndex, StaticSizeCircuit};

#[derive(Debug, Clone, Copy, Error)]
pub enum InvalidStaticPattern {
    #[error("pattern is disconnected")]
    Disconnected,
    #[error("pattern has no qubits")]
    EmptyPattern,
    #[error("Qubit {0:?} has no operations")]
    EmptyQubit(StaticQubitIndex),
}

impl From<DisconnectedCircuit> for InvalidStaticPattern {
    fn from(_: DisconnectedCircuit) -> Self {
        InvalidStaticPattern::Disconnected
    }
}

impl Pattern for StaticSizeCircuit {
    type Constraint = Constraint;
    type Error = InvalidStaticPattern;

    fn try_to_constraint_vec(&self) -> Result<Vec<Self::Constraint>, Self::Error> {
        if self.qubit_count() == 0 {
            return Err(InvalidStaticPattern::EmptyPattern);
        } else if let Some(q) = self.qubits_iter().find(|q| self.qubit_ops(*q).is_empty()) {
            return Err(InvalidStaticPattern::EmptyQubit(q));
        }

        let mut constraints = Vec::new();

        let starts = self.find_qubit_starts()?;
        let to_pattern_pos = |pos: OpPosition| {
            let (qubit_path, start) = starts[pos.qubit.0];
            let offset = (pos.index as i8) - (start as i8);
            PatternOpPosition::new(qubit_path, offset)
        };

        // Add IsOp and SameOp constraints
        for op in self.ops_iter() {
            let all_pos = self
                .positions(op)
                .unwrap()
                .iter()
                .copied()
                .map(to_pattern_pos)
                .collect_vec();
            // Add IsOp on the first location
            let pred = Predicate::IsOp {
                op: self.get(op).unwrap().op,
            };
            constraints.push(Constraint::try_new(pred, vec![all_pos[0]]).unwrap());

            // Add SameOp
            let arity = all_pos.len();
            if arity > 1 {
                let pred = Predicate::SameOp { arity };
                constraints.push(Constraint::try_new(pred, all_pos).unwrap());
            }
        }

        // Add Link constraints
        for pos in self.positions_iter() {
            if pos.index > 0 {
                let in_port = self.position_offset(pos).unwrap();
                let op = self.at_position(pos).unwrap();
                let (out_port, prev_op) = self
                    .linked_op(op, Port::new(Direction::Incoming, in_port))
                    .unwrap();
                let prev_pos = self.get_position(prev_op, out_port.index()).unwrap();
                constraints.push(
                    Constraint::try_new(
                        Predicate::Link {
                            out_port: out_port.index(),
                            in_port,
                        },
                        vec![to_pattern_pos(prev_pos), to_pattern_pos(pos)],
                    )
                    .unwrap(),
                );
            }
        }

        // Add DistinctQubits constraints
        let qubit_starts: BTreeSet<_> = starts
            .iter()
            .map(|&(qubit, _)| PatternOpPosition { qubit, op_idx: 0 })
            .collect();
        for &loc in &qubit_starts {
            let mut starts = vec![loc];
            starts.extend(qubit_starts.range(..loc).copied());
            if starts.len() > 1 {
                let n_qubits = starts.len();
                constraints.push(
                    Constraint::try_new(Predicate::DistinctQubits { n_qubits }, starts).unwrap(),
                );
            }
        }

        constraints.sort_unstable_by(|c1, c2| constraint_key(c1).cmp(&constraint_key(c2)));
        Ok(constraints)
    }
}

#[cfg(test)]
mod tests {

    use std::collections::HashSet;

    use cool_asserts::assert_matches;
    use hugr::builder::{DFGBuilder, Dataflow, DataflowHugr};
    use hugr::extension::prelude::QB_T;
    use hugr::ops::OpType;
    use hugr::types::Signature;

    use crate::extension::angle::ANGLE_TYPE;
    use crate::extension::REGISTRY;
    use crate::portmatching::NodeID;
    use crate::utils::build_simple_circuit;
    use crate::{Circuit, Tk2Op};

    use super::*;

    fn h_cx() -> StaticSizeCircuit {
        let circ = build_simple_circuit(2, |circ| {
            circ.append(Tk2Op::CX, [0, 1])?;
            circ.append(Tk2Op::H, [0])?;
            Ok(())
        })
        .unwrap();
        StaticSizeCircuit::try_from(&circ).unwrap()
    }

    /// A circuit with two rotation gates in sequence, sharing a param
    fn circ_with_copy() -> Circuit {
        let input_t = vec![QB_T, ANGLE_TYPE];
        let output_t = vec![QB_T];
        let mut h = DFGBuilder::new(Signature::new(input_t, output_t)).unwrap();

        let mut inps = h.input_wires();
        let qb = inps.next().unwrap();
        let f = inps.next().unwrap();

        let res = h.add_dataflow_op(Tk2Op::Rx, [qb, f]).unwrap();
        let qb = res.outputs().next().unwrap();
        let res = h.add_dataflow_op(Tk2Op::Rx, [qb, f]).unwrap();
        let qb = res.outputs().next().unwrap();

        h.finish_hugr_with_outputs([qb], &REGISTRY).unwrap().into()
    }

    /// A circuit with two rotation gates in parallel, sharing a param
    #[ignore = "reason"]
    fn circ_with_copy_disconnected() -> Circuit {
        let input_t = vec![QB_T, QB_T, ANGLE_TYPE];
        let output_t = vec![QB_T, QB_T];
        let mut h = DFGBuilder::new(Signature::new(input_t, output_t)).unwrap();

        let mut inps = h.input_wires();
        let qb1 = inps.next().unwrap();
        let qb2 = inps.next().unwrap();
        let f = inps.next().unwrap();

        let res = h.add_dataflow_op(Tk2Op::Rx, [qb1, f]).unwrap();
        let qb1 = res.outputs().next().unwrap();
        let res = h.add_dataflow_op(Tk2Op::Rx, [qb2, f]).unwrap();
        let qb2 = res.outputs().next().unwrap();

        h.finish_hugr_with_outputs([qb1, qb2], &REGISTRY)
            .unwrap()
            .into()
    }

    #[test]
    fn construct_pattern() {
        let circ = h_cx();

        insta::assert_debug_snapshot!(circ.try_to_constraint_vec().unwrap());
    }

    #[test]
    #[should_panic]
    fn disconnected_pattern() {
        let circ = build_simple_circuit(2, |circ| {
            circ.append(Tk2Op::X, [0])?;
            circ.append(Tk2Op::T, [1])?;
            Ok(())
        })
        .unwrap();
        let circ = StaticSizeCircuit::try_from(&circ).unwrap();
        circ.try_to_constraint_vec().unwrap();
    }

    #[test]
    #[should_panic]
    fn pattern_with_empty_qubit() {
        let circ = build_simple_circuit(2, |circ| {
            circ.append(Tk2Op::X, [0])?;
            Ok(())
        })
        .unwrap();
        let circ = StaticSizeCircuit::try_from(&circ).unwrap();
        circ.try_to_constraint_vec().unwrap();
    }

    fn get_nodes_by_tk2op(circ: &Circuit, t2_op: Tk2Op) -> Vec<Node> {
        let t2_op: OpType = t2_op.into();
        circ.hugr()
            .nodes()
            .filter(|n| circ.hugr().get_optype(*n) == &t2_op)
            .collect()
    }

    #[test]
    #[ignore = "reason"]
    fn pattern_with_copy() {
        let circ = circ_with_copy();
        let pattern = CircuitPattern::try_from_circuit(&circ).unwrap();
        let edges = pattern.pattern.edges().unwrap();
        let rx_ns = get_nodes_by_tk2op(&circ, Tk2Op::Rx);
        let inp = circ.input_node();
        for rx_n in rx_ns {
            assert!(edges.iter().any(|e| {
                e.reverse().is_none()
                    && e.source.unwrap() == rx_n.into()
                    && e.target.unwrap() == NodeID::new_copy(inp, 1)
            }));
        }
    }

    // #[test]
    // fn pattern_with_copy_disconnected() {
    //     let circ = circ_with_copy_disconnected();
    //     assert_eq!(
    //         CircuitPattern::try_from_circuit(&circ).unwrap_err(),
    //         InvalidPattern::NotConnected
    //     );
    // }
}
