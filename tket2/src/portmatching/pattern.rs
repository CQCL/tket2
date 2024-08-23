//! Circuit Patterns for pattern matching

use std::collections::BTreeSet;

use hugr::Port;
use hugr::{Direction, PortIndex};
use itertools::Itertools;
use portmatching::Pattern;

use super::constraint::constraint_key;
use super::indexing::PatternOpLocation;
use super::predicate::Predicate;
use super::Constraint;
use crate::static_circ::{OpLocation, StaticSizeCircuit};

impl Pattern for StaticSizeCircuit {
    type Constraint = Constraint;

    fn to_constraint_vec(&self) -> Vec<Self::Constraint> {
        let mut constraints = Vec::new();

        let starts = self.find_qubit_starts().unwrap();
        let to_pattern_loc = |loc: &OpLocation| {
            let (qubit_path, start) = starts[loc.qubit.0];
            let offset = (loc.op_idx as i8) - (start as i8);
            PatternOpLocation::new(qubit_path, offset)
        };
        // Keep one location per op
        let mut known_locations = BTreeSet::new();

        for loc in self.all_locations() {
            constraints.push(
                Constraint::try_new(
                    Predicate::IsOp {
                        op: self.get(loc).unwrap().clone(),
                    },
                    vec![to_pattern_loc(&loc)],
                )
                .unwrap(),
            );
            if loc.op_idx > 0 {
                let in_port = self.qubit_port(loc);
                let (out_port, prev_loc) = self
                    .linked_op(loc, Port::new(Direction::Incoming, in_port))
                    .unwrap();
                constraints.push(
                    Constraint::try_new(
                        Predicate::Link {
                            out_port: out_port.index(),
                            in_port,
                        },
                        vec![to_pattern_loc(&prev_loc), to_pattern_loc(&loc)],
                    )
                    .unwrap(),
                );
            }
            let loc_0 = to_pattern_loc(&self.equivalent_location(loc, 0).unwrap());
            let other_locations = known_locations
                .iter()
                .copied()
                .filter(|l| l != &loc_0)
                .collect_vec();
            if !other_locations.is_empty() {
                constraints.push(
                    Constraint::try_new(
                        Predicate::NotEq {
                            n_other: other_locations.len(),
                        },
                        Vec::from_iter(
                            [to_pattern_loc(&loc)]
                                .into_iter()
                                .chain(other_locations.iter().copied()),
                        ),
                    )
                    .unwrap(),
                );
            }
            known_locations.insert(loc_0);
        }
        constraints.sort_unstable_by(|c1, c2| constraint_key(c1).cmp(&constraint_key(c2)));
        constraints
    }
}

#[cfg(test)]
mod tests {

    use std::collections::HashSet;

    use cool_asserts::assert_matches;
    use hugr::builder::{DFGBuilder, Dataflow, DataflowHugr};
    use hugr::extension::prelude::QB_T;
    use hugr::ops::OpType;
    use hugr::std_extensions::arithmetic::float_types::FLOAT64_TYPE;
    use hugr::types::Signature;

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
        let input_t = vec![QB_T, FLOAT64_TYPE];
        let output_t = vec![QB_T];
        let mut h = DFGBuilder::new(Signature::new(input_t, output_t)).unwrap();

        let mut inps = h.input_wires();
        let qb = inps.next().unwrap();
        let f = inps.next().unwrap();

        let res = h.add_dataflow_op(Tk2Op::RxF64, [qb, f]).unwrap();
        let qb = res.outputs().next().unwrap();
        let res = h.add_dataflow_op(Tk2Op::RxF64, [qb, f]).unwrap();
        let qb = res.outputs().next().unwrap();

        h.finish_hugr_with_outputs([qb], &REGISTRY).unwrap().into()
    }

    /// A circuit with two rotation gates in parallel, sharing a param
    // fn circ_with_copy_disconnected() -> Circuit {
    //     let input_t = vec![QB_T, QB_T, FLOAT64_TYPE];
    //     let output_t = vec![QB_T, QB_T];
    //     let mut h = DFGBuilder::new(Signature::new(input_t, output_t)).unwrap();

    //     let mut inps = h.input_wires();
    //     let qb1 = inps.next().unwrap();
    //     let qb2 = inps.next().unwrap();
    //     let f = inps.next().unwrap();

    //     let res = h.add_dataflow_op(Tk2Op::RxF64, [qb1, f]).unwrap();
    //     let qb1 = res.outputs().next().unwrap();
    //     let res = h.add_dataflow_op(Tk2Op::RxF64, [qb2, f]).unwrap();
    //     let qb2 = res.outputs().next().unwrap();

    //     h.finish_hugr_with_outputs([qb1, qb2], &REGISTRY)
    //         .unwrap()
    //         .into()
    // }

    #[test]
    fn construct_pattern() {
        let circ = h_cx();

        insta::assert_debug_snapshot!(circ.to_constraint_vec());
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
        circ.to_constraint_vec();
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
        circ.to_constraint_vec();
    }

    // fn get_nodes_by_tk2op(circ: &Circuit, t2_op: Tk2Op) -> Vec<Node> {
    //     let t2_op: OpType = t2_op.into();
    //     circ.hugr()
    //         .nodes()
    //         .filter(|n| circ.hugr().get_optype(*n) == &t2_op)
    //         .collect()
    // }

    // #[test]
    // fn pattern_with_copy() {
    //     let circ = circ_with_copy();
    //     let pattern = CircuitPattern::try_from_circuit(&circ).unwrap();
    //     let edges = pattern.pattern.edges().unwrap();
    //     let rx_ns = get_nodes_by_tk2op(&circ, Tk2Op::RxF64);
    //     let inp = circ.input_node();
    //     for rx_n in rx_ns {
    //         assert!(edges.iter().any(|e| {
    //             e.reverse().is_none()
    //                 && e.source.unwrap() == rx_n.into()
    //                 && e.target.unwrap() == NodeID::new_copy(inp, 1)
    //         }));
    //     }
    // }

    // #[test]
    // fn pattern_with_copy_disconnected() {
    //     let circ = circ_with_copy_disconnected();
    //     assert_eq!(
    //         CircuitPattern::try_from_circuit(&circ).unwrap_err(),
    //         InvalidPattern::NotConnected
    //     );
    // }
}
