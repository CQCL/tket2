//! Circuit Patterns for pattern matching

use hugr::{ops::OpTrait, Node, Port};
use itertools::Itertools;
use portmatching::{patterns::NoRootFound, HashMap, Pattern, SinglePatternMatcher};
use std::fmt::Debug;
use thiserror::Error;

use super::{
    matcher::{validate_circuit_edge, validate_circuit_node},
    PEdge, PNode,
};
use crate::{circuit::Circuit, portmatching::NodeID};

#[cfg(feature = "pyo3")]
use pyo3::{create_exception, exceptions::PyException, pyclass, PyErr};

/// A pattern that match a circuit exactly
#[cfg_attr(feature = "pyo3", pyclass)]
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct CircuitPattern {
    pub(super) pattern: Pattern<NodeID, PNode, PEdge>,
    /// The input ports
    pub(super) inputs: Vec<Vec<(Node, Port)>>,
    /// The output ports
    pub(super) outputs: Vec<(Node, Port)>,
}

impl CircuitPattern {
    /// The number of edges in the pattern.
    pub fn n_edges(&self) -> usize {
        self.pattern.n_edges()
    }

    /// Construct a pattern from a circuit.
    pub fn try_from_circuit(circuit: &impl Circuit) -> Result<Self, InvalidPattern> {
        if circuit.num_gates() == 0 {
            return Err(InvalidPattern::EmptyCircuit);
        }
        let mut pattern = Pattern::new();
        for cmd in circuit.commands() {
            let op = cmd.optype().clone();
            pattern.require(cmd.node().into(), op.try_into().unwrap());
            for in_offset in 0..cmd.input_count() {
                let in_offset = Port::new_incoming(in_offset);
                let edge_prop =
                    PEdge::try_from_port(cmd.node(), in_offset, circuit).expect("Invalid HUGR");
                let (prev_node, prev_port) = circuit
                    .linked_ports(cmd.node(), in_offset)
                    .exactly_one()
                    .ok()
                    .expect("invalid HUGR");
                let prev_node = match edge_prop {
                    PEdge::InternalEdge { .. } => NodeID::HugrNode(prev_node),
                    PEdge::InputEdge { .. } => NodeID::CopyNode(prev_node, prev_port),
                };
                pattern.add_edge(cmd.node().into(), prev_node, edge_prop);
            }
        }
        pattern.set_any_root()?;
        if !pattern.is_valid() {
            return Err(InvalidPattern::NotConnected);
        }
        let (inp, out) = (circuit.input(), circuit.output());
        let inp_ports = circuit.get_optype(inp).signature().output_ports();
        let out_ports = circuit.get_optype(out).signature().input_ports();
        let inputs = inp_ports
            .map(|p| circuit.linked_ports(inp, p).collect())
            .collect_vec();
        let outputs = out_ports
            .map(|p| {
                circuit
                    .linked_ports(out, p)
                    .exactly_one()
                    .ok()
                    .expect("invalid circuit")
            })
            .collect_vec();
        if inputs.iter().flatten().any(|&(n, _)| n == out) {
            // An input is connected to an output => empty qubit, not allowed.
            return Err(InvalidPattern::NotConnected);
        }
        // This is a consequence of the test above.
        debug_assert!(outputs.iter().all(|(n, _)| *n != inp));
        Ok(Self {
            pattern,
            inputs,
            outputs,
        })
    }

    /// Compute the map from pattern nodes to circuit nodes in `circ`.
    pub fn get_match_map(&self, root: Node, circ: &impl Circuit) -> Option<HashMap<Node, Node>> {
        let single_matcher = SinglePatternMatcher::from_pattern(self.pattern.clone());
        single_matcher
            .get_match_map(
                root.into(),
                validate_circuit_node(circ),
                validate_circuit_edge(circ),
            )
            .map(|m| {
                m.into_iter()
                    .filter_map(|(node_p, node_c)| match (node_p, node_c) {
                        (NodeID::HugrNode(node_p), NodeID::HugrNode(node_c)) => {
                            Some((node_p, node_c))
                        }
                        (NodeID::CopyNode(..), NodeID::CopyNode(..)) => None,
                        _ => panic!("Invalid match map"),
                    })
                    .collect()
            })
    }
}

impl Debug for CircuitPattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.pattern.fmt(f)?;
        Ok(())
    }
}

/// Conversion error from circuit to pattern.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum InvalidPattern {
    /// An empty circuit cannot be a pattern.
    #[error("empty circuit is invalid pattern")]
    EmptyCircuit,
    /// Patterns must be connected circuits.
    #[error("pattern is not connected")]
    NotConnected,
}

impl From<NoRootFound> for InvalidPattern {
    fn from(_: NoRootFound) -> Self {
        InvalidPattern::NotConnected
    }
}

#[cfg(feature = "pyo3")]
create_exception!(
    pyrs,
    PyInvalidPatternError,
    PyException,
    "Invalid circuit pattern"
);

#[cfg(feature = "pyo3")]
impl From<InvalidPattern> for PyErr {
    fn from(err: InvalidPattern) -> Self {
        PyInvalidPatternError::new_err(err.to_string())
    }
}

#[cfg(test)]
mod tests {

    use std::collections::HashSet;

    use hugr::builder::{DFGBuilder, Dataflow, DataflowHugr};
    use hugr::extension::prelude::QB_T;
    use hugr::ops::LeafOp;
    use hugr::std_extensions::arithmetic::float_types::FLOAT64_TYPE;
    use hugr::types::FunctionType;
    use hugr::Hugr;

    use crate::extension::REGISTRY;
    use crate::utils::build_simple_circuit;
    use crate::T2Op;

    use super::*;

    fn h_cx() -> Hugr {
        build_simple_circuit(2, |circ| {
            circ.append(T2Op::CX, [0, 1])?;
            circ.append(T2Op::H, [0])?;
            Ok(())
        })
        .unwrap()
    }

    /// A circuit with two rotation gates in sequence, sharing a param
    fn circ_with_copy() -> Hugr {
        let input_t = vec![QB_T, FLOAT64_TYPE];
        let output_t = vec![QB_T];
        let mut h = DFGBuilder::new(FunctionType::new(input_t, output_t)).unwrap();

        let mut inps = h.input_wires();
        let qb = inps.next().unwrap();
        let f = inps.next().unwrap();

        let res = h.add_dataflow_op(T2Op::RxF64, [qb, f]).unwrap();
        let qb = res.outputs().next().unwrap();
        let res = h.add_dataflow_op(T2Op::RxF64, [qb, f]).unwrap();
        let qb = res.outputs().next().unwrap();

        h.finish_hugr_with_outputs([qb], &REGISTRY).unwrap()
    }

    /// A circuit with two rotation gates in parallel, sharing a param
    fn circ_with_copy_disconnected() -> Hugr {
        let input_t = vec![QB_T, QB_T, FLOAT64_TYPE];
        let output_t = vec![QB_T, QB_T];
        let mut h = DFGBuilder::new(FunctionType::new(input_t, output_t)).unwrap();

        let mut inps = h.input_wires();
        let qb1 = inps.next().unwrap();
        let qb2 = inps.next().unwrap();
        let f = inps.next().unwrap();

        let res = h.add_dataflow_op(T2Op::RxF64, [qb1, f]).unwrap();
        let qb1 = res.outputs().next().unwrap();
        let res = h.add_dataflow_op(T2Op::RxF64, [qb2, f]).unwrap();
        let qb2 = res.outputs().next().unwrap();

        h.finish_hugr_with_outputs([qb1, qb2], &REGISTRY).unwrap()
    }

    #[test]
    fn construct_pattern() {
        let hugr = h_cx();

        let p = CircuitPattern::try_from_circuit(&hugr).unwrap();

        let edges: HashSet<_> = p
            .pattern
            .edges()
            .unwrap()
            .iter()
            .map(|e| (e.source.unwrap(), e.target.unwrap()))
            .collect();
        let inp = hugr.input();
        let cx_gate = NodeID::HugrNode(get_nodes_by_t2op(&hugr, T2Op::CX)[0]);
        let h_gate = NodeID::HugrNode(get_nodes_by_t2op(&hugr, T2Op::H)[0]);
        assert_eq!(
            edges,
            [
                (cx_gate, h_gate),
                (cx_gate, NodeID::CopyNode(inp, Port::new_outgoing(0))),
                (cx_gate, NodeID::CopyNode(inp, Port::new_outgoing(1))),
            ]
            .into_iter()
            .collect()
        )
    }

    #[test]
    fn disconnected_pattern() {
        let circ = build_simple_circuit(2, |circ| {
            circ.append(T2Op::X, [0])?;
            circ.append(T2Op::T, [1])?;
            Ok(())
        })
        .unwrap();
        assert_eq!(
            CircuitPattern::try_from_circuit(&circ).unwrap_err(),
            InvalidPattern::NotConnected
        );
    }

    #[test]
    fn pattern_with_empty_qubit() {
        let circ = build_simple_circuit(2, |circ| {
            circ.append(T2Op::X, [0])?;
            Ok(())
        })
        .unwrap();
        assert_eq!(
            CircuitPattern::try_from_circuit(&circ).unwrap_err(),
            InvalidPattern::NotConnected
        );
    }

    fn get_nodes_by_t2op(circ: &impl Circuit, t2_op: T2Op) -> Vec<Node> {
        circ.nodes()
            .filter(|n| {
                let Ok(op): Result<LeafOp, _> = circ.get_optype(*n).clone().try_into() else {
                    return false;
                };
                op == t2_op.into()
            })
            .collect()
    }

    #[test]
    fn pattern_with_copy() {
        let circ = circ_with_copy();
        let pattern = CircuitPattern::try_from_circuit(&circ).unwrap();
        let edges = pattern.pattern.edges().unwrap();
        let rx_ns = get_nodes_by_t2op(&circ, T2Op::RxF64);
        let inp = circ.input();
        for rx_n in rx_ns {
            assert!(edges.iter().any(|e| {
                e.reverse().is_none()
                    && e.source.unwrap() == rx_n.into()
                    && e.target.unwrap() == NodeID::CopyNode(inp, Port::new_outgoing(1))
            }));
        }
    }

    #[test]
    fn pattern_with_copy_disconnected() {
        let circ = circ_with_copy_disconnected();
        assert_eq!(
            CircuitPattern::try_from_circuit(&circ).unwrap_err(),
            InvalidPattern::NotConnected
        );
    }
}
