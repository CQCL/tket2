//! Circuit Patterns for pattern matching

use hugr::{ops::OpTrait, Node, Port};
use itertools::Itertools;
use portmatching::{patterns::NoRootFound, HashMap, Pattern, SinglePatternMatcher};
use std::fmt::Debug;
use thiserror::Error;

use super::{
    matcher::{validate_unweighted_edge, validate_weighted_node},
    PEdge, PNode,
};
use crate::circuit::Circuit;

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

/// A pattern that match a circuit exactly
#[cfg_attr(feature = "pyo3", pyclass)]
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct CircuitPattern {
    pub(super) pattern: Pattern<Node, PNode, PEdge>,
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
    pub fn try_from_circuit<C: Circuit>(circuit: &C) -> Result<Self, InvalidPattern> {
        if circuit.num_gates() == 0 {
            return Err(InvalidPattern::EmptyCircuit);
        }
        let mut pattern = Pattern::new();
        for cmd in circuit.commands() {
            let op = circuit.command_optype(&cmd).clone();
            pattern.require(cmd.node(), op.try_into().unwrap());
            for out_offset in 0..cmd.outputs().len() {
                let out_offset = Port::new_outgoing(out_offset);
                for (next_node, in_offset) in circuit.linked_ports(cmd.node(), out_offset) {
                    if circuit.get_optype(next_node).tag() != hugr::ops::OpTag::Output {
                        pattern.add_edge(cmd.node(), next_node, (out_offset, in_offset));
                    }
                }
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
            .collect();
        let outputs = out_ports
            .map(|p| {
                circuit
                    .linked_ports(out, p)
                    .exactly_one()
                    .ok()
                    .expect("invalid circuit")
            })
            .collect();
        Ok(Self {
            pattern,
            inputs,
            outputs,
        })
    }

    /// Compute the map from pattern nodes to circuit nodes in `circ`.
    pub fn get_match_map<C: Circuit>(&self, root: Node, circ: &C) -> Option<HashMap<Node, Node>> {
        let single_matcher = SinglePatternMatcher::from_pattern(self.pattern.clone());
        single_matcher
            .get_match_map(
                root,
                validate_weighted_node(circ),
                validate_unweighted_edge(circ),
            )
            .map(|m| m.into_iter().collect())
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

#[cfg(test)]
mod tests {
    use hugr::Hugr;
    use itertools::Itertools;

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

    #[test]
    fn construct_pattern() {
        let hugr = h_cx();

        let p = CircuitPattern::try_from_circuit(&hugr).unwrap();

        let edges = p
            .pattern
            .edges()
            .unwrap()
            .iter()
            .map(|e| (e.source.unwrap(), e.target.unwrap()))
            .collect_vec();
        assert_eq!(
            // How would I construct hugr::Nodes for testing here?
            edges.len(),
            1
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
}
