//! Circuit Patterns for pattern matching

use hugr::{ops::OpTrait, Node, Port};
use itertools::Itertools;
use portmatching::{HashMap, Pattern, SinglePatternMatcher};
use std::fmt::Debug;

use super::{
    matcher::{validate_unweighted_edge, validate_weighted_node},
    PEdge, PNode,
};
use crate::circuit::Circuit;

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

/// A pattern that match a circuit exactly
#[cfg_attr(feature = "pyo3", pyclass)]
#[derive(Clone)]
pub struct CircuitPattern {
    pub(super) pattern: Pattern<Node, PNode, PEdge>,
    /// The input ports
    pub(super) inputs: Vec<Vec<(Node, Port)>>,
    /// The output ports
    pub(super) outputs: Vec<(Node, Port)>,
}

impl CircuitPattern {
    /// Construct a pattern from a circuit.
    pub fn from_circuit<'circ, C: Circuit<'circ>>(circuit: &'circ C) -> Self {
        let mut pattern = Pattern::new();
        for cmd in circuit.commands() {
            pattern.require(cmd.node, cmd.op.clone().try_into().unwrap());
            for out_offset in 0..cmd.outputs.len() {
                let out_offset = Port::new_outgoing(out_offset);
                for (next_node, in_offset) in circuit.linked_ports(cmd.node, out_offset) {
                    if circuit.get_optype(next_node).tag() != hugr::ops::OpTag::Output {
                        pattern.add_edge(cmd.node, next_node, (out_offset, in_offset));
                    }
                }
            }
        }
        pattern.set_any_root().unwrap();
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
        Self {
            pattern,
            inputs,
            outputs,
        }
    }

    /// Compute the map from pattern nodes to circuit nodes in `circ`.
    pub fn get_match_map<'a, C: Circuit<'a>>(
        &self,
        root: Node,
        circ: &C,
    ) -> Option<HashMap<Node, Node>> {
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

#[cfg(test)]
mod tests {
    use hugr::extension::prelude::QB_T;
    use hugr::hugr::views::{DescendantsGraph, HierarchyView};
    use hugr::ops::handle::DfgID;
    use hugr::types::FunctionType;
    use hugr::{
        builder::{DFGBuilder, Dataflow, DataflowHugr},
        Hugr, HugrView,
    };
    use itertools::Itertools;

    use crate::T2Op;

    use super::CircuitPattern;

    fn h_cx() -> Hugr {
        let qb = QB_T;
        let mut hugr = DFGBuilder::new(FunctionType::new_linear(vec![qb; 2])).unwrap();
        let mut circ = hugr.as_circuit(hugr.input_wires().collect());
        circ.append(T2Op::CX, [0, 1]).unwrap();
        circ.append(T2Op::H, [0]).unwrap();
        let out_wires = circ.finish();
        hugr.finish_hugr_with_outputs(out_wires).unwrap()
    }

    #[test]
    fn construct_pattern() {
        let hugr = h_cx();
        let circ: DescendantsGraph<'_, DfgID> = DescendantsGraph::new(&hugr, hugr.root());

        let p = CircuitPattern::from_circuit(&circ);

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
}
