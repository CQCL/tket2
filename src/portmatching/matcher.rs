//! Pattern and matcher objects for circuit matching

use std::fmt::Debug;

use super::OpType;
use hugr::{ops::OpTrait, Node, Port};
use itertools::Itertools;
use portmatching::{
    automaton::LineBuilder, matcher::PatternMatch, HashMap, ManyMatcher, Pattern, PatternID,
    PortMatcher,
};

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

use crate::circuit::Circuit;

type PEdge = (Port, Port);
type PNode = OpType;

/// A pattern that match a circuit exactly
#[cfg_attr(feature = "pyo3", pyclass)]
#[derive(Clone)]
pub struct CircuitPattern(Pattern<Node, PNode, PEdge>);

impl CircuitPattern {
    /// Construct a pattern from a circuit
    pub fn from_circuit<'circ, C: Circuit<'circ>>(circuit: &'circ C) -> Self {
        let mut p = Pattern::new();
        for cmd in circuit.commands() {
            p.require(cmd.node, cmd.op.clone().into());
            for out_offset in 0..cmd.outputs.len() {
                let out_offset = Port::new_outgoing(out_offset);
                for (next_node, in_offset) in circuit.linked_ports(cmd.node, out_offset) {
                    if circuit.get_optype(next_node).tag() != hugr::ops::OpTag::Output {
                        p.add_edge(cmd.node, next_node, (out_offset, in_offset));
                    }
                }
            }
        }
        p.set_any_root().unwrap();
        Self(p)
    }
}

impl Debug for CircuitPattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)?;
        Ok(())
    }
}

fn compatible_offsets((_, pout): &(Port, Port), (pin, _): &(Port, Port)) -> bool {
    pout.direction() != pin.direction() && pout.index() == pin.index()
}

/// A matcher object for fast pattern matching on circuits.
///
/// This uses a state automaton internally to match against a set of patterns
/// simultaneously.
#[cfg_attr(feature = "pyo3", pyclass)]
pub struct CircuitMatcher(ManyMatcher<Node, PNode, PEdge, Port>);

impl Debug for CircuitMatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)?;
        Ok(())
    }
}

impl CircuitMatcher {
    /// Construct a matcher from a set of patterns
    pub fn from_patterns(patterns: impl IntoIterator<Item = CircuitPattern>) -> Self {
        let patterns = patterns.into_iter().map(|p| p.0).collect_vec();
        let line_patterns = patterns
            .clone()
            .into_iter()
            .map(|p| {
                p.try_into_line_pattern(compatible_offsets)
                    .expect("Failed to express pattern as line pattern")
            })
            .collect_vec();
        let builder = LineBuilder::from_patterns(line_patterns);
        let automaton = builder.build();
        let matcher = ManyMatcher::new(automaton, patterns);
        Self(matcher)
    }

    /// Compute the map from pattern nodes to circuit nodes for a given match.
    pub fn get_match_map<'circ, C: Circuit<'circ>>(
        &self,
        m: PatternMatch<PatternID, Node>,
        circ: &C,
    ) -> Option<HashMap<Node, Node>> {
        self.0.get_match_map(
            m,
            validate_weighted_node(circ),
            validate_unweighted_edge(circ),
        )
    }
}

impl<'a: 'circ, 'circ, C: Circuit<'circ>> PortMatcher<&'a C, Node, Node> for CircuitMatcher {
    type PNode = PNode;
    type PEdge = PEdge;

    fn find_rooted_matches(&self, circ: &'a C, root: Node) -> Vec<PatternMatch<PatternID, Node>> {
        self.0.run(
            root,
            // Node weights (none)
            validate_weighted_node(circ),
            // Check edge exist
            validate_unweighted_edge(circ),
        )
    }

    fn get_pattern(&self, id: PatternID) -> Option<&Pattern<Node, Self::PNode, Self::PEdge>> {
        self.0.get_pattern(id)
    }

    fn find_matches(&self, circuit: &'a C) -> Vec<PatternMatch<PatternID, Node>> {
        let mut matches = Vec::new();
        for cmd in circuit.commands() {
            matches.append(&mut self.find_rooted_matches(circuit, cmd.node));
        }
        matches
    }
}

/// Check if an edge `e` is valid in a portgraph `g` without weights.
fn validate_unweighted_edge<'circ>(
    circ: &impl Circuit<'circ>,
) -> impl for<'a> Fn(Node, &'a PEdge) -> Option<Node> + '_ {
    move |src, &(src_port, tgt_port)| {
        let (next_node, _) = circ
            .linked_ports(src, src_port)
            .find(|&(_, tgt)| tgt == tgt_port)?;
        Some(next_node)
    }
}

/// Check if an edge `e` is valid in a weighted portgraph `g`.
pub(crate) fn validate_weighted_node<'circ>(
    circ: &impl Circuit<'circ>,
) -> impl for<'a> Fn(Node, &PNode) -> bool + '_ {
    move |v, prop| &OpType::from(circ.get_optype(v)) == prop
}

#[cfg(test)]
mod tests {
    use hugr::{
        builder::{DFGBuilder, Dataflow, DataflowHugr},
        hugr::region::{Region, RegionView},
        ops::LeafOp,
        types::SimpleType,
        Hugr, HugrView,
    };
    use itertools::Itertools;
    use portmatching::PortMatcher;

    use super::{CircuitMatcher, CircuitPattern};

    fn h_cx() -> Hugr {
        let qb = SimpleType::Qubit;
        let mut hugr = DFGBuilder::new(
            vec![qb.clone(); 2],
            vec![qb; 2],
        )
        .unwrap();
        let mut circ = hugr.as_circuit(hugr.input_wires().collect());
        circ.append(LeafOp::CX, [0, 1]).unwrap();
        circ.append(LeafOp::H, [0]).unwrap();
        let out_wires = circ.finish();
        hugr.finish_hugr_with_outputs(out_wires).unwrap()
    }

    #[test]
    fn construct_pattern() {
        let hugr = h_cx();
        let circ = RegionView::new(&hugr, hugr.root());

        let mut p = CircuitPattern::from_circuit(&circ);

        p.0.set_any_root().unwrap();
        let edges =
            p.0.edges()
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
    fn construct_matcher() {
        let hugr = h_cx();
        let circ = RegionView::new(&hugr, hugr.root());

        let p = CircuitPattern::from_circuit(&circ);
        let m = CircuitMatcher::from_patterns(vec![p]);

        let matches = m.find_matches(&circ);
        assert_eq!(matches.len(), 1);
    }
}
