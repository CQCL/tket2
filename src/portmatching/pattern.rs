//! Pattern and matcher objects for circuit matching

use super::OpType;
use hugr::{Node, Port};
use itertools::Itertools;
use portmatching::{
    automaton::{LineBuilder, ScopeAutomaton},
    matcher::PatternMatch,
    Pattern, PatternID, PortMatcher,
};

use crate::circuit::Circuit;

type PEdge = (Port, Port);
type PNode = OpType;

/// A pattern that match a circuit exactly
#[derive(Clone, Debug)]
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
                    p.add_edge(cmd.node, next_node, (out_offset, in_offset));
                }
            }
        }
        Self(p)
    }
}

fn compatible_offsets((_, pout): &(Port, Port), (pin, _): &(Port, Port)) -> bool {
    pout.direction() != pin.direction() && pout.index() == pin.index()
}

/// A matcher object for fast pattern matching on circuits.
///
/// This uses a state automaton internally to match against a set of patterns
/// simultaneously.
pub struct CircuitMatcher {
    automaton: ScopeAutomaton<PNode, PEdge, Port>,
    patterns: Vec<Pattern<Node, PNode, PEdge>>,
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
        Self {
            automaton,
            patterns,
        }
    }
}

impl<'a: 'circ, 'circ, C: Circuit<'circ>> PortMatcher<&'a C, Node, Node> for CircuitMatcher {
    type PNode = PNode;
    type PEdge = PEdge;

    fn find_rooted_matches(&self, circ: &'a C, root: Node) -> Vec<PatternMatch<PatternID, Node>> {
        // let (graph, weights) = graph.into();
        self.automaton
            .run(
                root,
                // Node weights (none)
                |v, prop| &OpType::from(circ.get_optype(v)) == prop,
                // Check edge exist
                |n, &(pout, pin)| {
                    let (next_node, _) = circ
                        .linked_ports(n, pout)
                        .find(|&(_, in_port)| in_port == pin)?;
                    Some(next_node)
                },
            )
            .map(|id| PatternMatch::new(id, root))
            .collect()
    }

    fn get_pattern(&self, id: PatternID) -> Option<&Pattern<Node, Self::PNode, Self::PEdge>> {
        self.patterns.get(id.0)
    }

    fn find_matches(&self, circuit: &'a C) -> Vec<PatternMatch<PatternID, Node>> {
        let mut matches = Vec::new();
        for cmd in circuit.commands() {
            matches.append(&mut self.find_rooted_matches(circuit, cmd.node));
        }
        matches
    }
}
