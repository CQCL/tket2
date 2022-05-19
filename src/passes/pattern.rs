use std::collections::HashMap;

use crate::{
    circuit::operation::Op,
    graph::graph::{Direction, EdgeIndex, Graph, IndexType, NodeIndex},
};

struct MatchFail();

struct PatternMatcher<'g: 'p, 'p, N, E, Ix: IndexType> {
    target: &'g Graph<N, E, Ix>,
    pattern: &'p Graph<N, E, Ix>,
    pattern_boundary: [NodeIndex<Ix>; 2],
}
type Match<Ix> = HashMap<NodeIndex<Ix>, NodeIndex<Ix>>;

impl<'g, 'p, N: PartialEq, E: PartialEq, Ix: IndexType> PatternMatcher<'g, 'p, N, E, Ix> {
    fn new(
        target: &'g Graph<N, E, Ix>,
        pattern: &'p Graph<N, E, Ix>,
        pattern_boundary: [NodeIndex<Ix>; 2],
    ) -> Self {
        Self {
            target,
            pattern,
            pattern_boundary,
        }
    }

    fn node_match(
        &self,
        pattern_node: NodeIndex<Ix>,
        target_node: NodeIndex<Ix>,
        node_comp: &impl Fn(&N, &N) -> bool,
    ) -> Result<(), MatchFail> {
        match (
            self.target.node_weight(target_node),
            self.pattern.node_weight(pattern_node),
        ) {
            (Some(x), Some(y)) if node_comp(x, y) => Ok(()),
            _ => Err(MatchFail()),
        }
    }

    fn edge_match(
        &self,
        pattern_edge: EdgeIndex<Ix>,
        target_edge: EdgeIndex<Ix>,
    ) -> Result<(), MatchFail> {
        let err = Err(MatchFail());
        if self.target.edge_weight(target_edge) != self.pattern.edge_weight(pattern_edge) {
            return err;
        }
        match (
            self.target.edge_endpoints(target_edge),
            self.pattern.edge_endpoints(pattern_edge),
        ) {
            (None, None) => (),
            (Some([ts, tt]), Some([ps, pt])) => {
                let [i, o] = self.pattern_boundary;
                if (ps.node != i && ps.port != ts.port) || (pt.node != o && pt.port != tt.port) {
                    return err;
                }
            }
            _ => return err,
        }
        Ok(())
    }

    fn cycle_node_edges(g: &Graph<N, E, Ix>, n: NodeIndex<Ix>) -> Vec<EdgeIndex<Ix>> {
        g.node_edges(n, Direction::Incoming)
            .chain(g.node_edges(n, Direction::Outgoing))
            .cloned()
            .collect()
    }

    fn match_from(
        &self,
        pattern_node: NodeIndex<Ix>,
        target_node: NodeIndex<Ix>,
        start_edge: EdgeIndex<Ix>,
        match_map: &mut Match<Ix>,
        node_comp: &impl Fn(&N, &N) -> bool,
    ) -> Result<(), MatchFail> {
        let err = Err(MatchFail());
        self.node_match(pattern_node, target_node, node_comp)?;
        match_map.insert(pattern_node, target_node);

        let p_edges = Self::cycle_node_edges(self.pattern, pattern_node);
        let t_edges = Self::cycle_node_edges(self.target, target_node);

        if p_edges.len() != t_edges.len() {
            return err;
        }
        // circle the edges of both nodes starting at the start edge
        for (e_p, e_t) in p_edges
            .iter()
            .zip(t_edges.iter())
            .cycle()
            .skip_while(|(p, _): &(&EdgeIndex<Ix>, _)| **p != start_edge)
            .take(p_edges.len())
        {
            self.edge_match(*e_p, *e_t)?;

            let [e_p_source, e_p_target] = self.pattern.edge_endpoints(*e_p).ok_or(MatchFail())?;
            if e_p_source.node == self.pattern_boundary[0]
                || e_p_target.node == self.pattern_boundary[1]
            {
                continue;
            }

            let (next_pattern_node, next_target_node) = if e_p_source.node == pattern_node {
                (
                    e_p_target.node,
                    self.target.edge_endpoints(*e_t).ok_or(MatchFail())?[1].node,
                )
            } else {
                (
                    e_p_source.node,
                    self.target.edge_endpoints(*e_t).ok_or(MatchFail())?[0].node,
                )
            };

            if let Some(matched_node) = match_map.get(&next_pattern_node) {
                if *matched_node == next_target_node {
                    continue;
                } else {
                    return err;
                }
            }
            self.match_from(
                next_pattern_node,
                next_target_node,
                *e_p,
                match_map,
                node_comp,
            )?;
        }

        Ok(())
    }

    fn start_pattern_node_edge(&self) -> (NodeIndex<Ix>, EdgeIndex<Ix>) {
        // TODO better first pick
        let e = self
            .pattern
            .node_edges(self.pattern_boundary[0], Direction::Outgoing)
            .next()
            .unwrap();

        (self.pattern.edge_endpoints(*e).unwrap()[1].node, *e)
    }
    fn find_matches(
        &'g self,
        node_comp: impl Fn(&N, &N) -> bool + 'g,
    ) -> impl Iterator<Item = Match<Ix>> + 'g {
        let (start, start_edge) = self.start_pattern_node_edge();

        self.target
            .nodes()
            // .filter(move |t_n| )
            .filter_map(move |candidate| {
                if self.node_match(start, candidate, &node_comp).is_err() {
                    return None;
                }
                let mut bijection = Match::new();
                self.match_from(start, candidate, start_edge, &mut bijection, &node_comp)
                    .ok()
                    .map(|()| bijection)
            })
    }
}

struct MatchIter<Ix, I: Iterator<Item = NodeIndex<Ix>>, F> {
    nodes: I,
    node_comp: F,
    start_node: NodeIndex<Ix>,
    start_edge: EdgeIndex<Ix>,
}

#[cfg(test)]
mod tests {
    use rstest::{fixture, rstest};

    use super::{Match, PatternMatcher};
    use crate::circuit::circuit::{Circuit, UnitID};
    use crate::circuit::operation::{Op, WireType};
    use crate::graph::dot::dot_string;
    use crate::graph::graph::{IndexType, NodeIndex, PortIndex};

    #[fixture]
    fn simple_circ() -> Circuit {
        let mut circ1 = Circuit::new();
        let [i, o] = circ1.boundary();
        for p in 0..2 {
            let noop = circ1.add_vertex(Op::Noop);
            circ1.add_edge((i, p), (noop, 0), WireType::Qubit);
            circ1.add_edge((noop, 0), (o, p), WireType::Qubit);
        }
        circ1
    }
    #[fixture]
    fn simple_isomorphic_circ() -> Circuit {
        let mut circ1 = Circuit::new();
        let [i, o] = circ1.boundary();
        for p in (0..2).rev() {
            let noop = circ1.add_vertex(Op::Noop);
            circ1.add_edge((noop, 0), (o, p), WireType::Qubit);
            circ1.add_edge((i, p), (noop, 0), WireType::Qubit);
        }
        circ1
    }

    #[fixture]
    fn noop_pattern_circ() -> Circuit {
        let mut circ1 = Circuit::new();
        let [i, o] = circ1.boundary();
        let noop = circ1.add_vertex(Op::Noop);
        circ1.add_edge((i, 0), (noop, 0), WireType::Qubit);
        circ1.add_edge((noop, 0), (o, 0), WireType::Qubit);
        circ1
    }

    #[rstest]
    fn test_node_match(simple_circ: Circuit, simple_isomorphic_circ: Circuit) {
        let [i, o] = simple_circ.boundary();
        let pattern_boundary = simple_isomorphic_circ.boundary().clone();
        let dag1 = simple_circ.dag;
        let dag2 = simple_isomorphic_circ.dag;
        let matcher = PatternMatcher::new(&dag1, &dag2, pattern_boundary);
        for (n1, n2) in dag1.nodes().zip(dag2.nodes()) {
            assert!(matcher.node_match(n1, n2, &PartialEq::eq).is_ok());
        }

        assert!(matcher.node_match(i, o, &PartialEq::eq).is_err());
    }

    #[rstest]
    fn test_edge_match(simple_circ: Circuit) {
        let fedges: Vec<_> = simple_circ.dag.edges().collect();
        let pattern_boundary = simple_circ.boundary().clone();

        let dag1 = simple_circ.dag.clone();
        let dag2 = simple_circ.dag;

        let matcher = PatternMatcher::new(&dag1, &dag2, pattern_boundary);
        for (e1, e2) in dag1.edges().zip(dag2.edges()) {
            assert!(matcher.edge_match(e1, e2).is_ok());
        }

        assert!(matcher.edge_match(fedges[0], fedges[3]).is_err());
    }

    fn match_maker<Ix: IndexType>(it: impl IntoIterator<Item = (usize, usize)>) -> Match<Ix> {
        Match::from_iter(
            it.into_iter()
                .map(|(i, j)| (NodeIndex::new(i.into()), NodeIndex::new(j.into()))),
        )
    }

    #[rstest]
    fn test_pattern(mut simple_circ: Circuit, noop_pattern_circ: Circuit) {
        let xop = simple_circ.add_vertex(Op::H);
        let [i, o] = simple_circ.boundary();
        simple_circ.add_edge((i, 3), (xop, 0), WireType::Qubit);
        simple_circ.add_edge((xop, 0), (o, 3), WireType::Qubit);

        let pattern_boundary = noop_pattern_circ.boundary().clone();
        let matcher =
            PatternMatcher::new(&simple_circ.dag, &noop_pattern_circ.dag, pattern_boundary);

        let matches: Vec<_> = matcher.find_matches(PartialEq::eq).collect();

        // match noop to two noops in target
        assert_eq!(matches[0], match_maker([(2, 2)]));
        assert_eq!(matches[1], match_maker([(2, 3)]));
    }

    #[test]
    fn test_cx_ladder() {
        let qubits = vec![
            UnitID::Qubit {
                name: "q".into(),
                index: vec![0],
            },
            UnitID::Qubit {
                name: "q".into(),
                index: vec![1],
            },
        ];
        let mut target_circ = Circuit::with_uids(qubits.clone());
        target_circ
            .append_op(Op::H, &vec![PortIndex::new(0)])
            .unwrap();
        target_circ
            .append_op(Op::H, &vec![PortIndex::new(1)])
            .unwrap();
        target_circ
            .append_op(Op::CX, &vec![PortIndex::new(0), PortIndex::new(1)])
            .unwrap();
        target_circ
            .append_op(Op::H, &vec![PortIndex::new(0)])
            .unwrap();
        target_circ
            .append_op(Op::H, &vec![PortIndex::new(1)])
            .unwrap();
        target_circ
            .append_op(Op::CX, &vec![PortIndex::new(0), PortIndex::new(1)])
            .unwrap();
        target_circ
            .append_op(Op::H, &vec![PortIndex::new(0)])
            .unwrap();
        target_circ
            .append_op(Op::H, &vec![PortIndex::new(1)])
            .unwrap();
        target_circ
            .append_op(Op::CX, &vec![PortIndex::new(1), PortIndex::new(0)])
            .unwrap();
        target_circ
            .append_op(Op::H, &vec![PortIndex::new(0)])
            .unwrap();
        target_circ
            .append_op(Op::H, &vec![PortIndex::new(1)])
            .unwrap();

        let mut pattern_circ = Circuit::with_uids(qubits);
        pattern_circ
            .append_op(Op::H, &vec![PortIndex::new(0)])
            .unwrap();
        pattern_circ
            .append_op(Op::H, &vec![PortIndex::new(1)])
            .unwrap();
        pattern_circ
            .append_op(Op::CX, &vec![PortIndex::new(0), PortIndex::new(1)])
            .unwrap();
        pattern_circ
            .append_op(Op::H, &vec![PortIndex::new(0)])
            .unwrap();
        pattern_circ
            .append_op(Op::H, &vec![PortIndex::new(1)])
            .unwrap();

        let pattern_boundary = pattern_circ.boundary().clone();

        let matcher = PatternMatcher::new(&target_circ.dag, &pattern_circ.dag, pattern_boundary);

        let matches: Vec<_> = matcher
            .find_matches(|op1, op2| match (&op1.op, &op2.op) {
                (Op::H, Op::H) => true,
                (Op::CX, Op::CX) => true,
                _ => false,
            })
            .collect();

        assert_eq!(matches.len(), 3);
        assert_eq!(
            matches[0],
            match_maker([(2, 2), (3, 3), (4, 4), (5, 5), (6, 6)])
        );
        assert_eq!(
            matches[1],
            match_maker([(2, 5), (3, 6), (4, 7), (5, 8), (6, 9)])
        );
        // check flipped match happens
        assert_eq!(
            matches[2],
            match_maker([(2, 9), (3, 8), (4, 10), (5, 12), (6, 11)])
        );
    }
}
