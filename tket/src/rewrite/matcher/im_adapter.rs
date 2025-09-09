//! An adaptor for [`CircuitMatcher`]s that matches patterns in
//! [`PersistentHugr`]s.

use std::collections::VecDeque;

use hugr::{
    hugr::views::SiblingSubgraph,
    persistent::{PatchNode, PersistentHugr, PersistentWire, PinnedSubgraph, Walker},
    Direction, HugrView, Port,
};
use itertools::Itertools;

use crate::{
    resource::ResourceScope,
    rewrite::matcher::{
        all_linear_ports, as_tket_or_extension_op, MatchContext, MatchOutcome, TketOrExtensionOp,
    },
    rewrite_space::RewriteSpace,
    Subcircuit,
};

use super::{CircuitMatcher, MatchOutcomeEnum};

/// An adaptor for [`CircuitMatcher`]s that matches patterns in
/// [`RewriteSpace`]s.
pub struct ImMatchAdapter<'m, M: ?Sized> {
    pub(super) matcher: &'m M,
}

impl<'m, M: ?Sized> ImMatchAdapter<'m, M> {
    /// Create a new [`ImMatchAdapter`].
    pub fn new(matcher: &'m M) -> Self {
        Self { matcher }
    }
}

#[derive(Debug, Clone)]
struct MatchState<'a, PartialMatchInfo> {
    /// The (currently incomplete) match.
    subcircuit: Subcircuit<PatchNode>,
    /// The wires that have been matched.
    matched_wires: Vec<PersistentWire>,
    /// The walker to use to traverse the state space.
    walker: Walker<'a>,
    /// Ports to traverse next for matching.
    active_ports: VecDeque<(PatchNode, Port)>,
    /// Context of the current partial match.
    match_info: PartialMatchInfo,
}

impl<'a, PartialMatchInfo> MatchState<'a, PartialMatchInfo> {
    #[must_use]
    fn extend_match(
        &mut self,
        wire: PersistentWire,
        walker: Walker<'a>,
        scope: &ResourceScope<impl HugrView<Node = PatchNode>>,
    ) -> bool {
        let mut subcircuit = self.subcircuit.clone();
        let mut new_pinned_ports = Vec::new();
        for (node, port) in walker.wire_pinned_ports(&wire, None) {
            match subcircuit.try_add_node(node, scope) {
                Err(..) => {
                    return false;
                }
                Ok(true) => {
                    new_pinned_ports.push((node, port));
                }
                Ok(false) => { /* nothing to do, port already in match */ }
            }
        }

        self.subcircuit = subcircuit;
        self.walker = walker;

        if !self.walker.is_complete(&wire, None) {
            if let Some(&pinned_node_port) = new_pinned_ports.first() {
                // Revisit this wire again, to further expand the wire
                self.active_ports.push_back(pinned_node_port);
            }
            return true;
        } else {
            for (node, port) in new_pinned_ports {
                for other_port in all_linear_ports(self.walker.as_hugr_view(), node) {
                    if other_port != port {
                        self.active_ports.push_back((node, other_port));
                    }
                }
            }
        }

        true
    }
}

fn to_resource_scope<H: HugrView>(hugr: H) -> ResourceScope<H> {
    let mut all_nodes = HugrView::children(&hugr, hugr.entrypoint());
    let [input, output] = (&mut all_nodes).take(2).collect_array().unwrap();
    let nodes = all_nodes.collect_vec();
    let incoming_ports = hugr
        .out_value_types(input)
        .map(|(p, _)| hugr.linked_inputs(input, p).collect_vec())
        .collect_vec();
    let outgoing_ports = hugr
        .in_value_types(output)
        .map(|(p, _)| hugr.single_linked_output(output, p).unwrap())
        .collect_vec();
    let subgraph = SiblingSubgraph::new_unchecked(incoming_ports, outgoing_ports, vec![], nodes);
    ResourceScope::new(hugr, subgraph)
}

pub struct ImMatchResult<MatchInfo> {
    pub subcircuit: Subcircuit<PatchNode>,
    pub subgraph: PinnedSubgraph,
    pub hugr: ResourceScope<PersistentHugr>,
    pub match_info: MatchInfo,
}

impl<'m, M: CircuitMatcher + ?Sized> ImMatchAdapter<'m, M> {
    /// Get all matching subcircuits within the [`RewriteSpace`].
    pub fn get_all_matches<C>(&self, space: &RewriteSpace<C>) -> Vec<ImMatchResult<M::MatchInfo>> {
        <RewriteSpace<C> as hugr::hugr::views::NodesIter>::nodes(space)
            .flat_map(|n| self.get_matches(space, n))
            .collect()
    }

    /// Get all matching subcircuits within the [`RewriteSpace`] with one of
    /// the given `root_nodes` as their root.
    pub fn get_matches<C>(
        &self,
        space: &RewriteSpace<C>,
        root_node: PatchNode,
    ) -> Vec<ImMatchResult<M::MatchInfo>>
// where
    //     M::PartialMatchInfo: std::fmt::Debug,
    //     M::MatchInfo: std::fmt::Debug,
    {
        let mut queue = VecDeque::new();
        let mut all_matches = Vec::new();

        // 1. Initialise queue
        {
            let walker = Walker::from_pinned_node(root_node, space.state_space());
            let scope = to_resource_scope(walker.as_hugr_view().clone());
            let Some(outcome) = match_node(
                root_node,
                &scope,
                &Subcircuit::new_empty(),
                M::PartialMatchInfo::default(),
                self.matcher,
            ) else {
                return all_matches;
            };

            if let Some(complete) = outcome.complete {
                all_matches.push(ImMatchResult {
                    subcircuit: Subcircuit::from_node(root_node, &scope),
                    subgraph: PinnedSubgraph::try_from_pinned([root_node], [], &walker).unwrap(),
                    hugr: scope.clone(),
                    match_info: complete,
                });
            }
            if let Some(proceed) = outcome.proceed {
                queue.push_back(MatchState {
                    subcircuit: Subcircuit::from_node(root_node, &scope),
                    matched_wires: vec![],
                    walker,
                    active_ports: all_linear_ports(scope.hugr(), root_node)
                        .map(|p| (root_node, p))
                        .collect(),
                    match_info: proceed.updated(&M::PartialMatchInfo::default()),
                });
            }
        }

        // 2. Find all matches
        while let Some(mut current_match) = queue.pop_front() {
            let Some((node, port)) = current_match.active_ports.pop_front() else {
                continue;
            };

            let wire = current_match.walker.get_wire(node, port);
            if current_match.walker.is_complete(&wire, None) {
                // nothing to do, can proceed without this wire
                queue.push_back(current_match);
                continue;
            }

            for new_walker in current_match.walker.expand(&wire, None) {
                let new_wire = new_walker.get_wire(node, port);
                let Ok((new_node, _)) = new_walker
                    .wire_pinned_ports(&new_wire, None)
                    .filter(|&np| np != (node, port))
                    .exactly_one()
                else {
                    panic!("walker must expand one port");
                };

                // Check if the new node should be matched
                let scope = to_resource_scope(new_walker.as_hugr_view());
                let Some(outcome) = match_node(
                    new_node,
                    &scope,
                    &current_match.subcircuit,
                    current_match.match_info.clone(),
                    self.matcher,
                ) else {
                    queue.push_back(current_match.clone());
                    continue;
                };

                for outcome in outcome.into_enum_vec(&current_match.match_info) {
                    match outcome {
                        MatchOutcomeEnum::Complete(complete) => {
                            if !new_walker.is_complete(&new_wire, None) {
                                eprintln!("wire is not complete, cannot create match");
                            }
                            let mut new_match = current_match.clone();
                            if !new_match.extend_match(new_wire.clone(), new_walker.clone(), &scope)
                            {
                                continue;
                            }
                            let subgraph = match PinnedSubgraph::try_from_wires(
                                new_match.matched_wires,
                                &new_walker,
                            ) {
                                Ok(subgraph) => subgraph,
                                Err(err) => {
                                    eprintln!(
                                        "warning: could not create pinned subgraph found: {err}"
                                    );
                                    continue;
                                }
                            };
                            all_matches.push(ImMatchResult {
                                subcircuit: new_match.subcircuit,
                                subgraph,
                                hugr: scope.cloned(),
                                match_info: complete,
                            });
                        }
                        MatchOutcomeEnum::Proceed(proceed) => {
                            let mut new_match = current_match.clone();
                            if !new_match.extend_match(new_wire.clone(), new_walker.clone(), &scope)
                            {
                                continue;
                            }
                            new_match.match_info = proceed;
                            queue.push_back(new_match);
                        }
                        MatchOutcomeEnum::Skip(skip) => {
                            queue.push_back(MatchState {
                                match_info: skip,
                                ..current_match.clone()
                            });
                        }
                    }
                }
            }
        }

        all_matches
    }
}

fn match_node<M: CircuitMatcher + ?Sized, H: HugrView<Node = PatchNode>>(
    node: PatchNode,
    circ: &ResourceScope<H>,
    subcircuit: &Subcircuit<PatchNode>,
    match_info: M::PartialMatchInfo,
    matcher: &M,
) -> Option<MatchOutcome<M::PartialMatchInfo, M::MatchInfo>> {
    let Some(input_args) = circ.get_circuit_units_slice(node, Direction::Incoming) else {
        // Unsupported op
        return None;
    };
    let Some(output_args) = circ.get_circuit_units_slice(node, Direction::Outgoing) else {
        // Unsupported op
        return None;
    };

    let match_context = MatchContext {
        subcircuit,
        match_info,
        circuit: circ,
        op_node: node,
    };

    match as_tket_or_extension_op(circ.hugr().get_optype(node)) {
        TketOrExtensionOp::Tket(tket_op) => {
            if output_args.len() > input_args.len()
                || !input_args.iter().zip(output_args).all(|(a, b)| a == b)
            {
                None
            } else {
                Some(matcher.match_tket_op(tket_op, &input_args, match_context))
            }
        }
        TketOrExtensionOp::Extension(extension_op) => {
            Some(matcher.match_extension_op(extension_op, &input_args, &output_args, match_context))
        }
        TketOrExtensionOp::Unsupported => None,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use hugr::persistent::PersistentHugr;
    use rstest::{fixture, rstest};

    use crate::{
        rewrite::{matcher::tests::TestHadamardMatcher, CircuitRewrite},
        utils::build_simple_circuit,
        TketOp,
    };

    use super::*;

    #[fixture]
    fn hugrs_in_rewrite_space() -> (RewriteSpace<u32>, [PersistentHugr; 2]) {
        let xx_circ = build_simple_circuit(1, |circ| {
            circ.append(TketOp::X, vec![0]).unwrap();
            circ.append(TketOp::X, vec![0]).unwrap();
            Ok(())
        })
        .unwrap();
        let h_circ = build_simple_circuit(1, |circ| {
            circ.append(TketOp::H, vec![0]).unwrap();
            Ok(())
        })
        .unwrap();

        let base = PersistentHugr::with_base(xx_circ.into_hugr());
        let rewrite_space = RewriteSpace::from_state_space(base.state_space().clone());

        let xx_nodes = base
            .base_hugr()
            .nodes()
            .filter(|&n| base.base_hugr().get_optype(n) == &TketOp::X.into())
            .map(|n| base.base_commit().to_patch_node(n))
            .collect_array::<2>()
            .unwrap();

        let circuit = to_resource_scope(&base);
        let rw1 = CircuitRewrite::try_new(
            Subcircuit::from_node(xx_nodes[0], &circuit),
            &circuit,
            h_circ.clone(),
        )
        .unwrap();
        let rw2 = CircuitRewrite::try_new(
            Subcircuit::from_node(xx_nodes[1], &circuit),
            &circuit,
            h_circ,
        )
        .unwrap();

        let rw1 = rewrite_space.try_add_rewrite(rw1, 1, &circuit).unwrap();
        let rw2 = rewrite_space.try_add_rewrite(rw2, 1, &circuit).unwrap();

        (
            rewrite_space.clone(),
            [
                PersistentHugr::from_commit(rw1.into()),
                PersistentHugr::from_commit(rw2.into()),
            ],
        )
    }

    #[rstest]
    fn test_im_match_adapter(hugrs_in_rewrite_space: (RewriteSpace<u32>, [PersistentHugr; 2])) {
        let (rewrite_space, [h1, h2]) = hugrs_in_rewrite_space;

        let matcher = TestHadamardMatcher.as_rewrite_space_matcher();
        let found_match = matcher
            .get_all_matches(&rewrite_space)
            .into_iter()
            .map(|ImMatchResult { subgraph, .. }| subgraph)
            .unique()
            .exactly_one()
            .ok()
            .unwrap();

        assert_eq!(
            found_match.selected_commits().collect::<HashSet<_>>(),
            h1.all_commit_ids()
                .chain(h2.all_commit_ids())
                .collect::<HashSet<_>>()
        )
    }
}
