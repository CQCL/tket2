//! An adaptor for [`CircuitMatcher`]s that matches patterns in
//! [`PersistentHugr`]s.

use std::cell::RefCell;
use std::collections::{BTreeSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::rc::Rc;

use hugr::{
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

use super::{CircuitMatcher, MatchOutcomeEnum, MatchingOptions};

mod resource_scope_cache;
pub use resource_scope_cache::ResourceScopeCache;

/// A walker that carries around a cache of recent [`ResourceScope`]s.
#[derive(Clone)]
pub struct CachedWalker<'w> {
    pub(crate) walker: Walker<'w>,
    pub(crate) cache: Rc<RefCell<ResourceScopeCache>>,
}

impl CachedWalker<'_> {
    /// Get the underlying hugr view of the walker.
    pub fn as_hugr_view(&self) -> &PersistentHugr {
        self.walker.as_hugr_view()
    }

    /// Create a new `CachedWalker` with the same cache but a different walker.
    pub fn with_walker<'a>(&self, new_walker: Walker<'a>) -> CachedWalker<'a> {
        CachedWalker {
            walker: new_walker,
            cache: self.cache.clone(),
        }
    }
}

impl hugr::hugr::views::NodesIter for CachedWalker<'_> {
    type Node = PatchNode;

    fn nodes(&self) -> impl Iterator<Item = PatchNode> {
        hugr::hugr::views::NodesIter::nodes(&self.walker)
    }
}

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
        self.matched_wires.push(wire.clone());

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

    /// Compute a hash of the state, invariant under reordering of the lines in
    /// the subcircuit, matched wires, the active ports and the set of selected
    /// commits in the walker.
    fn fx_hash(
        &self,
        partial_match_hash: impl FnOnce(&PartialMatchInfo, &mut fxhash::FxHasher),
    ) -> u64 {
        let mut hasher = fxhash::FxHasher::default();
        let mut intervals = self.subcircuit.intervals_iter().collect_vec();
        let mut active_ports: Vec<_> = self.active_ports.clone().into();
        let mut matched_wires = self.matched_wires.clone();
        let mut selected_ids = self.walker.as_hugr_view().all_commit_ids().collect_vec();

        intervals.sort_unstable_by_key(|l| l.resource_id());
        active_ports.sort_unstable_by_key(|&(n, p)| (n, p));
        matched_wires.sort_unstable();
        selected_ids.sort_unstable();

        intervals.hash(&mut hasher);
        active_ports.hash(&mut hasher);
        matched_wires.hash(&mut hasher);
        selected_ids.hash(&mut hasher);
        partial_match_hash(&self.match_info, &mut hasher);

        hasher.finish()
    }
}

pub struct ImMatchResult<MatchInfo> {
    pub subcircuit: Subcircuit<PatchNode>,
    pub subgraph: PinnedSubgraph,
    pub hugr: ResourceScope<PersistentHugr>,
    pub match_info: MatchInfo,
}

impl<'m, M: CircuitMatcher + ?Sized> ImMatchAdapter<'m, M> {
    /// Get all matching subcircuits within the [`RewriteSpace`].
    pub fn get_all_matches<C>(
        &self,
        space: &RewriteSpace<C>,
        options: &MatchingOptions<M::PartialMatchInfo, M::MatchInfo>,
    ) -> Vec<ImMatchResult<M::MatchInfo>> {
        if options.only_maximal_matches {
            eprintln!("ignoring unusupported option `only_maximal_matches`");
        }

        // hashes of complete matches, only used if deduplication is enabled
        let mut dedup_complete_matches = options
            .deduplicate_complete_matches
            .as_ref()
            .map(|_| BTreeSet::<u64>::new());

        // hashes of visited states, only used if deduplication is enabled
        let mut dedup_partial_matches = options
            .deduplicate_partial_matches
            .as_ref()
            .map(|_| BTreeSet::<u64>::new());

        let mut cache = ResourceScopeCache::new();
        let all_matches: Vec<ImMatchResult<M::MatchInfo>> =
            <RewriteSpace<C> as hugr::hugr::views::NodesIter>::nodes(space)
                .flat_map(|n| {
                    let walker = Walker::from_pinned_node(n, space.state_space());
                    self.get_matches_with_dedup_maps(
                        walker,
                        n,
                        options,
                        &mut cache,
                        &mut dedup_complete_matches,
                        &mut dedup_partial_matches,
                    )
                })
                .collect();

        // Note: only_maximal_matches not implemented for ImMatchResult yet
        // if options.only_maximal_matches {
        //     remove_non_maximal_matches(&mut all_matches);
        // }

        all_matches
    }

    /// Get all matching subcircuits within the [`RewriteSpace`] with one of
    /// the given `root_nodes` as their root.
    pub fn get_matches(
        &self,
        cached_walker: &CachedWalker,
        root_node: PatchNode,
        options: &MatchingOptions<M::PartialMatchInfo, M::MatchInfo>,
    ) -> Vec<ImMatchResult<M::MatchInfo>> {
        let walker = cached_walker.walker.clone();

        if options.only_maximal_matches {
            eprintln!("ignoring unusupported option `only_maximal_matches`");
        }

        // hashes of complete matches, only used if deduplication is enabled
        let mut dedup_complete_matches = options
            .deduplicate_complete_matches
            .as_ref()
            .map(|_| BTreeSet::<u64>::new());

        // hashes of visited states, only used if deduplication is enabled
        let mut dedup_partial_matches = options
            .deduplicate_partial_matches
            .as_ref()
            .map(|_| BTreeSet::<u64>::new());

        let all_matches = self.get_matches_with_dedup_maps(
            walker,
            root_node,
            options,
            &mut *cached_walker.cache.borrow_mut(),
            &mut dedup_complete_matches,
            &mut dedup_partial_matches,
        );

        all_matches
    }

    fn get_matches_with_dedup_maps(
        &self,
        walker: Walker,
        root_node: PatchNode,
        options: &MatchingOptions<M::PartialMatchInfo, M::MatchInfo>,
        cache: &mut ResourceScopeCache,
        dedup_complete_matches: &mut Option<BTreeSet<u64>>,
        dedup_partial_matches: &mut Option<BTreeSet<u64>>,
    ) -> Vec<ImMatchResult<M::MatchInfo>> {
        let mut queue = VecDeque::new();
        let mut all_matches = Vec::new();

        let scope = cache.init(&walker);

        // 1. Initialise queue
        {
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
                let match_result = ImMatchResult {
                    subcircuit: Subcircuit::from_node(root_node, &scope),
                    subgraph: PinnedSubgraph::try_from_pinned([root_node], [], &walker).unwrap(),
                    hugr: scope.clone(),
                    match_info: complete,
                };

                // Check for deduplication of complete matches
                if let Some(dedup_fn) = options.deduplicate_complete_matches.as_ref() {
                    let hash = fx_hash_match_result(&match_result, dedup_fn);
                    let complete_matches = dedup_complete_matches.as_mut().expect("enabled");
                    if complete_matches.insert(hash) {
                        all_matches.push(match_result);
                    }
                } else {
                    all_matches.push(match_result);
                }
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
            if let Some(partial_match_hash) = options.deduplicate_partial_matches.as_ref() {
                let hash = current_match.fx_hash(partial_match_hash);
                let visited_states = dedup_partial_matches.as_mut().expect("enabled");
                if !visited_states.insert(hash) {
                    continue;
                }
            }

            let Some((node, port)) = current_match.active_ports.pop_front() else {
                continue;
            };

            let wire = current_match.walker.get_wire(node, port);

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
                let Ok(scope) = cache.update(&new_walker, &current_match.walker) else {
                    eprintln!("Could not construct PersistentHugr from walker, skipping potentially valid commit combinations");
                    continue;
                };
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

                            let match_result = ImMatchResult {
                                subcircuit: new_match.subcircuit,
                                subgraph,
                                hugr: scope.clone(),
                                match_info: complete,
                            };

                            // Check for deduplication of complete matches
                            if let Some(dedup_fn) = options.deduplicate_complete_matches.as_ref() {
                                let hash = fx_hash_match_result(&match_result, dedup_fn);
                                let complete_matches =
                                    dedup_complete_matches.as_mut().expect("enabled");
                                if complete_matches.insert(hash) {
                                    all_matches.push(match_result);
                                }
                            } else {
                                all_matches.push(match_result);
                            }
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

fn fx_hash_match_result<I>(
    match_result: &ImMatchResult<I>,
    deduplicate_complete_matches: impl FnOnce(&I, &mut fxhash::FxHasher),
) -> u64 {
    let mut intervals = match_result.subcircuit.intervals_iter().collect_vec();
    intervals.sort_unstable_by_key(|l| l.resource_id());

    let mut hasher = fxhash::FxHasher::default();

    intervals.hash(&mut hasher);
    deduplicate_complete_matches(&match_result.match_info, &mut hasher);

    hasher.finish()
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
        rewrite::{
            matcher::{
                im_adapter::resource_scope_cache::to_resource_scope, tests::TestHadamardMatcher,
            },
            CircuitRewrite,
        },
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

        let rw1 = rewrite_space
            .try_add_rewrite(rw1, 1.into(), &circuit.as_ref())
            .unwrap();
        let rw2 = rewrite_space
            .try_add_rewrite(rw2, 1.into(), &circuit.as_ref())
            .unwrap();

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
            .get_all_matches(&rewrite_space, &MatchingOptions::default())
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
