//! An adaptor for [`CircuitMatcher`]s that matches patterns in Hugrs.
use std::{
    cmp::Reverse,
    collections::{BTreeSet, VecDeque},
    hash::{Hash, Hasher},
};

use hugr::{Direction, HugrView, Node, NodeIndex, Port};
use itertools::Itertools;
use portgraph::UnmanagedDenseMap;

use super::{
    as_tket_or_extension_op, CircuitMatcher, MatchContext, MatchOutcomeEnum, TketOrExtensionOp,
};
use crate::{
    resource::ResourceScope,
    rewrite::matcher::{all_linear_ports, MatchingOptions},
    Subcircuit,
};

/// An adaptor for [`CircuitMatcher`]s that matches patterns in concrete
/// circuits.
pub struct HugrMatchAdapter<'m, M: ?Sized> {
    pub(super) matcher: &'m M,
}

impl<'m, M: ?Sized> HugrMatchAdapter<'m, M> {
    /// Create a new [`HugrMatchAdapter`].
    pub fn new(matcher: &'m M) -> Self {
        Self { matcher }
    }
}

#[derive(Debug, Clone, Default)]
struct MatchState<PartialMatchInfo> {
    /// The (currently incomplete) match.
    subcircuit: Subcircuit,
    /// Ports to traverse next for matching.
    active_ports: VecDeque<(Node, Port)>,
    /// Context of the current partial match.
    match_info: PartialMatchInfo,
}

impl<PartialMatchInfo> MatchState<PartialMatchInfo> {
    /// Compute a hash of the state, invariant under reordering of the lines
    /// in the subcircuit and the active ports.
    fn fx_hash(
        &self,
        partial_match_hash: impl FnOnce(&PartialMatchInfo, &mut fxhash::FxHasher),
    ) -> u64 {
        let mut hasher = fxhash::FxHasher::default();
        let mut intervals = self.subcircuit.intervals_iter().collect_vec();
        let mut active_ports: Vec<_> = self.active_ports.clone().into();

        intervals.sort_unstable_by_key(|l| l.resource_id());
        active_ports.sort_unstable_by_key(|&(n, p)| (n, p));

        intervals.hash(&mut hasher);
        active_ports.hash(&mut hasher);
        partial_match_hash(&self.match_info, &mut hasher);

        hasher.finish()
    }
}

impl<'m, M: CircuitMatcher + ?Sized> HugrMatchAdapter<'m, M> {
    /// Get all matching subcircuits within the Circuit.
    pub fn get_all_matches<H: HugrView<Node = Node>>(
        &self,
        circ: &ResourceScope<H>,
        options: &MatchingOptions<M::PartialMatchInfo, M::MatchInfo>,
    ) -> Vec<(Subcircuit, M::MatchInfo)> {
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

        let mut all_matches = circ
            .nodes()
            .iter()
            .flat_map(|&n| {
                self.get_matches_with_dedup_maps(
                    circ,
                    n,
                    options,
                    &mut dedup_complete_matches,
                    &mut dedup_partial_matches,
                )
            })
            .collect();

        if options.only_maximal_matches {
            remove_non_maximal_matches(&mut all_matches, circ);
        }

        all_matches
    }

    /// Get all matching subcircuits within the Circuit rooted at a given node.
    pub fn get_matches<H: HugrView<Node = Node>>(
        &self,
        circ: &ResourceScope<H>,
        root_node: Node,
        options: &MatchingOptions<M::PartialMatchInfo, M::MatchInfo>,
    ) -> Vec<(Subcircuit, M::MatchInfo)> {
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

        let mut all_matches = self.get_matches_with_dedup_maps(
            circ,
            root_node,
            options,
            &mut dedup_complete_matches,
            &mut dedup_partial_matches,
        );

        if options.only_maximal_matches {
            remove_non_maximal_matches(&mut all_matches, circ);
        }

        all_matches
    }

    fn get_matches_with_dedup_maps<H: HugrView<Node = Node>>(
        &self,
        circ: &ResourceScope<H>,
        root_node: Node,
        options: &MatchingOptions<M::PartialMatchInfo, M::MatchInfo>,
        dedup_complete_matches: &mut Option<BTreeSet<u64>>,
        dedup_partial_matches: &mut Option<BTreeSet<u64>>,
    ) -> Vec<(Subcircuit, M::MatchInfo)> {
        let mut queue = VecDeque::new();
        let mut all_matches = Vec::new();

        // 1. Initialise queue
        {
            let mut matches = enqueue_extended_matches(
                &circ,
                MatchState::default(),
                root_node,
                self.matcher,
                &mut queue,
                None,
            );

            if let Some(deduplicate_complete_matches) =
                options.deduplicate_complete_matches.as_ref()
            {
                matches.retain(|(subcirc, info)| {
                    let hash = fx_hash_subcirc_info(subcirc, info, deduplicate_complete_matches);
                    let complete_matches = dedup_complete_matches.as_mut().expect("enabled");
                    complete_matches.insert(hash)
                });
            }

            all_matches.extend(
                matches
                    .into_iter()
                    .filter(|(subcirc, _)| subcirc.validate_subgraph(&circ).is_ok()),
            );
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

            let Ok((new_node, new_port)) = circ.hugr().linked_ports(node, port).exactly_one()
            else {
                continue;
            };

            let mut new_matches = enqueue_extended_matches(
                &circ,
                current_match,
                new_node,
                self.matcher,
                &mut queue,
                Some(new_port),
            );

            if let Some(deduplicate_complete_matches) =
                options.deduplicate_complete_matches.as_ref()
            {
                new_matches.retain(|(subcirc, info)| {
                    let hash = fx_hash_subcirc_info(subcirc, info, deduplicate_complete_matches);
                    let complete_matches = dedup_complete_matches.as_mut().expect("enabled");
                    complete_matches.insert(hash)
                });
            }

            all_matches.extend(
                new_matches
                    .into_iter()
                    .filter(|(subcirc, _)| subcirc.validate_subgraph(&circ).is_ok()),
            );
        }

        all_matches
    }
}

fn fx_hash_subcirc_info<I>(
    subcirc: &Subcircuit,
    info: &I,
    deduplicate_complete_matches: impl FnOnce(&I, &mut fxhash::FxHasher),
) -> u64 {
    let mut intervals = subcirc.intervals_iter().collect_vec();
    intervals.sort_unstable_by_key(|l| l.resource_id());

    let mut hasher = fxhash::FxHasher::default();

    intervals.hash(&mut hasher);
    deduplicate_complete_matches(info, &mut hasher);

    hasher.finish()
}

fn enqueue_extended_matches<M, H>(
    circ: &ResourceScope<H>,
    current_match: MatchState<M::PartialMatchInfo>,
    new_node: Node,
    matcher: &M,
    queue: &mut VecDeque<MatchState<M::PartialMatchInfo>>,
    // The port `new_node` was reached from, or none if this is the first node
    traversed_port: Option<Port>,
) -> Vec<(Subcircuit, M::MatchInfo)>
where
    M: CircuitMatcher + ?Sized,
    H: HugrView<Node = Node>,
{
    let mut new_subcircuit = current_match.subcircuit.clone();

    let mut all_matches = Vec::new();

    let Some(input_args) = circ.get_circuit_units_slice(new_node, Direction::Incoming) else {
        // Unsupported op
        queue.push_back(current_match);
        return all_matches;
    };
    let Some(output_args) = circ.get_circuit_units_slice(new_node, Direction::Outgoing) else {
        // Unsupported op
        queue.push_back(current_match);
        return all_matches;
    };

    if new_subcircuit.try_add_node(new_node, circ).is_err()
        || new_subcircuit == current_match.subcircuit
    {
        // non-convex match or already matched
        queue.push_back(current_match);
        return all_matches;
    }

    let match_context = MatchContext {
        subcircuit: &current_match.subcircuit,
        match_info: current_match.match_info.clone(),
        circuit: circ,
        op_node: new_node,
    };

    let match_outcome = match as_tket_or_extension_op(circ.hugr().get_optype(new_node)) {
        TketOrExtensionOp::Tket(tket_op) => {
            if output_args.len() > input_args.len()
                || !input_args.iter().zip(output_args).all(|(a, b)| a == b)
            {
                // Currently only support TKET ops where the outputs are a prefix
                // of the inputs (all pure quantum ops satisfy this)
                queue.push_back(current_match);
                return all_matches;
            }
            matcher.match_tket_op(tket_op, &input_args, match_context)
        }
        TketOrExtensionOp::Extension(extension_op) => {
            matcher.match_extension_op(extension_op, &input_args, &output_args, match_context)
        }
        TketOrExtensionOp::Unsupported => {
            queue.push_back(current_match);
            return all_matches;
        }
    };

    for match_ in match_outcome.into_enum_vec(&current_match.match_info) {
        match match_ {
            MatchOutcomeEnum::Complete(match_info) => {
                all_matches.push((new_subcircuit.clone(), match_info));
            }
            MatchOutcomeEnum::Proceed(match_info) => {
                let mut active_ports = current_match.active_ports.clone();
                active_ports.extend(
                    all_linear_ports(circ.hugr(), new_node)
                        .filter(|p| Some(p) != traversed_port.as_ref())
                        .map(|p| (new_node, p)),
                );
                queue.push_back(MatchState {
                    subcircuit: new_subcircuit.clone(),
                    active_ports,
                    match_info,
                });
            }
            MatchOutcomeEnum::Skip(match_info) => {
                // do not enqueue "skip"s if this is the first matched node (root)
                // (it's equivalent to choosing another root)
                if traversed_port.is_some() {
                    queue.push_back(MatchState {
                        match_info,
                        subcircuit: current_match.subcircuit.clone(),
                        active_ports: current_match.active_ports.clone(),
                    });
                }
            }
        }
    }

    all_matches
}

/// Remove non-maximal subgraphs (i.e. subgraphs fully contained within another
/// subgraph) from the list of matches.
///
/// This may be an expensive operation: it may take time up to quadratic in
/// the total number of matches (and linear in the size of the largest match).
fn remove_non_maximal_matches<MatchInfo, H: HugrView<Node = Node>>(
    all_matches: &mut Vec<(Subcircuit, MatchInfo)>,
    circ: &ResourceScope<H>,
) {
    // Sort matches from largest to smallest. Thus if A \subseteq B then B is
    // processed before A.
    all_matches.sort_unstable_by_key(|(subg, _)| Reverse(subg.node_count(circ)));

    // A map from node indices to the set of matches that contain that node.
    let mut node_to_matches = UnmanagedDenseMap::<usize, BTreeSet<usize>>::new();
    // A counter to assign unique indices to each match.
    let mut match_ind = 0;

    all_matches.retain(|(subg, _)| {
        // Find a match that covers all nodes of subg
        let match_all_nodes = subg
            .nodes(circ)
            .fold(None, |acc: Option<BTreeSet<usize>>, n| {
                if let Some(acc) = acc {
                    let matches = node_to_matches.get(n.index());
                    Some(acc.intersection(matches).copied().collect())
                } else {
                    Some(node_to_matches.get(n.index()).clone())
                }
            })
            .unwrap_or_default();
        if match_all_nodes.is_empty() {
            // A maximal match. Keep this match and store it in the map
            for n in subg.nodes(circ) {
                node_to_matches.get_mut(n.index()).insert(match_ind);
            }
            match_ind += 1;
            true
        } else {
            false
        }
    });
}

#[cfg(test)]
mod tests {
    use hugr::{extension::prelude::qb_t, types::Signature};
    use rstest::{fixture, rstest};

    use crate::{
        extension::rotation::rotation_type,
        rewrite::matcher::{
            tests::{TestHadamardMatcher, TestRzMatcher},
            CircuitMatcher,
        },
        serialize::TKETDecode,
        utils::build_simple_circuit,
        TketOp,
    };

    use super::*;

    #[fixture]
    fn three_hadamards() -> ResourceScope {
        let circ = build_simple_circuit(1, |circ| {
            for _ in 0..3 {
                circ.append(TketOp::H, vec![0]).unwrap();
            }
            Ok(())
        })
        .unwrap();
        ResourceScope::from_circuit(circ)
    }

    #[rstest]
    fn test_hugr_match_adapter(three_hadamards: ResourceScope) {
        ///// Test matching, default behaviour
        let matcher = TestHadamardMatcher.as_hugr_matcher();
        let matches = matcher.get_all_matches(&three_hadamards, &MatchingOptions::default());
        let matches_by_nodes = matches.into_iter().into_group_map_by(|(subcirc, _)| {
            subcirc.nodes(&three_hadamards).collect::<BTreeSet<_>>()
        });

        // there are two pairs of Hadamards, and one set of three Hadamards
        assert_eq!(matches_by_nodes.len(), 2 + 1);

        for (nodes, matches) in matches_by_nodes {
            // There are as many (duplicate) matches as there are nodes in the match
            // (each node can be chosen as the root)
            assert_eq!(matches.len(), nodes.len());
        }

        ///// Test matching only maximal matches
        let matches = matcher.get_all_matches(
            &three_hadamards,
            &MatchingOptions::default().only_maximal_matches(),
        );
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].0.node_count(&three_hadamards), 3);

        ///// Test matching with deduplication
        let matches: [_; 3] = matcher
            .get_all_matches(&three_hadamards, &MatchingOptions::with_deduplication())
            .into_iter()
            .collect_array()
            .expect("3 matches");

        // there are two pairs of Hadamards, and one set of three Hadamards
        let mut match_sizes = matches.map(|(subg, _)| subg.node_count(&three_hadamards));
        match_sizes.sort_unstable();
        assert_eq!(match_sizes, [2, 2, 3]);

        ///// Test matching with deduplication and only maximal matches
        let match_ = matcher
            .get_all_matches(
                &three_hadamards,
                &MatchingOptions::with_deduplication().only_maximal_matches(),
            )
            .into_iter()
            .exactly_one()
            .expect("1 match");

        assert_eq!(match_.0.node_count(&three_hadamards), 3);
    }

    #[test]
    fn test_hugr_match_adapter_const_f64() {
        // A circuit with two constant angle Rz gates, one of them is 0.123.
        const CIRC: &str = r#"{"bits": [], "commands": [{"args": [["q", [0]]], "op": {"params": ["0.123"], "type": "Rz"}}, {"args": [["q", [0]]], "op": {"params": ["0.5"], "type": "Rz"}}], "created_qubits": [], "discarded_qubits": [], "implicit_permutation": [[["q", [0]], ["q", [0]]]], "phase": "0.0", "qubits": [["q", [0]]]}"#;
        let ser_circ: tket_json_rs::SerialCircuit = serde_json::from_str(CIRC).unwrap();
        let circuit = ResourceScope::from_circuit(ser_circ.decode().unwrap());

        let matcher = TestRzMatcher.as_hugr_matcher();
        let (match_subcirc, ()) = matcher
            .get_all_matches(&circuit, &MatchingOptions::default())
            .into_iter()
            .exactly_one()
            .unwrap();

        assert_eq!(match_subcirc.node_count(&circuit), 1);
        assert_eq!(
            match_subcirc.dataflow_signature(&circuit),
            Signature::new(vec![qb_t(), rotation_type()], vec![qb_t()],)
        );
        assert_eq!(
            match_subcirc.nodes(&circuit).collect::<BTreeSet<_>>(),
            match_subcirc
                .try_to_subgraph(&circuit)
                .unwrap()
                .nodes()
                .iter()
                .copied()
                .collect::<BTreeSet<_>>(),
        )
    }
}
