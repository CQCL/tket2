use std::{
    cmp::Reverse,
    collections::{BTreeSet, VecDeque},
    hash::{Hash, Hasher},
};

use hugr::{
    extension::prelude::qb_t,
    hugr::views::SiblingSubgraph,
    ops::{OpTag, OpTrait, OpType},
    std_extensions::arithmetic::float_types::{float64_type, ConstF64},
    types::Type,
    Direction, HugrView, IncomingPort, Node, NodeIndex, Port,
};
use itertools::{Either, Itertools};
use portgraph::{algorithms::convex::LineIndex, UnmanagedDenseMap};

use super::{as_tket_op, CircuitMatcher, MatchContext, MatchOutcomeEnum, OpArg};
use crate::{extension::rotation::rotation_type, subcircuit::LineConvexChecker, Subcircuit};
use crate::{
    extension::rotation::{ConstRotation, RotationOp},
    Circuit,
};

/// An adaptor for [`CircuitMatcher`]s that matches patterns in concrete circuits.
pub struct HugrMatchAdapter<'m, M: ?Sized> {
    pub(super) matcher: &'m M,
}

impl<'m, M: ?Sized> HugrMatchAdapter<'m, M> {
    /// Create a new [`HugrMatchAdapter`].
    pub fn new(matcher: &'m M) -> Self {
        Self { matcher }
    }
}
#[derive(Debug, Clone)]
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

        intervals.sort_unstable_by_key(|&(l, _)| l);
        active_ports.sort_unstable_by_key(|&(n, p)| (n, p));

        intervals.hash(&mut hasher);
        active_ports.hash(&mut hasher);
        partial_match_hash(&self.match_info, &mut hasher);

        hasher.finish()
    }
}

/// A function that hashes a value into an [`fxhash::FxHasher`].
type HashFn<T> = Box<dyn Fn(&T, &mut fxhash::FxHasher)>;

/// Options for matching circuits with [`HugrMatchAdapter`].
#[non_exhaustive]
pub struct MatchingOptions<PartialMatchInfo, MatchInfo> {
    /// A hash function to deduplicate partial matches during matching.
    ///
    /// When this option is set, the matcher will keep track of all seen partial
    /// matches and will abort the matching process early if a match has been
    /// seen before.
    ///
    /// The provided function will be called for every partial match and should
    /// hash the match into the provided hasher.
    ///
    /// ## Performance considerations
    ///
    /// A naively implemented `CircuitMatcher` running without deduplication
    /// may perform many redundant operations and run slowly. On the other hand,
    /// deduplication of partial matches comes at a significant performance cost.
    ///
    /// The recommended approach is to design matchers that run well without
    /// deduplication by ensuring that matches are unique by construction rather
    /// than relying on hashing. An example of a deduplication strategy within
    /// the matcher would be to never match nodes with an index smaller than the
    /// first matched node.
    pub deduplicate_partial_matches: Option<HashFn<PartialMatchInfo>>,

    /// A hash function to deduplicate complete matches.
    ///
    /// When this option is set, the matcher will ensure that every returned match
    /// is unique. The provided function will be called for every complete match
    /// and should hash the match into the provided hasher.
    pub deduplicate_complete_matches: Option<HashFn<MatchInfo>>,

    /// If set, will discard all non-maximal matches.
    ///
    /// This may be an expensive operation: it may take time up to quadratic in
    /// the total number of matches (and linear in the size of the largest match).
    pub only_maximal_matches: bool,
}

impl<PartialMatchInfo, MatchInfo> Default for MatchingOptions<PartialMatchInfo, MatchInfo> {
    fn default() -> Self {
        Self {
            deduplicate_partial_matches: None,
            deduplicate_complete_matches: None,
            only_maximal_matches: false,
        }
    }
}

impl<PartialMatchInfo: Hash, MatchInfo: Hash> MatchingOptions<PartialMatchInfo, MatchInfo> {
    /// Create a new [`MatchingOptions`] with deduplication enabled.
    ///
    /// This will use the `Hash` implementation of `PartialMatchInfo` to deduplicate
    /// partial matches.
    ///
    /// ## Performance considerations
    ///
    /// A naively implemented `CircuitMatcher` running without deduplication
    /// may perform many redundant operations and run slowly. On the other hand,
    /// deduplication of partial matches comes at a significant performance cost.
    ///
    /// The recommended approach is to design matchers that run well without
    /// deduplication by ensuring that matches are unique by construction rather
    /// than relying on hashing. An example of a deduplication strategy within
    /// the matcher would be to never match nodes with an index smaller than the
    /// first matched node.
    pub fn with_deduplication() -> Self {
        Self {
            deduplicate_partial_matches: Some(Box::new(|info, hasher| info.hash(hasher))),
            deduplicate_complete_matches: Some(Box::new(|info, hasher| info.hash(hasher))),
            only_maximal_matches: false,
        }
    }

    /// Only return maximal matches during matches.
    ///
    /// In other words, never return matches that are fully contained within
    /// another match.
    ///
    /// This may be an expensive operation: it may take time up to quadratic in
    /// the total number of matches (and linear in the size of the largest match).
    pub fn only_maximal_matches(self) -> Self {
        Self {
            only_maximal_matches: true,
            ..self
        }
    }
}

impl<'m, M: CircuitMatcher + ?Sized> HugrMatchAdapter<'m, M> {
    /// Get all matching subcircuits within the Circuit.
    pub fn get_all_matches<H: HugrView<Node = Node>>(
        &self,
        circ: &Circuit<H>,
        options: &MatchingOptions<M::PartialMatchInfo, M::MatchInfo>,
    ) -> Vec<(SiblingSubgraph, M::MatchInfo)> {
        let mut queue = VecDeque::new();
        let mut all_matches = Vec::new();
        let checker = LineConvexChecker::from_entrypoint(circ.hugr());

        // hashes of visited states, only used if deduplication is enabled
        let mut visited_states = options
            .deduplicate_partial_matches
            .as_ref()
            .map(|_| BTreeSet::<u64>::new());
        // hashes of complete matches, only used if deduplication is enabled
        let mut complete_matches = options
            .deduplicate_complete_matches
            .as_ref()
            .map(|_| BTreeSet::<u64>::new());

        // 1. Initialise queue
        for cmd in circ.commands() {
            let empty_match = MatchState {
                subcircuit: Subcircuit::new_empty(),
                active_ports: VecDeque::new(),
                match_info: M::PartialMatchInfo::default(),
            };

            let mut new_matches = enqueue_extended_matches(
                circ,
                empty_match,
                cmd.node(),
                self.matcher,
                &checker,
                &mut queue,
                None,
            );

            if let Some(deduplicate_complete_matches) =
                options.deduplicate_complete_matches.as_ref()
            {
                new_matches.retain(|(subcirc, info)| {
                    let hash = fx_hash_subcirc_info(subcirc, info, deduplicate_complete_matches);
                    let complete_matches = complete_matches.as_mut().expect("enabled");
                    complete_matches.insert(hash)
                });
            }

            all_matches.extend(new_matches.into_iter().filter_map(|(subcirc, info)| {
                let subgraph = subcirc.try_to_subgraph(&checker).ok()?;
                Some((subgraph, info))
            }));
        }

        // 2. Find all matches
        while let Some(mut current_match) = queue.pop_front() {
            if let Some(partial_match_hash) = options.deduplicate_partial_matches.as_ref() {
                let hash = current_match.fx_hash(partial_match_hash);
                let visited_states = visited_states.as_mut().expect("enabled");
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
                circ,
                current_match,
                new_node,
                self.matcher,
                &checker,
                &mut queue,
                Some(new_port),
            );

            if let Some(deduplicate_complete_matches) =
                options.deduplicate_complete_matches.as_ref()
            {
                new_matches.retain(|(subcirc, info)| {
                    let hash = fx_hash_subcirc_info(subcirc, info, deduplicate_complete_matches);
                    let complete_matches = complete_matches.as_mut().expect("enabled");
                    complete_matches.insert(hash)
                });
            }

            all_matches.extend(new_matches.into_iter().filter_map(|(subcirc, info)| {
                let subgraph = subcirc.try_to_subgraph(&checker).ok()?;
                Some((subgraph, info))
            }));
        }

        if options.only_maximal_matches {
            remove_non_maximal_matches(&mut all_matches);
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
    intervals.sort_unstable_by_key(|&(l, _)| l);

    let mut hasher = fxhash::FxHasher::default();

    intervals.hash(&mut hasher);
    deduplicate_complete_matches(info, &mut hasher);

    hasher.finish()
}

impl<H: HugrView> Circuit<H> {
    /// Get the arguments of an operation and the nodes that the operation span.
    ///
    /// The operation "spans" at least `new_node`, but might span further nodes,
    /// typically related to constant definitions.
    fn get_op_args(
        &self,
        node: H::Node,
        checker: &LineConvexChecker<H>,
    ) -> Option<(Vec<OpArg>, Vec<H::Node>)> {
        let mut op_nodes = vec![node];
        let inputs = self.hugr().in_value_types(node);
        let outputs = self.hugr().out_value_types(node);

        let mut args = Vec::with_capacity(inputs.size_hint().0);

        for io in inputs.zip_longest(outputs) {
            use itertools::EitherOrBoth::*;
            match io {
                Both(inp, out) => {
                    // pass-through argument: currently only support qubit
                    let line = get_through_line(
                        node,
                        [(inp.0.into(), inp.1), (out.0.into(), out.1)],
                        qb_t(),
                        checker,
                    )?;

                    args.push(OpArg::Qubit(line));
                }
                Left((inp, inp_type)) => {
                    // input-only argument: currently only support constant F64
                    // (or rotation type converted from a constant F64)
                    if ![float64_type(), rotation_type()].contains(&inp_type) {
                        return None;
                    }
                    let (const_f64, add_nodes) = get_const_f64(node, inp, self.hugr())?;
                    op_nodes.extend(add_nodes);
                    args.push(OpArg::ConstF64(const_f64));
                }
                Right(_) => {
                    // output-only argument not supported
                    return None;
                }
            }
        }

        Some((args, op_nodes))
    }
}

fn enqueue_extended_matches<M: CircuitMatcher + ?Sized, H: HugrView<Node = Node>>(
    circ: &Circuit<H>,
    current_match: MatchState<M::PartialMatchInfo>,
    new_node: Node,
    matcher: &M,
    checker: &LineConvexChecker<H>,
    queue: &mut VecDeque<MatchState<M::PartialMatchInfo>>,
    // The port `new_node` was reached from, or none if this is the first node
    traversed_port: Option<Port>,
) -> Vec<(Subcircuit, M::MatchInfo)> {
    let MatchState {
        subcircuit: mut new_subcircuit,
        active_ports,
        ..
    } = current_match.clone();

    let mut all_matches = Vec::new();

    let Some((args, op_nodes)) = circ.get_op_args(new_node, checker) else {
        // Unsupported op
        queue.push_back(current_match);
        return all_matches;
    };
    for node in op_nodes {
        if !new_subcircuit.try_extend(node, checker) {
            // non-convex match
            queue.push_back(current_match);
            return all_matches;
        }
    }

    if new_subcircuit == current_match.subcircuit {
        // no extension; already matched
        queue.push_back(current_match);
        return all_matches;
    }

    let Some(tket_op) = as_tket_op(circ.hugr().get_optype(new_node)) else {
        queue.push_back(current_match);
        return all_matches;
    };
    // Change op arguments to be relative to the subcircuit already matched
    let args = args
        .into_iter()
        .map(|arg| {
            arg.relative_to(new_node, &current_match.subcircuit, checker)
                .expect("extending to neighbouring node")
        })
        .collect_vec();

    let match_context = MatchContext {
        subcircuit: &current_match.subcircuit,
        match_info: current_match.match_info,
        checker,
        op_node: new_node,
    };
    for match_ in matcher
        .match_tket_op(tket_op, &args, match_context)
        .into_enum_vec()
    {
        match match_ {
            MatchOutcomeEnum::Complete(match_info) => {
                all_matches.push((new_subcircuit.clone(), match_info));
            }
            MatchOutcomeEnum::Proceed(match_info) => {
                let mut active_ports = active_ports.clone();
                active_ports.extend(
                    ports_of_type(circ.hugr(), new_node, qb_t(), None)
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

/// Get the unique line that passes through all the listed ports at `node`.
///
/// `expected_type` should be a linear type: this function will panic if any
/// of the given ports are not attached to exactly one line.
fn get_through_line<H: HugrView>(
    node: H::Node,
    ports: impl IntoIterator<Item = (Port, Type)>,
    expected_type: Type,
    checker: &LineConvexChecker<H>,
) -> Option<LineIndex> {
    let mut expected_line = None;
    for (port, typ) in ports {
        if typ != expected_type {
            return None;
        }

        let &line = checker
            .lines_at_port(node, port)
            .iter()
            .exactly_one()
            .expect("expected linear type");
        if let Some(expected_line) = expected_line {
            if expected_line != line {
                return None;
            }
        } else {
            expected_line = Some(line);
        }
    }

    expected_line
}

/// Return the constant F64 that is linked as input to `port` in `node`.
///
/// Follow the chain of nodes backwards from `port` to find a constant F64.
/// Ops allowed along the chain are conversions to Rotation type and LoadConst. and LoadConst.
///
/// Returns `None` if the port is not a constant F64 or rotation type value.
fn get_const_f64<H: HugrView>(
    node: H::Node,
    port: IncomingPort,
    host: &H,
) -> Option<(f64, Vec<H::Node>)> {
    let (mut find_const_node, _) = host.linked_outputs(node, port).exactly_one().ok().unwrap();
    let mut op_nodes = vec![node, find_const_node];
    loop {
        let op = host.get_optype(find_const_node);
        if let OpType::Const(const_val) = op {
            if let Some(const_rot) = const_val.value().get_custom_value::<ConstRotation>() {
                return Some((const_rot.half_turns(), op_nodes));
            } else if let Some(const_f64) = const_val.value().get_custom_value::<ConstF64>() {
                return Some((const_f64.value(), op_nodes));
            } else {
                panic!("unknown constant type: {:?}", const_val.value());
            }
        };
        if ![
            OpType::from(RotationOp::from_halfturns_unchecked),
            OpType::from(RotationOp::from_halfturns),
        ]
        .contains(op)
            && op.tag() != OpTag::LoadConst
        {
            return None;
        }
        let (next_node, _) = host
            .linked_outputs(find_const_node, 0)
            .exactly_one()
            .ok()
            .unwrap();
        op_nodes.push(next_node);
        find_const_node = next_node;
    }
}

fn ports_of_type<H: HugrView>(
    host: &H,
    node: H::Node,
    typ: Type,
    dir: impl Into<Option<Direction>>,
) -> impl Iterator<Item = Port> + '_ {
    let ports = match dir.into() {
        Some(dir) => Either::Left(host.value_types(node, dir)),
        None => Either::Right(
            host.value_types(node, Direction::Incoming)
                .chain(host.value_types(node, Direction::Outgoing)),
        ),
    };
    ports.filter_map(move |(port, t)| (t == typ).then_some(port))
}

/// Remove non-maximal subgraphs (i.e. subgraphs fully contained within another
/// subgraph) from the list of matches.
///
/// This may be an expensive operation: it may take time up to quadratic in
/// the total number of matches (and linear in the size of the largest match).
fn remove_non_maximal_matches<MatchInfo>(all_matches: &mut Vec<(SiblingSubgraph, MatchInfo)>) {
    // Sort matches from largest to smallest. Thus if A \subseteq B then B is
    // processed before A.
    all_matches.sort_unstable_by_key(|(subg, _)| Reverse(subg.node_count()));

    // A map from node indices to the set of matches that contain that node.
    let mut node_to_matches = UnmanagedDenseMap::<usize, BTreeSet<usize>>::new();
    // A counter to assign unique indices to each match.
    let mut match_ind = 0;

    all_matches.retain(|(subg, _)| {
        // Find a match that covers all nodes of subg
        let match_all_nodes = subg
            .nodes()
            .iter()
            .fold(None, |acc: Option<BTreeSet<usize>>, &n| {
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
            for &n in subg.nodes() {
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
    use rstest::{fixture, rstest};

    use crate::{
        rewrite::matcher::MatchOutcome, serialize::TKETDecode, utils::build_simple_circuit, TketOp,
    };

    use super::*;

    /// A matcher finding all sequences of two or more Hadamard gates.
    struct TestHadamardMatcher;
    type NumHadamards = usize;

    impl CircuitMatcher for TestHadamardMatcher {
        type PartialMatchInfo = NumHadamards;
        type MatchInfo = NumHadamards;

        fn match_tket_op(
            &self,
            op: TketOp,
            _op_args: &[OpArg],
            match_context: MatchContext<Self::PartialMatchInfo, impl HugrView>,
        ) -> MatchOutcome<Self::PartialMatchInfo, Self::MatchInfo> {
            // We can always skip this op
            let mut outcomes = MatchOutcome::default().skip(match_context.match_info);
            match op {
                TketOp::H => {
                    // we have a hadamard, so we can match this op and proceed
                    let num_hadamards = match_context.match_info + 1;
                    outcomes = outcomes.proceed(num_hadamards);
                    if num_hadamards >= 2 {
                        // We have enough hadamards to report the current match
                        outcomes.complete(num_hadamards)
                    } else {
                        // Proceed (without reporting a match)
                        outcomes
                    }
                }
                _ => outcomes,
            }
        }
    }

    #[fixture]
    fn three_hadamards() -> Circuit {
        build_simple_circuit(1, |circ| {
            for _ in 0..3 {
                circ.append(TketOp::H, vec![0]).unwrap();
            }
            Ok(())
        })
        .unwrap()
    }

    #[rstest]
    fn test_hugr_match_adapter(three_hadamards: Circuit) {
        ///// Test matching, default behaviour
        let matcher = TestHadamardMatcher.as_hugr_matcher();
        let matches = matcher.get_all_matches(&three_hadamards, &MatchingOptions::default());
        let matches_by_nodes = matches
            .into_iter()
            .into_group_map_by(|(subg, _)| subg.nodes().iter().copied().collect::<BTreeSet<_>>());

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
        assert_eq!(matches[0].0.node_count(), 3);

        ///// Test matching with deduplication
        let matches: [_; 3] = matcher
            .get_all_matches(&three_hadamards, &MatchingOptions::with_deduplication())
            .into_iter()
            .collect_array()
            .expect("3 matches");

        // there are two pairs of Hadamards, and one set of three Hadamards
        let mut match_sizes = matches.map(|(subg, _)| subg.node_count());
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

        assert_eq!(match_.0.node_count(), 3);
    }

    /// A matcher finding Rz gates with constant angle `0.123`.
    struct TestRzMatcher;

    impl CircuitMatcher for TestRzMatcher {
        type PartialMatchInfo = ();
        type MatchInfo = ();

        fn match_tket_op(
            &self,
            op: TketOp,
            op_args: &[OpArg],
            _match_context: MatchContext<Self::PartialMatchInfo, impl HugrView>,
        ) -> MatchOutcome<Self::PartialMatchInfo, Self::MatchInfo> {
            if op == TketOp::Rz && op_args[1] == OpArg::ConstF64(0.123) {
                MatchOutcome::default().complete(())
            } else {
                MatchOutcome::stop()
            }
        }
    }

    #[test]
    fn test_hugr_match_adapter_const_f64() {
        // A circuit with two constant angle Rz gates, one of them is 0.123.
        const CIRC: &str = r#"{"bits": [], "commands": [{"args": [["q", [0]]], "op": {"params": ["0.123"], "type": "Rz"}}, {"args": [["q", [0]]], "op": {"params": ["0.5"], "type": "Rz"}}], "created_qubits": [], "discarded_qubits": [], "implicit_permutation": [[["q", [0]], ["q", [0]]]], "phase": "0.0", "qubits": [["q", [0]]]}"#;
        let ser_circ: tket_json_rs::SerialCircuit = serde_json::from_str(CIRC).unwrap();
        let circuit = ser_circ.decode().unwrap();

        let matcher = TestRzMatcher.as_hugr_matcher();
        let match_ = matcher
            .get_all_matches(&circuit, &MatchingOptions::default())
            .into_iter()
            .exactly_one()
            .unwrap();
        // Match one Rz, one ConstF64, one LoadConst, one conversion to Rotation type
        assert_eq!(match_.0.node_count(), 4);
    }
}
