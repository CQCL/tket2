//! Factory producing rewrites that simplify Clifford circuits.

use std::{
    collections::{BTreeSet, VecDeque},
    iter,
};

use crate::{
    rewrite_space::{factory::value_ports, CommitFactory, IterMatched, PersistentHugr, Walker},
    serialize::TKETDecode,
    Circuit, Tk2Op,
};
use hugr::{
    extension::simple_op::MakeExtensionOp,
    hugr::views::SiblingSubgraph,
    ops::OpType,
    persistent::{PatchNode, PersistentWire},
    Direction, Hugr, HugrView, Port,
};
use itertools::Itertools;
use tket_json_rs::SerialCircuit;

/// Factory producing rewrites that simplify Clifford circuits using TKET1's
/// `clifford_simp` pass.
pub struct CliffordSimpFactory;

/// A rewrite produced by [`CliffordSimpFactory`] that simplifies a Clifford
/// circuit.
///
/// Store a HUGR subgraph with pure quantum dataflow and only Clifford
/// operations.
#[derive(Debug, Clone)]
pub struct CliffordSubcircuit {
    root: PatchNode,
    wires: BTreeSet<PersistentWire>,
}

impl CliffordSubcircuit {
    /// Create a new empty match.
    fn new(root: PatchNode) -> Self {
        Self {
            root,
            wires: BTreeSet::new(),
        }
    }

    /// All nodes that are invalidated by the rewrite.
    pub fn invalidated_nodes<'h>(
        &'h self,
        host: &'h PersistentHugr,
    ) -> impl Iterator<Item = PatchNode> + 'h {
        iter::once(self.root)
            .chain(
                self.wires
                    .iter()
                    .flat_map(|w| w.all_ports(host, None).map(|(n, _)| n)),
            )
            .unique()
    }
}

impl IterMatched for CliffordSubcircuit {
    fn matched_wires(&self) -> impl Iterator<Item = &PersistentWire> + '_ {
        self.wires.iter()
    }

    fn matched_isolated_nodes(&self) -> impl Iterator<Item = PatchNode> + '_ {
        // if the pattern is just the root, it is an isolated node
        self.wires.is_empty().then_some(self.root).into_iter()
    }
}

impl CommitFactory for CliffordSimpFactory {
    type PatternMatch = CliffordSubcircuit;

    type Cost = usize;

    const PATTERN_RADIUS: usize = 5;

    fn get_replacement(
        &self,
        _pattern_match: &Self::PatternMatch,
        matched_subgraph: &SiblingSubgraph<PatchNode>,
        host: &PersistentHugr,
    ) -> Option<Hugr> {
        let extracted_match =
            Circuit::new(matched_subgraph.extract_subgraph(host, "CliffordResynthesis"));
        let resyn_circ = apply_clifford_simp(&extracted_match)?;
        Some(resyn_circ.into_hugr())
    }

    fn find_pattern_matches<'w: 'a, 'a>(
        &'a self,
        pattern_root: PatchNode,
        walker: Walker<'w>,
    ) -> impl Iterator<Item = (Self::PatternMatch, Walker<'w>)> + 'a {
        find_clifford_pattern_matches(pattern_root, walker).into_iter()
    }
}

#[derive(Debug, Clone)]
struct PatternMatchState<'w> {
    /// The (currently incomplete) match.
    curr_match: CliffordSubcircuit,
    /// Ports to traverse next for matching.
    active_ports: VecDeque<(PatchNode, Port)>,
    /// The walker to use for matching.
    walker: Walker<'w>,
    /// The root node of the match.
    root: PatchNode,
    /// Nodes of the pattern such that no node in their past has been
    /// matched.
    ///
    /// Used to ensure that in all returned matches: root <= first_nodes.
    /// This guarantees that every Clifford pattern is returned exactly
    /// once.
    first_nodes: BTreeSet<PatchNode>,
}

fn find_clifford_pattern_matches(
    pattern_root: PatchNode,
    walker: Walker,
) -> Vec<(CliffordSubcircuit, Walker)> {
    // Matches to be returned
    let mut all_matches = Vec::new();

    if !is_clifford(walker.as_hugr_view().get_optype(pattern_root)) {
        return all_matches;
    }

    // Initialise queue
    let mut queue = VecDeque::new();
    let root_active_ports = value_ports(walker.as_hugr_view(), pattern_root, Direction::Outgoing)
        .map(|p| (pattern_root, p))
        .collect();
    queue.push_back(PatternMatchState {
        curr_match: CliffordSubcircuit::new(pattern_root),
        active_ports: root_active_ports,
        walker,
        root: pattern_root,
        first_nodes: BTreeSet::from_iter([pattern_root]),
    });

    while let Some(mut state) = queue.pop_front() {
        let Some((node, port)) = state.active_ports.pop_front() else {
            // Only add match if root is the smallest first node.
            if state.first_nodes.iter().min() == Some(&state.root) {
                all_matches.push((state.curr_match, state.walker));
            }
            continue;
        };

        let wire = state.walker.get_wire(node, port);
        let walkers = if state.walker.is_complete(&wire, port.direction().reverse()) {
            vec![state.walker.clone()]
        } else {
            state
                .walker
                .expand(&wire, port.direction().reverse())
                .collect_vec()
        };
        for expanded_walker in walkers {
            let expanded_wire = expanded_walker.get_wire(node, port);
            debug_assert!(
                expanded_walker.is_complete(&expanded_wire, None),
                "quantum wire must have two (pinned) endpoints"
            );
            let (opp_node, opp_port) = expanded_walker
                .wire_pinned_ports(&expanded_wire, port.direction().reverse())
                .exactly_one()
                .ok()
                .expect("wire is quantum and complete");

            let PatternMatchState {
                mut curr_match,
                mut active_ports,
                root,
                mut first_nodes,
                ..
            } = state.clone();

            // Stop matching if we hit a non-Clifford op.
            if !is_clifford(expanded_walker.as_hugr_view().get_optype(opp_node)) {
                queue.push_back(PatternMatchState {
                    curr_match,
                    active_ports,
                    walker: expanded_walker,
                    root,
                    first_nodes,
                });
                continue;
            }

            // Extend match
            curr_match.wires.insert(expanded_wire.clone());

            // If we crossed through an incoming port, `node` cannot be a first node
            if port.direction() == Direction::Incoming {
                first_nodes.remove(&node);
            }

            // If this is a new node (was not pinned before), add all its ports to the
            // queue.
            if !state.walker.is_pinned(opp_node) {
                let new_ports = value_ports(expanded_walker.as_hugr_view(), opp_node, None)
                    .filter(|&p| p != opp_port)
                    .map(|p| (opp_node, p));
                active_ports.extend(new_ports);
                // a new node -> possibly a first node
                first_nodes.insert(opp_node);
            }

            // Enqueue
            queue.push_back(PatternMatchState {
                curr_match,
                active_ports,
                walker: expanded_walker,
                root,
                first_nodes,
            });
        }
    }

    all_matches
}

fn as_tk2_op(op: &OpType) -> Option<Tk2Op> {
    let Some(ext_op) = op.as_extension_op() else {
        return None;
    };
    Tk2Op::from_extension_op(ext_op).ok()
}

fn is_clifford(op: &OpType) -> bool {
    as_tk2_op(op).map_or(false, |op| op.is_clifford())
}

fn apply_clifford_simp(circuit: &Circuit) -> Option<Circuit> {
    let encoded = SerialCircuit::encode(circuit).ok()?;
    let mut tk1_circ = tket1_passes::Tket1Circuit::from_serial_circuit(&encoded).ok()?;
    tk1_circ
        .clifford_simp(tket_json_rs::OpType::CX, true)
        .ok()?;
    let encoded = tk1_circ.to_serial_circuit().ok()?;
    encoded.decode().ok()
}

#[cfg(test)]
mod tests {
    use std::{fs::File, io::BufReader};

    use crate::Circuit;

    use super::*;
    use hugr::{
        builder::{endo_sig, DFGBuilder, Dataflow, DataflowHugr},
        extension::prelude::qb_t,
        SimpleReplacement,
    };
    use rstest::{fixture, rstest};

    const SIX_CZ: &str = "../test_files/6cz.hugr";

    #[fixture]
    fn six_cz_hugr() -> PersistentHugr {
        let f = File::open(SIX_CZ).unwrap();
        let circ = Circuit::load(BufReader::new(f), None).unwrap();
        PersistentHugr::with_base(circ.into_hugr())
    }

    #[fixture]
    fn six_cz_hugr_plus_one_commit() -> PersistentHugr {
        let f = File::open(SIX_CZ).unwrap();
        let circ = Circuit::load(BufReader::new(f), None).unwrap();
        let mut hugr = PersistentHugr::with_base(circ.into_hugr());
        let one_cz = {
            let mut builder = DFGBuilder::new(endo_sig(vec![qb_t(), qb_t()])).unwrap();
            let [q0, q1] = builder.input_wires_arr();
            let cz = builder.add_dataflow_op(Tk2Op::CZ, vec![q0, q1]).unwrap();
            let [q0, q1] = cz.outputs_arr();
            builder.finish_hugr_with_outputs(vec![q0, q1]).unwrap()
        };
        let last_cz = hugr
            .nodes()
            .filter(|n| is_clifford(hugr.get_optype(*n)))
            .max()
            .unwrap();
        let repl =
            SimpleReplacement::try_new(SiblingSubgraph::from_node(last_cz, &hugr), &hugr, one_cz)
                .unwrap();
        hugr.add_replacement(repl);
        hugr
    }

    #[rstest]
    fn test_find_clifford_pattern_matches(six_cz_hugr: PersistentHugr) {
        let factory = CliffordSimpFactory;

        let cz_nodes: BTreeSet<_> = six_cz_hugr
            .nodes()
            .filter(|&n| is_clifford(six_cz_hugr.get_optype(n)))
            .collect();
        let &min_cz = cz_nodes.first().unwrap();
        let &max_cz = cz_nodes.last().unwrap();

        // There is a pattern match when
        //  - the root is the smallest CZ (pattern: all 6 cz)
        //  - the root is the last CZ (pattern: just itself)
        for &cz in cz_nodes.iter() {
            let walker = Walker::from_pinned_node(cz, six_cz_hugr.as_state_space());
            let mut matches = factory.find_pattern_matches(cz, walker);
            match cz {
                cz if cz == min_cz => {
                    let (match_, _) = matches.exactly_one().ok().unwrap();
                    // two outgoing wires for each CZ, minus three that lead to output
                    assert_eq!(match_.wires.len(), 2 * cz_nodes.len() - 3);
                    assert_eq!(
                        BTreeSet::from_iter(match_.invalidated_nodes(&six_cz_hugr)),
                        cz_nodes
                    );
                }
                cz if cz == max_cz => {
                    let (match_, _) = matches.exactly_one().ok().unwrap();
                    assert_eq!(
                        match_.invalidated_nodes(&six_cz_hugr).collect_vec(),
                        vec![max_cz]
                    );
                }
                _ => {
                    assert!(matches.next().is_none());
                }
            }
        }
    }

    #[rstest]
    fn test_find_clifford_pattern_matches_non_trivial_history(
        six_cz_hugr_plus_one_commit: PersistentHugr,
    ) {
        let per_hugr = six_cz_hugr_plus_one_commit;
        let factory = CliffordSimpFactory;

        let cz_nodes: BTreeSet<_> = per_hugr
            .nodes()
            .filter(|&n| n.owner() == per_hugr.base() && is_clifford(per_hugr.get_optype(n)))
            .collect();
        let &min_cz = cz_nodes.first().unwrap();

        // There are two matches, both selecting all nodes, one with the new
        // commit and one without.
        let walker = Walker::from_pinned_node(min_cz, per_hugr.as_state_space());
        assert!(per_hugr.contains_node(min_cz));
        let matches: [_; 2] = factory
            .find_pattern_matches(min_cz, walker)
            .collect_array()
            .unwrap();

        let mut expected_commits = BTreeSet::from_iter([
            BTreeSet::from_iter([per_hugr.base()]),
            per_hugr.all_commit_ids().collect(),
        ]);

        for (match_, walker) in matches {
            let subgraph = match_.to_subgraph(&walker).unwrap();
            let hugr_subgraph = subgraph.to_sibling_subgraph(walker.as_hugr_view()).unwrap();
            assert_eq!(hugr_subgraph.nodes().len(), 6);
            let commits = hugr_subgraph
                .nodes()
                .into_iter()
                .map(|n| n.owner())
                .unique()
                .collect();
            assert!(expected_commits.remove(&commits));
        }
    }
}
