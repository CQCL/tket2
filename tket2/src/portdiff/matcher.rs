use std::collections::{BTreeMap, BTreeSet};

use hugr::{Direction, Port, PortIndex};
use itertools::Itertools;
use portdiff::{self as pd, port_diff::Owned, Graph, PortDiff, Site};
use portmatching::{
    self as pm, indexing as pmx, IndexingScheme, ManyMatcher, PatternFallback, PortMatcher,
};

use crate::{
    portmatching::{
        indexing::{CircuitPath, OpLocationMap, PatternOpPosition, StaticIndexScheme},
        pattern::InvalidStaticPattern,
        predicate::Predicate,
    },
    static_circ::{OpPosition, StaticSizeCircuit},
};

pub type DiffCircuit = PortDiff<StaticSizeCircuit>;

/// A matcher object for fast pattern matching on circuits using diffed rewriting.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DiffCircuitMatcher(
    ManyMatcher<StaticSizeCircuit, PatternOpPosition, Predicate, StaticIndexScheme>,
);

impl PortMatcher<DiffCircuit> for DiffCircuitMatcher {
    type Match = <StaticIndexScheme as IndexingScheme<DiffCircuit>>::Map;

    fn find_matches<'a>(
        &'a self,
        host: &'a DiffCircuit,
    ) -> impl Iterator<Item = pm::PatternMatch<Self::Match>> + 'a {
        self.0.find_matches(host)
    }
}

impl DiffCircuitMatcher {
    /// Create a new DiffCircuitMatcher from a list of patterns.
    pub fn try_from_patterns(
        patterns: Vec<StaticSizeCircuit>,
    ) -> Result<Self, InvalidStaticPattern> {
        let matcher = ManyMatcher::try_from_patterns(patterns, PatternFallback::Skip)?;
        Ok(DiffCircuitMatcher(matcher))
    }
}

impl pm::Predicate<DiffCircuit> for Predicate {
    type Value = Owned<OpPosition, StaticSizeCircuit>;

    fn check(&self, _: &DiffCircuit, args: &[impl std::borrow::Borrow<Self::Value>]) -> bool {
        match self {
            &Predicate::Link { out_port, in_port } => {
                let Owned {
                    data: out_pos,
                    owner: out_owner,
                } = args[0].borrow();
                let Owned {
                    data: in_pos,
                    owner: in_owner,
                } = args[1].borrow();
                // Make sure the location refers to the right port
                let Some(out_pos) = out_owner.graph().equivalent_position(*out_pos, out_port)
                else {
                    return false;
                };
                let Some(in_pos) = in_owner.graph().equivalent_position(*in_pos, in_port) else {
                    return false;
                };
                let opp_out_pos = follow_link(
                    &Owned {
                        data: out_pos,
                        owner: out_owner.clone(),
                    },
                    Direction::Outgoing,
                    &[],
                );
                opp_out_pos.contains(&Owned {
                    data: in_pos,
                    owner: in_owner.clone(),
                })
            }
            &Predicate::IsOp { op } => {
                let loc = args[0].borrow();
                let id = loc.owner.graph().at_position(loc.data).unwrap();
                loc.owner.graph().get(id).map(|op| op.op) == Some(op)
            }
            Predicate::SameOp { .. } => args.iter().tuple_windows().all(|(a, b)| {
                let pos_a = a.borrow();
                let pos_b = b.borrow();
                if pos_a.owner != pos_b.owner {
                    return false;
                }
                let data = pos_a.owner.graph();
                data.at_position(pos_a.data) == data.at_position(pos_b.data)
            }),
            &Predicate::DistinctQubits { .. } => {
                let mut qubits = BTreeMap::new();
                for loc in args.iter().map(|a| a.borrow()) {
                    let owner_qubits: &mut BTreeSet<_> =
                        qubits.entry(PortDiff::as_ptr(&loc.owner)).or_default();
                    if !owner_qubits.insert(loc.data.qubit) {
                        return false;
                    }
                }
                true
            }
        }
    }
}

impl pmx::IndexingScheme<DiffCircuit> for StaticIndexScheme {
    type Map = OpLocationMap<CircuitPath, Owned<OpPosition, StaticSizeCircuit>>;

    fn valid_bindings(
        &self,
        key: &pmx::Key<Self, DiffCircuit>,
        known_bindings: &Self::Map,
        data: &DiffCircuit,
    ) -> pmx::BindingResult<Self, DiffCircuit> {
        let get_known = |key| <Self::Map as pmx::IndexMap>::get(known_bindings, key);
        let known_diffs = known_bindings
            .values()
            .map(|v| v.owner.clone())
            .collect_vec();
        if let Some(v) = get_known(key) {
            // Already bound.
            Ok(vec![v.clone()].into())
        } else {
            let mut all_locs = key.all_locations_on_path();
            all_locs.pop(); // Remove the last location: the one we are looking for

            if all_locs.is_empty() {
                // It is a root of the pattern, all locations are valid
                // Heuristic: we consider nodes at neighbours of the current diff
                let diffs = all_neighbours(data);
                return Ok(Vec::from_iter(diffs.flat_map(|diff| {
                    diff.graph()
                        .ops_iter()
                        .map(|op| Owned {
                            data: diff.graph().get_position(op, 0).unwrap(),
                            owner: diff.clone(),
                        })
                        .collect_vec()
                }))
                .into());
            }

            let missing_keys = all_locs
                .iter()
                .filter(|loc| get_known(loc).is_none())
                .cloned()
                .collect_vec();
            if !missing_keys.is_empty() {
                return Err(pmx::MissingIndexKeys(missing_keys));
            }

            let last_loc = get_known(all_locs.last().unwrap()).unwrap();
            if key.op_idx == 0 {
                // Same op, but on a different port
                let curr_circ = last_loc.owner.graph();
                let port = key.qubit.port(key.qubit.len() - 1) as usize;
                let new_loc = curr_circ.equivalent_position(last_loc.data, port);
                Ok(Vec::from_iter(new_loc.map(|new_loc| Owned {
                    data: new_loc,
                    owner: last_loc.owner.clone(),
                }))
                .into())
            } else {
                Ok(follow_link(
                    last_loc,
                    if key.op_idx > 0 {
                        Direction::Outgoing
                    } else {
                        Direction::Incoming
                    },
                    &known_diffs,
                )
                .into())
            }
        }
    }
}

/// All locations at the other end of a link from `loc`, both in the same diff
/// and in compatible diffs.
fn follow_link(
    pos: &Owned<OpPosition, StaticSizeCircuit>,
    dir: Direction,
    known_diffs: &[PortDiff<StaticSizeCircuit>],
) -> Vec<Owned<OpPosition, StaticSizeCircuit>> {
    let curr_circ = pos.owner.graph();
    let offset = curr_circ.position_offset(pos.data).unwrap();
    let node = curr_circ.at_position(pos.data).unwrap();
    let port = Port::new(dir, offset);

    let site = Site { node, port };
    // All ports that correspond to site (note: in static circuits there should be at most 1)
    let mut all_ports = Vec::new();

    // Check if there is a boundary port that corresponds to site
    all_ports.extend(
        pos.owner
            .boundary_iter()
            .find(|b| pos.owner.boundary_site(*b).try_as_site_ref() == Some(&site))
            .map(pd::Port::Boundary),
    );
    // Find bound ports that corresponds to site
    all_ports.extend(curr_circ.get_bound_ports(site).map(pd::Port::Bound));

    let mut ret = Vec::new();
    // Follow link within curr_circ if there is one
    if let Some((linked_port, linked_op)) = curr_circ.linked_op(node, port) {
        let data = curr_circ
            .get_position(linked_op, linked_port.index())
            .unwrap();
        ret.push(Owned {
            owner: pos.owner.clone(),
            data,
        });
    }
    // Find opposite ports for all ports in all_ports
    for port in all_ports {
        ret.extend(
            pos.owner
                .opposite_ports(port)
                .into_iter()
                .filter(|p| PortDiff::are_compatible(known_diffs.iter().chain([&p.owner])))
                .flat_map(|p|
                    // This handles empty wires in the graph by recursively finding the
                    // first "real" port.
                    p.owner.resolve_port(p.data)
                )
                .filter_map(|p| {
                    let data = p
                        .owner
                        .graph()
                        .get_position(p.site().unwrap().node, p.site().unwrap().port.index())
                        .unwrap();
                    Some(Owned {
                        data,
                        owner: p.owner,
                    })
                }),
        );
    }

    ret
}

fn all_neighbours(
    data: &PortDiff<StaticSizeCircuit>,
) -> impl Iterator<Item = PortDiff<StaticSizeCircuit>> + '_ {
    let bound_ports = data
        .graph()
        .nodes_iter()
        .flat_map(|node| data.graph().get_sites(node))
        .flat_map(|site| data.graph().get_bound_ports(site))
        .map(pd::Port::Bound);
    let boundary_ports = data.boundary_iter().map(pd::Port::Boundary);

    bound_ports
        .chain(boundary_ports)
        .flat_map(|port| data.opposite_ports(port))
        .map(|port| port.owner)
        .filter(move |other| pd::PortDiff::are_compatible([data, other]))
}
