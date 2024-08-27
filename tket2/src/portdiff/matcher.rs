use hugr::{Direction, Port};
use itertools::Itertools;
use portdiff::{self as pd, port_diff::Owned, Graph, PortDiff, Site};
use portgraph::PortOffset;
use portmatching::{self as pm, indexing as pmx, IndexingScheme, ManyMatcher, PortMatcher};

use crate::{
    portmatching::{
        indexing::{CircuitPath, OpLocationMap, PatternOpLocation, StaticIndexScheme},
        predicate::Predicate,
    },
    static_circ::{OpLocation, StaticSizeCircuit},
};

type DiffCircuit = PortDiff<StaticSizeCircuit>;

/// A matcher object for fast pattern matching on circuits using diffed rewriting.
pub struct DiffCircuitMatcher(
    ManyMatcher<StaticSizeCircuit, PatternOpLocation, Predicate, StaticIndexScheme>,
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

impl pm::Predicate<DiffCircuit> for Predicate {
    type Value = Owned<OpLocation, StaticSizeCircuit>;

    fn check(&self, _: &DiffCircuit, args: &[impl std::borrow::Borrow<Self::Value>]) -> bool {
        match self {
            &Predicate::Link { out_port, in_port } => {
                let Owned {
                    data: out_loc,
                    owner: out_owner,
                } = args[0].borrow();
                let Owned {
                    data: in_loc,
                    owner: in_owner,
                } = args[1].borrow();
                // Make sure the location refers to the right port
                let out_loc = out_owner
                    .graph()
                    .equivalent_location(*out_loc, out_port)
                    .unwrap();
                let in_loc = in_owner
                    .graph()
                    .equivalent_location(*in_loc, in_port)
                    .unwrap();
                let op_out_locs = follow_link(
                    &Owned {
                        data: out_loc,
                        owner: out_owner.clone(),
                    },
                    Direction::Outgoing,
                    &[],
                );
                op_out_locs.contains(&Owned {
                    data: in_loc,
                    owner: in_owner.clone(),
                })
            }
            Predicate::IsOp { op } => {
                let loc = args[0].borrow();
                loc.owner.graph().get(loc.data) == Some(op)
            }
            &Predicate::NotEq { n_other } => {
                let loc = args[0].borrow();
                let op = loc.owner.graph().get_ptr(loc.data).unwrap();
                for i in 0..n_other {
                    let other_loc = args[i + 1].borrow();
                    if other_loc.owner.graph().get_ptr(other_loc.data) == Some(op) {
                        return false;
                    }
                }
                true
            }
        }
    }
}

impl pmx::IndexingScheme<DiffCircuit> for StaticIndexScheme {
    type Map = OpLocationMap<CircuitPath, Owned<OpLocation, StaticSizeCircuit>>;

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
                        .nodes_iter()
                        .map(|data| Owned {
                            data,
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
                let port = curr_circ.qubit_port(last_loc.data);
                let new_loc = curr_circ.equivalent_location(last_loc.data, port);
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
    loc: &Owned<OpLocation, StaticSizeCircuit>,
    dir: Direction,
    known_diffs: &[PortDiff<StaticSizeCircuit>],
) -> Vec<Owned<OpLocation, StaticSizeCircuit>> {
    let curr_circ = loc.owner.graph();
    let port_index = curr_circ.qubit_port(loc.data);
    let port = Port::new(dir, port_index);

    let site = Site {
        node: loc.data,
        port: port,
    };
    // All ports that correspond to site (note: in static circuits there should be at most 1)
    let mut all_ports = Vec::new();

    // Check if there is a boundary port that corresponds to site
    all_ports.extend(
        loc.owner
            .boundary_iter()
            .find(|b| loc.owner.boundary_site(*b) == &site)
            .map(pd::Port::Boundary),
    );
    // Find bound ports that corresponds to site
    all_ports.extend(curr_circ.get_bound_ports(site).map(pd::Port::Bound));

    let mut ret = Vec::new();
    // Follow link within curr_circ if there is one
    if let Some((_, linked_loc)) = curr_circ.linked_op(loc.data, port) {
        ret.push(Owned {
            owner: loc.owner.clone(),
            data: linked_loc,
        });
    }
    // Find opposite ports for all ports in all_ports
    for port in all_ports {
        ret.extend(
            loc.owner
                .opposite_ports(port)
                .into_iter()
                .filter(|p| PortDiff::are_compatible(known_diffs.iter().chain([&p.owner])))
                .map(|p| Owned {
                    data: p.site().node,
                    owner: p.owner,
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
