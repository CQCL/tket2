use std::collections::{BTreeMap, BTreeSet};

use hugr::{Direction, Port};
use itertools::Itertools;
use portdiff::{self as pd, port_diff::Owned, Site};

use crate::{
    portmatching::indexing::{CircuitPath, OpLocationMap, PatternOpPosition},
    static_circ::{OpId, OpPosition, StaticSizeCircuit},
};

use super::{site_to_port, DiffCircuit};

type PortMap = BTreeMap<
    Owned<pd::Port<StaticSizeCircuit>, StaticSizeCircuit>,
    pd::BoundaryPort<StaticSizeCircuit>,
>;

/// A rewrite that applies on a static circuit.
#[derive(Debug, Clone)]
pub struct DiffRewrite {
    /// The nodes to be replaced.
    pub nodes: BTreeSet<Owned<OpId, StaticSizeCircuit>>,
    /// The edges to be replaced.
    pub edges: BTreeSet<(
        Owned<pd::Port<StaticSizeCircuit>, StaticSizeCircuit>,
        Owned<pd::Port<StaticSizeCircuit>, StaticSizeCircuit>,
    )>,
    /// The port map from ports of `nodes` to sites in the replacement circuit.
    pub port_map: PortMap,
    /// The replacement circuit.
    pub replacement: StaticSizeCircuit,

    // TODO: remove this
    /// The pattern circuit. (keeping it here to compute cost delta)
    pub pattern: StaticSizeCircuit,
}

impl DiffRewrite {
    /// Rewrite a subcircuit in the circuit with a replacement circuit.
    pub fn apply(self, _: &DiffCircuit) -> Result<DiffCircuit, pd::InvalidRewriteError> {
        pd::PortDiff::rewrite(
            self.nodes.iter().cloned(),
            self.edges.iter().cloned(),
            self.replacement,
            move |p| self.port_map.get(&p).unwrap().clone(),
        )
    }

    /// Create a DiffRewrite from a pattern match.
    pub fn try_from_pattern_match(
        match_map: &OpLocationMap<CircuitPath, Owned<OpPosition, StaticSizeCircuit>>,
        pattern: &StaticSizeCircuit,
        replacement: StaticSizeCircuit,
    ) -> Result<Self, ()> {
        let starts = pattern.find_qubit_starts().unwrap();

        // Fill nodes
        let mut nodes = BTreeSet::new();
        for pos in pattern.positions_iter() {
            let ploc = PatternOpPosition::from_position(pos, &starts);
            let Owned { data, owner } = match_map
                .get_val(&ploc.qubit, ploc.op_idx as isize)
                .unwrap()
                .clone();
            nodes.insert(Owned {
                data: owner.graph().at_position(data).unwrap(),
                owner: owner.clone(),
            });
        }

        let pos_to_port = |pos: OpPosition, dir: Direction| {
            if !pattern.exists(pos) {
                return None;
            }
            let ploc = PatternOpPosition::from_position(pos, &starts);
            let offset = pattern.position_offset(pos).unwrap();
            if let Some(Owned { data, owner }) = match_map
                .get_val(&ploc.qubit, ploc.op_idx as isize)
                .cloned()
            {
                let site = Site {
                    node: owner.graph().at_position(data).unwrap(),
                    port: Port::new(dir, offset),
                };
                site_to_port(Owned {
                    data: site,
                    owner: owner,
                })
            } else {
                None
            }
        };

        // Fill edges
        let mut edges = BTreeSet::new();
        for pos in pattern.positions_iter() {
            if pos.index > 0 {
                let src_pos = pos.try_add_op_idx(-1).unwrap();
                let src = pos_to_port(src_pos, Direction::Outgoing).unwrap();
                let tgt = pos_to_port(pos, Direction::Incoming).unwrap();
                edges.insert((src, tgt));
            }
        }

        // Fill port map
        let mut port_map = PortMap::new();
        for qubit in pattern.qubits_iter() {
            assert!(
                pattern.qubit_ops(qubit).len() > 0,
                "found empty qubit in pattern"
            );
            let start = OpPosition { qubit, index: 0 };
            let end = OpPosition {
                qubit,
                index: pattern.qubit_ops(qubit).len() - 1,
            };
            let start_repl = OpPosition { qubit, index: 0 };
            if replacement.qubit_ops(qubit).len() > 0 {
                let start_port_repl = replacement.position_offset(start_repl.into()).unwrap();
                let end_repl = OpPosition {
                    qubit,
                    index: replacement.qubit_ops(qubit).len() - 1,
                };
                if let Some(start_port) = pos_to_port(start, Direction::Incoming) {
                    port_map.insert(
                        start_port,
                        pd::BoundaryPort::Site(Site {
                            node: replacement.at_position(start_repl).unwrap(),
                            port: Port::new(Direction::Incoming, start_port_repl),
                        }),
                    );
                }
                if let Some(end_port) = pos_to_port(end, Direction::Outgoing) {
                    let end_port_repl = replacement.position_offset(end_repl.into()).unwrap();
                    port_map.insert(
                        end_port,
                        pd::BoundaryPort::Site(Site {
                            node: replacement.at_position(end_repl).unwrap(),
                            port: Port::new(Direction::Outgoing, end_port_repl),
                        }),
                    );
                }
            } else {
                // if let Some(start_port) = pos_to_port(start, Direction::Incoming) {
                //     port_map.insert(start_port, pd::BoundaryPort::Sentinel(qubit.0));
                // }
                // if let Some(end_port) = pos_to_port(end, Direction::Outgoing) {
                //     port_map.insert(end_port, pd::BoundaryPort::Sentinel(qubit.0));
                // }
                return Err(());
            }
        }

        Ok(Self {
            nodes,
            edges,
            pattern: pattern.clone(),
            port_map,
            replacement,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use hugr::OutgoingPort;
    use portdiff::{self as pd, BoundPort, EdgeEnd, GraphView, PortDiff};

    use crate::{portdiff::EdgeId, Tk2Op};

    use super::*;

    #[test]
    fn test_apply_rewrite() {
        let mut circ = StaticSizeCircuit::with_qubit_count(2);
        circ.append_op(Tk2Op::CX, vec![0.into(), 1.into()]);
        let owner = DiffCircuit::from_graph(circ.clone());
        let pos = OpPosition {
            qubit: 0.into(),
            index: 0,
        };
        let rw = DiffRewrite {
            nodes: vec![Owned {
                data: owner.graph().at_position(pos).unwrap(),
                owner: owner.clone(),
            }]
            .into_iter()
            .collect(),
            edges: vec![].into_iter().collect(),
            port_map: BTreeMap::new(),
            replacement: circ.clone(),
            pattern: circ.clone(),
        };
        let new_circ = rw.apply(&owner).unwrap();
        PortDiff::extract_graph(vec![new_circ.clone()]).unwrap();

        let g = GraphView::from_sinks(vec![new_circ]);
        let ser = serde_json::to_string(&g).unwrap();
        let g: GraphView<StaticSizeCircuit> = serde_json::from_str(&ser).unwrap();
        let new_circ = g.sinks().next().unwrap();
        PortDiff::extract_graph(vec![new_circ.clone()]).unwrap();
    }

    #[test]
    fn test_apply_rewrite_2() {
        let mut circ = StaticSizeCircuit::with_qubit_count(2);
        let first_cx_id = circ.append_op(Tk2Op::CX, vec![0.into(), 1.into()]);
        circ.append_op(Tk2Op::CX, vec![0.into(), 1.into()]);
        circ.append_op(Tk2Op::CX, vec![0.into(), 1.into()]);
        let owner = DiffCircuit::from_graph(circ.clone());
        let pos = OpPosition {
            qubit: 0.into(),
            index: 0,
        };
        let op = |p: usize| Port::new(Direction::Outgoing, p);
        let mut pattern = StaticSizeCircuit::with_qubit_count(2);
        pattern.append_op(Tk2Op::CX, vec![0.into(), 1.into()]);
        let replacement = pattern.clone();
        let port_map = BTreeMap::from_iter([
            (
                Owned {
                    data: pd::Port::Bound(BoundPort {
                        edge: EdgeId::try_from_site(
                            Site {
                                node: first_cx_id,
                                port: op(0),
                            },
                            owner.graph(),
                        )
                        .unwrap(),
                        end: EdgeEnd::Left,
                    }),
                    owner: owner.clone(),
                },
                Site {
                    node: replacement.at_position(pos).unwrap(),
                    port: op(0),
                }
                .into(),
            ),
            (
                Owned {
                    data: pd::Port::Bound(BoundPort {
                        edge: EdgeId::try_from_site(
                            Site {
                                node: first_cx_id,
                                port: op(1),
                            },
                            owner.graph(),
                        )
                        .unwrap(),
                        end: EdgeEnd::Left,
                    }),
                    owner: owner.clone(),
                },
                Site {
                    node: replacement.at_position(pos).unwrap(),
                    port: op(1),
                }
                .into(),
            ),
        ]);
        let rw = DiffRewrite {
            nodes: vec![Owned {
                data: owner.graph().at_position(pos).unwrap(),
                owner: owner.clone(),
            }]
            .into_iter()
            .collect(),
            edges: vec![].into_iter().collect(),
            port_map,
            replacement,
            pattern,
        };
        let new_circ = rw.apply(&owner).unwrap();
        let out_circ = PortDiff::extract_graph(vec![new_circ.clone()]).unwrap();
        assert_eq!(out_circ, circ);

        let g = GraphView::from_sinks(vec![new_circ]);
        let ser = serde_json::to_string_pretty(&g).unwrap();
        let g: GraphView<StaticSizeCircuit> = serde_json::from_str(&ser).unwrap();
        let new_circ = g.sinks().next().unwrap();
        PortDiff::extract_graph(vec![new_circ.clone()]).unwrap();
    }
}
