//! Portdiff support for StaticSizeCircuits.

mod matcher;
mod rewrite;

use std::collections::BTreeMap;

pub use matcher::{DiffCircuit, DiffCircuitMatcher};
pub use rewrite::DiffRewrite;

use derive_more::Into;
use hugr::{Direction, IncomingPort, OutgoingPort, Port, PortIndex};
use portdiff::{self as pd, port_diff::Owned, BoundPort, EdgeEnd, Site};

use crate::static_circ::{OpId, OpPosition, StaticSizeCircuit};

impl pd::Graph for StaticSizeCircuit {
    type Node = OpId;
    type Edge = EdgeId;

    type PortLabel = Port; // use port offset

    fn nodes_iter(&self) -> impl Iterator<Item = Self::Node> + '_ {
        self.ops_iter()
    }

    fn edges_iter(&self) -> impl Iterator<Item = Self::Edge> + '_ {
        self.positions_iter() // each location is a node on a qubit
            .filter_map(|pos| {
                let out_port: OutgoingPort = self.position_offset(pos).unwrap().into();
                let node = self.at_position(pos).unwrap();
                let port = out_port.into();
                EdgeId::try_from_site(Site { node, port }, self)
            })
    }

    fn get_port_site(
        &self,
        bound_port: BoundPort<Self::Edge>,
    ) -> Site<Self::Node, Self::PortLabel> {
        match bound_port.end {
            EdgeEnd::Left => Site {
                node: bound_port.edge.source(),
                port: bound_port.edge.source_port().into(),
            },
            EdgeEnd::Right => Site {
                node: bound_port.edge.target(self),
                port: bound_port.edge.target_port(self).into(),
            },
        }
    }

    fn get_bound_ports(
        &self,
        site: Site<Self::Node, Self::PortLabel>,
    ) -> impl Iterator<Item = BoundPort<Self::Edge>> + '_ {
        let Some(edge) = EdgeId::try_from_site(site, self) else {
            return None.into_iter();
        };
        let end = match site.port.direction() {
            Direction::Incoming => EdgeEnd::Right,
            Direction::Outgoing => EdgeEnd::Left,
        };
        Some(BoundPort { edge, end }).into_iter()
    }

    fn get_sites(
        &self,
        node: Self::Node,
    ) -> impl Iterator<Item = Site<Self::Node, Self::PortLabel>> + '_ {
        let op = self.get(node).unwrap();
        op.positions.iter().flat_map(move |&pos| {
            let port = self.position_offset(pos).unwrap();
            let out_port: OutgoingPort = port.into();
            let in_port: IncomingPort = port.into();
            vec![
                Site {
                    node,
                    port: out_port.into(),
                },
                Site {
                    node,
                    port: in_port.into(),
                },
            ]
        })
    }

    fn link_sites(
        &mut self,
        left: Site<Self::Node, Self::PortLabel>,
        right: Site<Self::Node, Self::PortLabel>,
    ) {
        // Make sure the sites are not already linked
        if let Some(new_right) = self.linked_op(left.node, left.port) {
            panic!("left site is already linked to {new_right:?}");
        }
        if self
            .linked_op(right.node, right.port)
            .is_some()
        {
            panic!("right site is already linked");
        }

        // Find the qubits at the sites
        let left_qubit = self
            .get_position(left.node, left.port.index())
            .expect("invalid location")
            .qubit;
        let right_qubit = self
            .get_position(right.node, right.port.index())
            .expect("invalid location")
            .qubit;

        self.merge_qubits(left_qubit, right_qubit);
    }

    fn add_subgraph(
        &mut self,
        graph: &Self,
        nodes: &std::collections::BTreeSet<Self::Node>,
    ) -> BTreeMap<Self::Node, Self::Node> {
        let node_map = self.add_subcircuit(graph, nodes);
        node_map
            .into_iter()
            .map(|(k, v)| (graph.at_position(k).unwrap(), self.at_position(v).unwrap()))
            .collect()
    }
}

/// An edge in a StaticSizeCircuit.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Into,
    serde::Serialize,
    serde::Deserialize,
)]
pub struct EdgeId {
    /// use the edge source and outport index
    source: OpId,
    out_port: OutgoingPort,
}

impl EdgeId {
    /// Create an edge from a site in the circuit.
    pub fn try_from_site(site: Site<OpId, Port>, circuit: &StaticSizeCircuit) -> Option<Self> {
        let (opp_port, opp_node) = circuit.linked_op(site.node, site.port)?;
        let (out_port, source) = match site.port.direction() {
            Direction::Incoming => (opp_port, opp_node),
            Direction::Outgoing => (site.port, site.node),
        };
        let out_port = out_port.as_outgoing().expect("checked direction above");
        Some(Self { source, out_port })
    }

    /// Get the source of the edge.
    pub fn source(&self) -> OpId {
        self.source
    }

    /// Get the source port of the edge.
    pub fn source_port(&self) -> OutgoingPort {
        self.out_port
    }

    /// Get the source position of the edge.
    ///
    /// This requires the `circuit` the edge belongs to.
    pub fn source_position(&self, circuit: &StaticSizeCircuit) -> OpPosition {
        circuit
            .get_position(self.source(), self.source_port().index())
            .expect("invalid edge ID")
    }

    /// Get the target of the edge.
    ///
    /// This requires the `circuit` the edge belongs to.
    pub fn target(&self, circuit: &StaticSizeCircuit) -> OpId {
        let (_, target) = circuit
            .linked_op(self.source(), self.source_port().into())
            .expect("invalid edge ID");
        target
    }

    /// Get the target port of the edge.
    ///
    /// This requires the `circuit` the edge belongs to.
    pub fn target_port(&self, circuit: &StaticSizeCircuit) -> IncomingPort {
        let (target_port, _) = circuit
            .linked_op(self.source(), self.source_port().into())
            .expect("invalid edge ID");
        target_port.as_incoming().expect("invalid edge ID")
    }

    /// Get the target position of the edge.
    ///
    /// This requires the `circuit` the edge belongs to.
    pub fn target_position(&self, circuit: &StaticSizeCircuit) -> OpPosition {
        circuit
            .get_position(self.target(circuit), self.target_port(circuit).index())
            .expect("invalid edge ID")
    }
}

type OwnedSite = Owned<Site<OpId, Port>, StaticSizeCircuit>;
type OwnedPort = Owned<pd::Port<StaticSizeCircuit>, StaticSizeCircuit>;

fn site_to_port(site: OwnedSite) -> Option<OwnedPort> {
    as_boundary_port(&site).or_else(|| {
        as_bound_port(Owned {
            data: site.data,
            owner: site.owner,
        })
    })
}

fn as_boundary_port(site: &OwnedSite) -> Option<OwnedPort> {
    site.owner
        .boundary_iter()
        .find(|&bd| {
            let s = site.owner.boundary_site(bd).try_as_site_ref();
            s == Some(&site.data)
        })
        .map(|bd| Owned {
            data: pd::Port::Boundary(bd),
            owner: site.owner.clone(),
        })
}

fn as_bound_port(site: OwnedSite) -> Option<OwnedPort> {
    let edge = EdgeId::try_from_site(site.data, site.owner.graph())?;
    let end = match site.data.port.direction() {
        Direction::Incoming => EdgeEnd::Right,
        Direction::Outgoing => EdgeEnd::Left,
    };
    Some(Owned {
        data: pd::Port::Bound(BoundPort { edge, end }),
        owner: site.owner,
    })
}
