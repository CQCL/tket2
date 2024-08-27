//! Portdiff support for StaticSizeCircuits.

mod matcher;

pub use matcher::DiffCircuitMatcher;

use hugr::{IncomingPort, OutgoingPort, Port, PortIndex};
use itertools::{Either, Itertools};
use portdiff::{self as pd, BoundPort, EdgeEnd, Site};

use crate::static_circ::{OpLocation, StaticSizeCircuit};

impl pd::Graph for StaticSizeCircuit {
    type Node = OpLocation; // use the location at the 0-th qubit

    type Edge = (OpLocation, OutgoingPort); // use the edge source and outport index

    type PortLabel = Port; // use port offset

    fn nodes_iter(&self) -> impl Iterator<Item = Self::Node> + '_ {
        self.all_locations() // each location is a node on a qubit
            .map(|loc| self.equivalent_location(loc, 0).unwrap()) // get location at port 0
            .unique()
    }

    fn edges_iter(&self) -> impl Iterator<Item = Self::Edge> + '_ {
        self.all_locations() // each location is a node on a qubit
            .map(|loc| (loc, self.qubit_port(loc).into()))
            // filter out port locations with no successor op
            .filter(|&(loc, port): &(_, OutgoingPort)| self.linked_op(loc, port.into()).is_some())
    }

    fn get_port_site(
        &self,
        bound_port: BoundPort<Self::Edge>,
    ) -> Site<Self::Node, Self::PortLabel> {
        match bound_port.end {
            EdgeEnd::Left => {
                let (loc, port) = bound_port.edge;
                Site {
                    node: loc,
                    port: port.into(),
                }
            }
            EdgeEnd::Right => {
                let (left_loc, left_port) = bound_port.edge;
                let (right_port, right_loc) = self
                    .linked_op(left_loc, left_port.into())
                    .expect("invalid bound port");
                Site {
                    node: right_loc,
                    port: right_port,
                }
            }
        }
    }

    fn get_bound_ports(
        &self,
        site: Site<Self::Node, Self::PortLabel>,
    ) -> impl Iterator<Item = BoundPort<Self::Edge>> + '_ {
        let Site { node: loc, port } = site;
        let bound_port = match port.as_directed() {
            Either::Right(port) => Some(BoundPort {
                edge: (loc, port),
                end: EdgeEnd::Left,
            }),
            Either::Left(port) => {
                if let Some((port, loc)) = self.linked_op(loc, port.into()) {
                    let port = port.as_outgoing().expect("two incoming ports are linked");
                    Some(BoundPort {
                        edge: (loc, port),
                        end: EdgeEnd::Right,
                    })
                } else {
                    None
                }
            }
        };
        bound_port.into_iter()
    }

    fn get_sites(
        &self,
        node: Self::Node,
    ) -> impl Iterator<Item = Site<Self::Node, Self::PortLabel>> + '_ {
        let op = self.get_rc(node).unwrap();
        self.op_locations(op).iter().flat_map(|&loc| {
            let port = self.qubit_port(loc);
            let out_port: OutgoingPort = port.into();
            let in_port: IncomingPort = port.into();
            vec![
                Site {
                    node: loc,
                    port: out_port.into(),
                },
                Site {
                    node: loc,
                    port: in_port.into(),
                },
            ]
        })
    }

    fn link_sites(
        &mut self,
        left: Site<Self::Node, Self::PortLabel>,
        right: Site<Self::Node, Self::PortLabel>,
    ) -> (BoundPort<Self::Edge>, BoundPort<Self::Edge>) {
        let Site {
            node: left_loc,
            port: left_port,
        } = left;
        let Site {
            node: right_loc,
            port: right_port,
        } = right;

        // Make sure the sites are not already linked
        if self.linked_op(left_loc, left_port).is_some() {
            panic!("left site is already linked");
        }
        if self.linked_op(right_loc, right_port).is_some() {
            panic!("right site is already linked");
        }
        let left_port = left_port.as_outgoing().unwrap();

        // Get the qubits of the sites
        let left_qubit = self
            .equivalent_location(left_loc, left_port.index())
            .unwrap()
            .qubit;
        let right_qubit = self
            .equivalent_location(right_loc, right_port.index())
            .unwrap()
            .qubit;

        // Merge the qubits
        let new_qubit = self.merge_qubits(left_qubit, right_qubit);

        // Find new location and create edge ID
        let new_loc = self
            .equivalent_location(
                OpLocation {
                    qubit: new_qubit,
                    op_idx: left_port.index(),
                },
                0,
            )
            .unwrap();
        let edge = (new_loc, left_port.into());

        (
            BoundPort {
                edge,
                end: EdgeEnd::Left,
            },
            BoundPort {
                edge,
                end: EdgeEnd::Right,
            },
        )
    }

    fn add_subgraph(
        &mut self,
        graph: &Self,
        nodes: &std::collections::BTreeSet<Self::Node>,
    ) -> std::collections::BTreeMap<Self::Node, Self::Node> {
        let nodes = nodes
            .iter()
            .map(|&loc| graph.get_ptr(loc).expect("invalid op location"))
            .collect();
        self.add_subcircuit(graph, &nodes)
    }
}
