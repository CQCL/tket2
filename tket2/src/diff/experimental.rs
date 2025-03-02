//! Experimental implementation of HugrView for [`CircuitHistory`].
//!
//! Access the implementation using the [`ExperimentalHugrWrapper`] type.
//!
//! ## Limitations
//!  - Panics on histories with more than 2^16 = 65 536 diffs or which contain
//!    a diff with more than 2^16 - 1 nodes.
//!  - Does not implement [`HugrInternals::portgraph`] or
//!    [`HugrInternals::base_hugr`], as these are not well defined for the
//!    history as a whole.
//!  - [`HugrView::nodes`] and [`HugrView::node_count`] are inefficient: they
//!    iterate over the entire hugr. Currently, no implementation of
//!    [`HugrView::edge_count`] is provided.
//!
//! Better support for this would require modifications to the [`HugrView`]
//! and [`HugrInternals`] traits.

use std::collections::BTreeSet;

use derive_more::{From, Into};
use derive_where::derive_where;
use hugr::{
    hugr::views::ExtractHugr,
    ops::{OpType, DEFAULT_OPTYPE},
    Direction, Hugr, HugrView, Node, NodeIndex, Port,
};
use hugr_core::hugr::internal::HugrInternals;
use itertools::Itertools;

use crate::{
    diff::{CircuitDiff, Owned},
    Circuit,
};

use super::CircuitHistory;

/// A wrapper around a [`CircuitHistory`] which implements [`HugrView`].
///
/// This is experimental and has significant limitations.
///
/// ## Limitations
///  - Panics on histories with more than 2^16 = 65 536 diffs or which contain
///    a diff with more than 2^16 - 1 nodes.
///  - Does not implement [`HugrInternals::portgraph`] or
///    [`HugrInternals::base_hugr`], as these are not well defined for the
///    history as a whole.
///  - [`HugrView::nodes`] and [`HugrView::node_count`] are inefficient: they
///    iterate over the entire hugr. Currently, no implementation of
///    [`HugrView::edge_count`] is provided.
#[derive(Clone, From, Into)]
pub struct ExperimentalHugrWrapper<H: HugrView>(pub CircuitHistory<H>);

impl<H: HugrView<Node = Node>> ExperimentalHugrWrapper<H> {
    /// View the history as a circuit
    pub fn as_circuit(&self) -> Circuit<&ExperimentalHugrWrapper<H>> {
        Circuit::new(self, self.root())
    }

    /// Get the underlying hugr of a diff
    fn get_diff_hugr(&self, diff_index: usize) -> &H {
        // Get the diff_index-th element from all_nodes()
        let &diff_id = self
            .0
            .diffs
            .all_nodes()
            .iter()
            .nth(diff_index)
            .expect("invalid diff index");
        let diff = self.0.diffs.get_node(diff_id);
        diff.value().circuit.circuit().hugr()
    }

    fn get_diff(&self, diff_index: usize) -> CircuitDiff<H> {
        // Get the diff_index-th element from all_nodes()
        let &diff_id = self
            .0
            .diffs
            .all_nodes()
            .iter()
            .nth(diff_index)
            .expect("invalid diff index");
        CircuitDiff(self.0.diffs.get_node_rc(diff_id))
    }

    fn get_diff_index(&self, diff: &CircuitDiff<H>) -> Option<usize> {
        self.0
            .diffs
            .all_nodes()
            .iter()
            .position(|id| self.0.diffs.get_node_rc(*id).ptr_eq(&diff.0))
    }
}

impl<H: HugrView<Node = Node>> HugrView for ExperimentalHugrWrapper<H> {
    fn contains_node(&self, node: Node) -> bool {
        let node: CircuitHistoryNode = node.into();
        // Get the diff_index-th element from all_nodes()
        let diff = node.diff(self);
        if !self.0.is_root(&diff) && diff.io_nodes().contains(&node.node()) {
            // Non-root IO nodes are not part of the hugr
            return false;
        }
        diff.as_hugr().contains_node(node.node())
    }

    fn node_count(&self) -> usize {
        self.nodes().count()
    }

    fn edge_count(&self) -> usize {
        unimplemented!()
    }

    fn nodes(&self) -> impl Iterator<Item = Node> + Clone {
        let current = self.get_io(self.root()).unwrap().to_vec();
        let children = NodesIter {
            visited: BTreeSet::default(),
            current,
            history: self,
        };
        [self.root_node()].into_iter().chain(children)
    }

    fn node_ports(&self, node: Node, dir: Direction) -> impl Iterator<Item = Port> + Clone {
        let node: CircuitHistoryNode = node.into();
        let hugr = self.get_diff_hugr(node.diff_index as usize);
        hugr.node_ports(node.node(), dir)
    }

    fn all_node_ports(&self, node: Node) -> impl Iterator<Item = Port> + Clone {
        let node: CircuitHistoryNode = node.into();
        let hugr = self.get_diff_hugr(node.diff_index as usize);
        hugr.all_node_ports(node.node())
    }

    fn linked_ports(
        &self,
        node: Node,
        port: impl Into<Port>,
    ) -> impl Iterator<Item = (Node, Port)> + Clone {
        let node: CircuitHistoryNode = node.into();
        let node_port = Owned {
            owner: node.diff(self),
            data: (node.node(), port.into()),
        };
        let into_node = |node_port: Owned<H, (Node, Port)>| {
            let Owned { owner, data } = node_port;
            let diff_index = self.get_diff_index(&owner).unwrap();
            let node = CircuitHistoryNode::try_new(diff_index, data.0)
                .expect("diff_index or node_index too large for CircuitHistoryNode");
            (node.into(), data.1)
        };
        self.0.linked_ports(node_port).map(into_node)
    }

    fn node_connections(&self, node: Node, other: Node) -> impl Iterator<Item = [Port; 2]> + Clone {
        let ports = self
            .node_inputs(node)
            .map_into()
            .chain(self.node_outputs(node).map_into());
        ports.flat_map(move |p| {
            self.linked_ports(node, p)
                .filter(move |(n, _)| n == &other)
                .map(move |(_, other_p)| [p, other_p])
        })
    }

    fn num_ports(&self, node: Node, dir: Direction) -> usize {
        let node: CircuitHistoryNode = node.into();
        let hugr = self.get_diff_hugr(node.diff_index as usize);
        hugr.num_ports(node.node(), dir)
    }

    fn children(&self, node: Node) -> impl DoubleEndedIterator<Item = Node> + Clone {
        let node: CircuitHistoryNode = node.into();
        let hugr = self.get_diff_hugr(node.diff_index as usize);
        let to_owned = move |n| {
            CircuitHistoryNode::try_new(node.diff_index as usize, n)
                .unwrap()
                .into()
        };
        hugr.children(node.node()).map(to_owned)
    }

    fn neighbours(&self, node: Node, dir: Direction) -> impl Iterator<Item = Node> + Clone {
        self.node_ports(node, dir)
            .flat_map(move |p| self.linked_ports(node, p))
            .map(|(n, _)| n)
    }

    fn all_neighbours(&self, node: Node) -> impl Iterator<Item = Node> + Clone {
        self.neighbours(node, Direction::Incoming)
            .chain(self.neighbours(node, Direction::Outgoing))
    }

    /// Returns the operation type of a node.
    fn get_optype(&self, node: Node) -> &OpType {
        match self.contains_node(node) {
            true => {
                let node: CircuitHistoryNode = node.into();
                // Get the diff_index-th element from all_nodes()
                let hugr = self.get_diff_hugr(node.diff_index as usize);
                hugr.get_optype(node.node())
            }
            false => &DEFAULT_OPTYPE,
        }
    }
}

/// Iterator over all nodes in the graph, using a simple depth-first search
#[derive_where(Clone)]
struct NodesIter<'h, H: HugrView> {
    visited: BTreeSet<H::Node>,
    current: Vec<H::Node>,
    history: &'h ExperimentalHugrWrapper<H>,
}

impl<'h, H: HugrView<Node = Node>> Iterator for NodesIter<'h, H> {
    type Item = Node;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(node) = self.current.pop() {
            debug_assert!(self.history.contains_node(node));
            if self.visited.insert(node) {
                // Add all neighbours to current
                self.current.extend(
                    self.history
                        .all_neighbours(node)
                        .filter(|n| !self.visited.contains(n)),
                );
                return Some(node);
            }
        }
        None
    }
}

impl<H: HugrView<Node = Node>> HugrInternals for ExperimentalHugrWrapper<H> {
    type Portgraph<'p> = H::Portgraph<'p> where Self: 'p;
    type Node = Node;

    fn portgraph(&self) -> Self::Portgraph<'_> {
        unimplemented!("no single portgraph for history")
    }

    fn base_hugr(&self) -> &hugr::Hugr {
        unimplemented!("no single base hugr for history")
    }

    fn root_node(&self) -> hugr::Node {
        let diff_index = self.get_diff_index(&self.0.root).unwrap();
        let node = CircuitHistoryNode::try_new(diff_index, self.0.root.as_hugr().root_node())
            .expect("diff_index or node_index too large for CircuitHistoryNode");
        node.into()
    }

    fn to_pg_index(&self, node: Self::Node) -> portgraph::NodeIndex {
        portgraph::NodeIndex::new(node.index())
    }

    fn to_node(&self, index: portgraph::NodeIndex) -> Self::Node {
        index.into()
    }
}

impl<H: HugrView<Node = Node>> ExtractHugr for ExperimentalHugrWrapper<H> {
    fn extract_hugr(self) -> Hugr {
        self.0.extract_hugr()
    }
}

/// A node in the history, that can disguise as a [`Node`]
///
/// This is an ugly hack, where we encode the two integers of information
/// (the diff index in the history and the node index) into a single
/// [Node] by using the lower and upper 16 bits of the index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct CircuitHistoryNode {
    /// The diff index this node belongs to
    diff_index: u16,
    /// The node index within the diff
    node_index: u16,
}

impl CircuitHistoryNode {
    fn try_new(diff_index: usize, node: Node) -> Option<Self> {
        let diff_index = u16::try_from(diff_index).ok()?;
        let node_index = u16::try_from(node.index()).ok()?;

        Some(Self {
            diff_index,
            node_index,
        })
    }

    fn node(&self) -> Node {
        Node::from(portgraph::NodeIndex::new(self.node_index as usize))
    }

    fn diff<H: HugrView<Node = Node>>(
        &self,
        history: &ExperimentalHugrWrapper<H>,
    ) -> CircuitDiff<H> {
        history.get_diff(self.diff_index as usize)
    }
}

impl From<Node> for CircuitHistoryNode {
    fn from(node: Node) -> Self {
        let index = node.index();
        // The node index is the lower 16 bits of the node index
        let node_index = (index & 0xFFFF) as u16;
        // The diff index is the upper 16 bits of the node index
        let diff_index = (index >> 16) as u16;
        Self {
            diff_index,
            node_index,
        }
    }
}

impl From<CircuitHistoryNode> for Node {
    fn from(node: CircuitHistoryNode) -> Self {
        let mut index = node.node_index as usize;
        index |= (node.diff_index as usize) << 16;
        Node::from(portgraph::NodeIndex::new(index))
    }
}
