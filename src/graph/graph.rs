#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
use std::collections::HashMap;
use std::fmt::{self, Debug};
use std::hash::Hash;
use std::mem::replace;
use std::{iter, slice};

/// Trait for the unsigned integer type used for node and edge indices.
///
/// Marked `unsafe` because: the trait must faithfully preserve
pub trait IndexType: Copy + Default + Hash + Ord + fmt::Debug + 'static {
    fn new(x: usize) -> Self;
    fn index(&self) -> usize;
    fn max() -> Self;
}

/// The default integer type for graph indices.
/// `u32` is the default to reduce the size of the graph's data and improve
/// performance in the common case.
///
/// Used for node and edge indices in `Graph` and `StableGraph`, used
/// for node indices in `Csr`.
pub type DefaultIx = u32;
/// and convert index values.
// unsafe impl IndexType for usize {
//     #[inline(always)]
//     fn new(x: usize) -> Self {
//         x
//     }
//     #[inline(always)]
//     fn index(&self) -> Self {
//         *self
//     }
//     #[inline(always)]
//     fn max() -> Self {
//         ::std::usize::MAX
//     }
// }

impl IndexType for u32 {
    #[inline(always)]
    fn new(x: usize) -> Self {
        x as u32
    }
    #[inline(always)]
    fn index(&self) -> usize {
        *self as usize
    }
    #[inline(always)]
    fn max() -> Self {
        ::std::u32::MAX
    }
}

// unsafe impl IndexType for u16 {
//     #[inline(always)]
//     fn new(x: usize) -> Self {
//         x as u16
//     }
//     #[inline(always)]
//     fn index(&self) -> usize {
//         *self as usize
//     }
//     #[inline(always)]
//     fn max() -> Self {
//         ::std::u16::MAX
//     }
// }

impl IndexType for u8 {
    #[inline(always)]
    fn new(x: usize) -> Self {
        x as u8
    }
    #[inline(always)]
    fn index(&self) -> usize {
        *self as usize
    }
    #[inline(always)]
    fn max() -> Self {
        ::std::u8::MAX
    }
}

/// Node identifier.
// #[cfg_attr(feature = "pyo3", pyclass)]
#[cfg_attr(feature = "pyo3", derive(FromPyObject))]
#[derive(Copy, Clone, Default, PartialEq, PartialOrd, Eq, Ord, Hash, Debug)]
pub struct NodeIndex(DefaultIx);

impl NodeIndex {
    #[inline]
    pub fn new(x: usize) -> Self {
        NodeIndex(IndexType::new(x))
    }

    #[inline]
    pub fn index(self) -> usize {
        self.0.index()
    }

    #[inline]
    pub fn end() -> Self {
        NodeIndex(IndexType::max())
    }

    fn _into_edge(self) -> EdgeIndex {
        EdgeIndex(self.0)
    }
}

impl IndexType for NodeIndex {
    fn index(&self) -> usize {
        self.0.index()
    }
    fn new(x: usize) -> Self {
        NodeIndex::new(x)
    }
    fn max() -> Self {
        NodeIndex(<DefaultIx as IndexType>::max())
    }
}

impl From<DefaultIx> for NodeIndex {
    fn from(ix: DefaultIx) -> Self {
        NodeIndex(ix)
    }
}
/// Edge identifier.
// #[cfg_attr(feature = "pyo3", pyclass)]
#[cfg_attr(feature = "pyo3", derive(FromPyObject))]
#[derive(Copy, Clone, Default, PartialEq, PartialOrd, Eq, Ord, Hash, Debug)]
pub struct EdgeIndex(DefaultIx);

impl EdgeIndex {
    #[inline]
    pub fn new(x: usize) -> Self {
        EdgeIndex(IndexType::new(x))
    }

    #[inline]
    pub fn index(self) -> usize {
        self.0.index()
    }

    /// An invalid `EdgeIndex` used to denote absence of an edge, for example
    /// to end an adjacency list.
    #[inline]
    pub fn end() -> Self {
        EdgeIndex(IndexType::max())
    }

    fn _into_node(self) -> NodeIndex {
        NodeIndex(self.0)
    }
}

impl From<DefaultIx> for EdgeIndex {
    fn from(ix: DefaultIx) -> Self {
        EdgeIndex(ix)
    }
}

#[cfg_attr(feature = "pyo3", derive(FromPyObject))]
#[derive(Copy, Clone, Default, PartialEq, PartialOrd, Eq, Ord, Hash, Debug)]
pub struct PortIndex(u8);

impl PortIndex {
    #[inline]
    pub fn new(x: usize) -> Self {
        PortIndex(IndexType::new(x))
    }

    #[inline]
    pub fn index(&self) -> usize {
        self.0.index()
    }

    #[inline]
    pub fn end() -> Self {
        PortIndex(IndexType::max())
    }
}

#[cfg_attr(feature = "pyo3", pyclass)]
#[derive(Copy, Clone, Default, PartialEq, PartialOrd, Eq, Ord, Hash, Debug)]
pub struct NodePort {
    pub node: NodeIndex,
    pub port: PortIndex,
}

impl NodePort {
    pub fn new(node: NodeIndex, port: PortIndex) -> Self {
        Self { node, port }
    }
}

impl<Px: Into<u8>> From<(NodeIndex, Px)> for NodePort {
    fn from((node, p): (NodeIndex, Px)) -> Self {
        Self {
            node,
            port: PortIndex(p.into()),
        }
    }
}

/// The graph's node type.
#[derive(Debug)]
pub(crate) struct Node<N> {
    /// Associated node data.
    pub weight: Option<N>,

    incoming: Vec<EdgeIndex>,
    outgoing: Vec<EdgeIndex>,
}

impl<N: Clone> Clone for Node<N> {
    fn clone(&self) -> Self {
        Self {
            weight: self.weight.clone(),
            incoming: self.incoming.clone(),
            outgoing: self.outgoing.clone(),
        }
    }
}

impl<N: PartialEq> PartialEq for &Node<N> {
    fn eq(&self, other: &Self) -> bool {
        self.weight == other.weight
            && self.incoming == other.incoming
            && self.outgoing == other.outgoing
    }
}

/// The graph's edge type.
#[derive(Debug)]
pub(super) struct Edge<E> {
    /// Associated edge data.
    pub weight: Option<E>,
    // / Next edge in outgoing and incoming edge lists.
    // next: [EdgeIndex; 2],
    /// Start and End node index
    pub(super) node_ports: [NodePort; 2],
}

impl<E: Clone> Clone for Edge<E> {
    fn clone(&self) -> Self {
        Self {
            weight: self.weight.clone(),
            node_ports: self.node_ports,
        }
    }
}

impl<E: PartialEq> PartialEq for &Edge<E> {
    fn eq(&self, other: &Self) -> bool {
        self.weight == other.weight && self.node_ports == other.node_ports
    }
}

#[cfg_attr(feature = "pyo3", pyclass)]
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub enum Direction {
    Incoming = 0,
    Outgoing = 1,
}

impl Default for Direction {
    fn default() -> Self {
        Direction::Incoming
    }
}

type NodeMap = HashMap<NodeIndex, NodeIndex>;
type EdgeMap = HashMap<EdgeIndex, EdgeIndex>;

pub struct Graph<N, E> {
    pub(crate) nodes: Vec<Node<N>>,
    pub(super) edges: Vec<Edge<E>>,
    node_count: usize,
    edge_count: usize,

    // node and edge free lists (both work the same way)
    //
    // free_node, if not NodeIndex::end(), points to a node index
    // that is vacant (after a deletion).
    // The free nodes form a doubly linked list using the fields Node.next[0]
    // for forward references and Node.next[1] for backwards ones.
    // The nodes are stored as EdgeIndex, and the _into_edge()/_into_node()
    // methods convert.
    // free_edge, if not EdgeIndex::end(), points to a free edge.
    // The edges only form a singly linked list using Edge.next[0] to store
    // the forward reference.
    free_node: NodeIndex,
    free_edge: EdgeIndex,
}

impl<N: Clone, E: Clone> Clone for Graph<N, E> {
    fn clone(&self) -> Self {
        Self {
            nodes: self.nodes.clone(),
            edges: self.edges.clone(),
            node_count: self.node_count,
            edge_count: self.edge_count,
            free_node: self.free_node,
            free_edge: self.free_edge,
        }
    }
}

impl<N: Debug, E: Debug> Debug for Graph<N, E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Graph")
            .field("nodes", &self.nodes)
            .field("edges", &self.edges)
            .field("node_count", &self.node_count)
            .field("edge_count", &self.edge_count)
            .field("free_node", &self.free_node)
            .field("free_edge", &self.free_edge)
            .finish()
    }
}

impl<E: PartialEq, N: PartialEq> PartialEq for Graph<N, E> {
    fn eq(&self, other: &Self) -> bool {
        self.nodes.iter().eq(other.nodes.iter())
            && self.edges.iter().eq(other.edges.iter())
            && self.node_count == other.node_count
            && self.edge_count == other.edge_count
            && self.free_node == other.free_node
            && self.free_edge == other.free_edge
    }
}

impl<N, E> Default for Graph<N, E> {
    fn default() -> Self {
        Self::new()
    }
}

impl<N, E> Graph<N, E> {
    pub fn new() -> Self {
        Self::with_capacity(0, 0)
    }

    pub fn with_capacity(nodes: usize, edges: usize) -> Self {
        Self {
            nodes: Vec::with_capacity(nodes),
            edges: Vec::with_capacity(edges),
            node_count: 0,
            edge_count: 0,
            free_node: NodeIndex::end(),
            free_edge: EdgeIndex::end(),
        }
    }

    pub fn add_node_with_capacity(&mut self, capacity: usize, weight: N) -> NodeIndex {
        if self.free_node != NodeIndex::end() {
            let node_idx = self.free_node;
            let node_slot = &mut self.nodes[node_idx.index()];

            let _old = replace(&mut node_slot.weight, Some(weight));
            debug_assert!(_old.is_none());

            let previous_node = node_slot.incoming[0];
            let next_node = node_slot.outgoing[0];

            node_slot.incoming = Vec::with_capacity(capacity);
            node_slot.outgoing = Vec::with_capacity(capacity);

            if previous_node != EdgeIndex::end() {
                self.nodes[previous_node.index()].outgoing[0] = next_node;
            }
            if next_node != EdgeIndex::end() {
                self.nodes[next_node.index()].incoming[0] = previous_node;
            }
            self.free_node = next_node._into_node();
            self.node_count += 1;
            node_idx
        } else {
            let node = Node {
                weight: Some(weight),
                incoming: Vec::with_capacity(capacity),
                outgoing: Vec::with_capacity(capacity),
            };
            let node_idx = NodeIndex::new(self.node_count);
            self.node_count += 1;
            self.nodes.push(node);
            node_idx
        }
    }

    pub fn add_node(&mut self, weight: N) -> NodeIndex {
        self.add_node_with_capacity(0, weight)
    }

    pub fn add_edge<T: Into<NodePort>, S: Into<NodePort>>(
        &mut self,
        a: T,
        b: S,
        weight: E,
    ) -> EdgeIndex {
        let a = a.into();
        let b = b.into();
        let edge_idx;

        if self.free_edge != EdgeIndex::end() {
            edge_idx = self.free_edge;
            let edge = &mut self.edges[edge_idx.index()];
            let _old = replace(&mut edge.weight, Some(weight));
            debug_assert!(_old.is_none());
            self.free_edge = edge.node_ports[0].node._into_edge();
            edge.node_ports = [a, b];
        } else {
            edge_idx = EdgeIndex::new(self.edges.len());
            debug_assert!(
                <DefaultIx as IndexType>::max().index() == !0 || EdgeIndex::end() != edge_idx
            );
            let new_edge = Edge {
                weight: Some(weight),
                node_ports: [a, b],
            };
            self.edges.push(new_edge);
        }

        let wire_up_port = |port: PortIndex, port_array: &mut Vec<_>| {
            let port = port.index();
            if let Some(e) = port_array.get_mut(port) {
                *e = edge_idx;
            } else {
                // TODO figure out if this poses any danger
                while port > port_array.len() {
                    port_array.push(EdgeIndex::end());
                }
                port_array.push(edge_idx);
            }
        };

        let node_a = self.nodes.get_mut(a.node.index()).expect("Node not found.");
        wire_up_port(a.port, &mut node_a.outgoing);

        let node_b = self.nodes.get_mut(b.node.index()).expect("Node not found.");
        wire_up_port(b.port, &mut node_b.incoming);

        self.edge_count += 1;
        edge_idx
    }

    pub(crate) fn update_edge<T: Into<NodePort>, S: Into<NodePort>>(
        &mut self,
        e: EdgeIndex,
        a: T,
        b: S,
    ) {
        let a = a.into();
        let b = b.into();

        let wire_up_port = |port: PortIndex, port_array: &mut Vec<_>| {
            let port = port.index();
            if let Some(port_e) = port_array.get_mut(port) {
                *port_e = e;
            } else {
                debug_assert!(port == port_array.len());
                port_array.push(e);
            }
        };

        let edge = &mut self.edges[e.index()];

        let old_source = self
            .nodes
            .get_mut(edge.node_ports[0].node.index())
            .expect("Node not found.");

        old_source.outgoing[edge.node_ports[0].port.index()] = EdgeIndex::end();

        let old_target = self
            .nodes
            .get_mut(edge.node_ports[1].node.index())
            .expect("Node not found.");
        old_target.incoming[edge.node_ports[1].port.index()] = EdgeIndex::end();

        let node_a = self.nodes.get_mut(a.node.index()).expect("Node not found.");
        wire_up_port(a.port, &mut node_a.outgoing);

        let node_b = self.nodes.get_mut(b.node.index()).expect("Node not found.");
        wire_up_port(b.port, &mut node_b.incoming);

        edge.node_ports = [a, b];
    }

    pub fn remove_node(&mut self, a: NodeIndex) -> Option<N> {
        let node_weight = self.nodes.get_mut(a.index())?.weight.take()?;

        let node = &self.nodes[a.index()];
        let remove_edges: Vec<_> = node
            .incoming
            .iter()
            .chain(&node.outgoing)
            .copied()
            .filter(|e| *e != EdgeIndex::end())
            .collect();
        // for port_array in [&, &node.outgoing] {
        for e in remove_edges {
            let ret = self.remove_edge(e);
            // debug_assert!(ret.is_some());
            let _ = ret;
        }
        // }

        //let node_weight = replace(&mut self.g.nodes[a.index()].weight, Entry::Empty(self.free_node));
        //self.g.nodes[a.index()].next = [EdgeIndex::end(), EdgeIndex::end()];
        let node_slot = self.nodes.get_mut(a.index())?;

        if let Some(e) = node_slot.outgoing.get_mut(0) {
            *e = self.free_node._into_edge();
        } else {
            node_slot.outgoing.push(self.free_node._into_edge());
        }
        if let Some(e) = node_slot.incoming.get_mut(0) {
            *e = EdgeIndex::end();
        } else {
            node_slot.incoming.push(EdgeIndex::end());
        }
        // node_slot.incoming[0] =
        if self.free_node != NodeIndex::end() {
            let free_node_slot = self.nodes.get_mut(self.free_node.index())?;
            if let Some(e) = free_node_slot.incoming.get_mut(0) {
                *e = a._into_edge();
            } else {
                free_node_slot.incoming.push(a._into_edge());
            }
        }
        self.free_node = a;
        self.node_count -= 1;

        Some(node_weight)
    }

    /// Remove an edge and return its edge weight, or `None` if it didn't exist.
    ///
    /// Invalidates the edge index `e` but no other.
    ///
    /// Computes in **O(e')** time, where **e'** is the number of edges
    /// connected to the same endpoints as `e`.
    pub fn remove_edge(&mut self, e: EdgeIndex) -> Option<E> {
        // every edge is part of two lists,
        // outgoing and incoming edges.
        // Remove it from both
        match self.edges.get(e.index()) {
            None => return None,
            Some(x) if x.weight.is_none() => return None,
            _ => (),
        };

        let edge = &mut self.edges[e.index()];

        let clear_up_port = |port: PortIndex, port_array: &mut Vec<_>| {
            let port = port.index();
            if let Some(e) = port_array.get_mut(port) {
                if *e != EdgeIndex::end() {
                    *e = EdgeIndex::end();
                }
            }
        };

        let NodePort { node, port } = edge.node_ports[0];
        let (n1, p1) = (
            self.nodes.get_mut(node.index()).expect("Node not found."),
            port,
        );
        clear_up_port(p1, &mut n1.outgoing);

        let NodePort { node, port } = edge.node_ports[1];

        let (n2, p2) = (
            self.nodes.get_mut(node.index()).expect("Node not found."),
            port,
        );

        clear_up_port(p2, &mut n2.incoming);

        // Clear the edge and put it in the free list
        edge.node_ports = [
            NodePort::new(self.free_edge._into_node(), PortIndex::end()),
            NodePort::new(NodeIndex::end(), PortIndex::end()),
        ];
        self.free_edge = e;
        self.edge_count -= 1;
        edge.weight.take()
    }

    pub fn node_weight(&self, a: NodeIndex) -> Option<&N> {
        self.nodes.get(a.index())?.weight.as_ref()
    }

    pub fn edge_weight(&self, e: EdgeIndex) -> Option<&E> {
        self.edges.get(e.index())?.weight.as_ref()
    }

    pub fn edge_endpoints(&self, e: EdgeIndex) -> Option<[NodePort; 2]> {
        Some(self.edges.get(e.index())?.node_ports)
    }
    fn get_node(&self, n: NodeIndex) -> &Node<N> {
        self.nodes.get(n.index()).expect("Node not found")
    }
    pub fn node_edges(
        &self,
        n: NodeIndex,
        direction: Direction,
    ) -> impl Iterator<Item = &EdgeIndex> {
        let node = self.get_node(n);
        (match direction {
            Direction::Incoming => &node.incoming,
            Direction::Outgoing => &node.outgoing,
        })
        .iter()
        .filter(|e| **e != EdgeIndex::end())
    }

    pub fn neighbours(
        &self,
        n: NodeIndex,
        direction: Direction,
    ) -> impl Iterator<Item = NodePort> + '_ {
        self.node_edges(n, direction)
            .map(move |e| self.edge_endpoints(*e).unwrap()[direction as usize])
    }

    pub fn node_boundary_size(&self, n: NodeIndex) -> [usize; 2] {
        let node = self.get_node(n);

        [node.incoming.len(), node.outgoing.len()]
    }
    // pub(crate) fn incoming_ports(&self, n: NodeIndex) -> &[EdgeIndex] {
    //     &self.get_node(n).incoming[..]
    // }
    pub fn edge_at_port(&self, np: NodePort, direction: Direction) -> Option<EdgeIndex> {
        let node = self.get_node(np.node);
        (match direction {
            Direction::Incoming => &node.incoming,
            Direction::Outgoing => &node.outgoing,
        })
        .get(np.port.index())
        .copied()
    }

    /// Return an iterator over the node indices of the graph
    pub fn node_indices(&self) -> NodeIndices<N> {
        NodeIndices {
            iter: self.nodes.iter().enumerate(),
        }
    }

    pub fn node_weights(&self) -> impl Iterator<Item = &N> + '_ {
        self.nodes.iter().filter_map(|n| n.weight.as_ref())
    }

    pub fn node_count(&self) -> usize {
        self.node_count
    }

    pub fn edge_count(&self) -> usize {
        self.edge_count
    }

    pub fn edge_indices(&self) -> impl Iterator<Item = EdgeIndex> + '_ {
        self.edges
            .iter()
            .enumerate()
            .filter_map(|(i, n)| n.weight.as_ref().map(|_| EdgeIndex::new(i)))
    }

    pub fn edge_weights(&self) -> impl Iterator<Item = &E> + '_ {
        self.edges.iter().filter_map(|e| e.weight.as_ref())
    }

    pub fn next_edge(&self, e: &EdgeIndex) -> Option<EdgeIndex> {
        let NodePort { node, port, .. } = self.edges[e.index()].node_ports[1];
        if node == NodeIndex::end() {
            return None;
        }
        if let Some(e) = self.nodes[node.index()].outgoing.get(port.index()) {
            if *e != EdgeIndex::end() {
                Some(*e)
            } else {
                None
            }
        } else {
            None
        }
    }

    pub fn insert_graph(&mut self, other: Self) -> (NodeMap, EdgeMap) {
        let node_map: HashMap<NodeIndex, NodeIndex> = other
            .nodes
            .into_iter()
            .enumerate()
            .filter_map(|(i, n)| {
                n.weight
                    .map(|weight| (NodeIndex::new(i), self.add_node(weight)))
            })
            .collect();

        let edge_map: HashMap<EdgeIndex, EdgeIndex> = other
            .edges
            .into_iter()
            .enumerate()
            .filter_map(|(i, e)| {
                e.weight.map(|weight| {
                    let [mut np1, mut np2] = e.node_ports;
                    np1.node = node_map[&np1.node];
                    np2.node = node_map[&np2.node];
                    (EdgeIndex::new(i), self.add_edge(np1, np2, weight))
                })
            })
            .collect();
        (node_map, edge_map)
    }

    /**
    Remove all invalid nodes and edges. Update internal references.
    INVALIDATES EXTERNAL NODE AND EDGE REFERENCES
    */
    pub fn remove_invalid(mut self) -> (Self, NodeMap, EdgeMap) {
        // TODO optimise
        let (old_indices, mut new_nodes): (Vec<_>, Vec<_>) = self
            .nodes
            .into_iter()
            .enumerate()
            .filter(|(_, x)| x.weight.is_some())
            .unzip();

        let index_map: HashMap<_, _> = old_indices
            .into_iter()
            .enumerate()
            .map(|(a, b)| (NodeIndex::new(b), NodeIndex::new(a)))
            .collect();

        let (old_edge_indices, new_edges): (Vec<_>, Vec<_>) = self
            .edges
            .into_iter()
            .enumerate()
            .filter(|(_, x)| x.weight.is_some())
            .map(|(i, mut e)| {
                for np in &mut e.node_ports {
                    np.node = index_map[&np.node];
                }
                (i, e)
            })
            .unzip();

        let edge_index_map: HashMap<_, _> = old_edge_indices
            .into_iter()
            .enumerate()
            .map(|(a, b)| (EdgeIndex::new(b), EdgeIndex::new(a)))
            .collect();

        for node in &mut new_nodes {
            for lst in [&mut node.incoming, &mut node.outgoing] {
                for e in lst.iter_mut() {
                    if *e != EdgeIndex::end() {
                        *e = edge_index_map[e];
                    }
                }
            }
        }

        self.node_count = new_nodes.len();
        self.nodes = new_nodes;
        self.edge_count = new_edges.len();
        self.edges = new_edges;
        self.free_node = NodeIndex::end();
        self.free_edge = EdgeIndex::end();

        (self, index_map, edge_index_map)
    }
}

#[cfg(test)]
mod tests {
    use crate::graph::graph::Direction;

    use super::Graph;

    #[test]
    fn test_update_edge() {
        let mut g = Graph::<u8, i8>::with_capacity(3, 2);

        let n0 = g.add_node(0);
        let n1 = g.add_node(1);
        let n2 = g.add_node(2);

        let e0 = g.add_edge((n0, 0), (n1, 0), -3);
        let e1 = g.add_edge((n1, 0), (n2, 0), -4);
        let e2 = g.add_edge((n0, 1), (n2, 1), -5);
        let n3 = g.add_node(3);
        g.update_edge(e0, (n0, 0), (n3, 0));
        g.update_edge(e1, (n3, 0), (n2, 0));

        assert_eq!(g.remove_node(n1), Some(1));

        assert_eq!(g.node_count(), 3);
        assert_eq!(g.edge_count(), 3);

        assert_eq!(g.edge_weight(e0), Some(&-3));
        assert_eq!(g.edge_weight(e1), Some(&-4));

        assert_eq!(g.edge_endpoints(e0), Some([(n0, 0).into(), (n3, 0).into()]));
        assert_eq!(g.edge_endpoints(e1), Some([(n3, 0).into(), (n2, 0).into()]));

        assert_eq!(
            g.node_edges(n0, Direction::Outgoing).collect::<Vec<_>>(),
            vec![&e0, &e2]
        );
        assert_eq!(
            g.node_edges(n2, Direction::Incoming).collect::<Vec<_>>(),
            vec![&e1, &e2]
        );
        assert_eq!(
            g.node_edges(n3, Direction::Incoming).collect::<Vec<_>>(),
            vec![&e0]
        );
        assert_eq!(
            g.node_edges(n3, Direction::Outgoing).collect::<Vec<_>>(),
            vec![&e1]
        );
    }
}

/// Iterator over the node indices of a graph.
#[derive(Debug, Clone)]
pub struct NodeIndices<'a, N: 'a> {
    iter: iter::Enumerate<slice::Iter<'a, Node<N>>>,
}

impl<'a, N> Iterator for NodeIndices<'a, N> {
    type Item = NodeIndex;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.find_map(|(i, node)| {
            if node.weight.is_some() {
                Some(NodeIndex::new(i))
            } else {
                None
            }
        })
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (_, upper) = self.iter.size_hint();
        (0, upper)
    }
}
