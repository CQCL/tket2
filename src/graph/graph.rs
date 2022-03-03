use std::fmt;
use std::hash::Hash;
use std::mem::replace;

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

// unsafe impl IndexType for u8 {
//     #[inline(always)]
//     fn new(x: usize) -> Self {
//         x as u8
//     }
//     #[inline(always)]
//     fn index(&self) -> usize {
//         *self as usize
//     }
//     #[inline(always)]
//     fn max() -> Self {
//         ::std::u8::MAX
//     }
// }

/// Node identifier.
#[derive(Copy, Clone, Default, PartialEq, PartialOrd, Eq, Ord, Hash, Debug)]
pub struct NodeIndex<Ix = DefaultIx>(Ix);
impl<Ix: IndexType> NodeIndex<Ix> {
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

    fn _into_edge(self) -> EdgeIndex<Ix> {
        EdgeIndex(self.0)
    }
}

impl<Ix: IndexType> IndexType for NodeIndex<Ix> {
    fn index(&self) -> usize {
        self.0.index()
    }
    fn new(x: usize) -> Self {
        NodeIndex::new(x)
    }
    fn max() -> Self {
        NodeIndex(<Ix as IndexType>::max())
    }
}

impl<Ix: IndexType> From<Ix> for NodeIndex<Ix> {
    fn from(ix: Ix) -> Self {
        NodeIndex(ix)
    }
}
/// Edge identifier.
#[derive(Copy, Clone, Default, PartialEq, PartialOrd, Eq, Ord, Hash, Debug)]
pub struct EdgeIndex<Ix = DefaultIx>(Ix);

impl<Ix: IndexType> EdgeIndex<Ix> {
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

    fn _into_node(self) -> NodeIndex<Ix> {
        NodeIndex(self.0)
    }
}

impl<Ix: IndexType> From<Ix> for EdgeIndex<Ix> {
    fn from(ix: Ix) -> Self {
        EdgeIndex(ix)
    }
}

#[derive(Copy, Clone, Default, PartialEq, PartialOrd, Eq, Ord, Hash, Debug)]
pub struct PortIndex(u8);

impl PortIndex {
    #[inline]
    pub fn index(&self) -> usize {
        self.0 as usize
    }

    #[inline]
    pub fn end() -> Self {
        PortIndex(u8::MAX)
    }
}
#[derive(Copy, Clone, Default, PartialEq, PartialOrd, Eq, Ord, Hash, Debug)]
pub struct NodePort<Ix = DefaultIx> {
    node: NodeIndex<Ix>,
    port: PortIndex,
}

impl<Ix> NodePort<Ix> {
    pub fn new(node: NodeIndex<Ix>, port: PortIndex) -> Self {
        Self { node, port }
    }
}

impl<Ix, Px: Into<u8>> From<(NodeIndex<Ix>, Px)> for NodePort<Ix> {
    fn from((node, p): (NodeIndex<Ix>, Px)) -> Self {
        Self {
            node,
            port: PortIndex(p.into()),
        }
    }
}

/// The graph's node type.
#[derive(Debug)]
pub struct Node<N, Ix = DefaultIx> {
    /// Associated node data.
    pub weight: Option<N>,

    incoming: Vec<EdgeIndex<Ix>>,
    outgoing: Vec<EdgeIndex<Ix>>,
}

/// The graph's edge type.
#[derive(Debug)]
pub struct Edge<E, Ix = DefaultIx> {
    /// Associated edge data.
    pub weight: Option<E>,
    // / Next edge in outgoing and incoming edge lists.
    // next: [EdgeIndex<Ix>; 2],
    /// Start and End node index
    node_ports: [NodePort<Ix>; 2],
}

pub struct Graph<N, E, Ix = DefaultIx> {
    nodes: Vec<Node<N, Ix>>,
    edges: Vec<Edge<E, Ix>>,
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
    free_node: NodeIndex<Ix>,
    free_edge: EdgeIndex<Ix>,
}

impl<N, E, Ix: IndexType> Graph<N, E, Ix> {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            node_count: 0,
            edge_count: 0,
            free_node: NodeIndex::end(),
            free_edge: EdgeIndex::end(),
        }
    }

    pub fn add_node_with_capacity(&mut self, capacity: usize, weight: N) -> NodeIndex<Ix> {
        if self.free_node != NodeIndex::end() {
            let node_idx = self.free_node;
            let node_slot = &mut self.nodes[node_idx.index()];

            let _old = replace(&mut node_slot.weight, Some(weight));
            debug_assert!(_old.is_none());

            let previous_node = node_slot.incoming[1];
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

    pub fn add_node(&mut self, weight: N) -> NodeIndex<Ix> {
        self.add_node_with_capacity(0, weight)
    }

    pub fn add_edge<T: Into<NodePort<Ix>>>(&mut self, a: T, b: T, weight: E) -> EdgeIndex<Ix> {
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
            assert!(<Ix as IndexType>::max().index() == !0 || EdgeIndex::end() != edge_idx);
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
                debug_assert!(port == port_array.len());
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

    pub fn remove_node(&mut self, a: NodeIndex<Ix>) -> Option<N> {
        let node_weight = self.nodes.get_mut(a.index())?.weight.take()?;

        let node = &self.nodes[a.index()];
        let remove_edges: Vec<_> = node
            .incoming
            .iter()
            .chain(&node.outgoing)
            .cloned()
            .filter(|e| *e != EdgeIndex::end())
            .collect();
        // for port_array in [&, &node.outgoing] {
        for e in remove_edges {
            let ret = self.remove_edge(e);
            debug_assert!(ret.is_some());
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
        // node_slot.incoming[0] =
        if self.free_node != NodeIndex::end() {
            self.nodes[self.free_node.index()].incoming[0] = a._into_edge();
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
    pub fn remove_edge(&mut self, e: EdgeIndex<Ix>) -> Option<E> {
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

        let (n1, p1) = match edge.node_ports[0] {
            NodePort { node, port } => (
                self.nodes.get_mut(node.index()).expect("Node not found."),
                port,
            ),
        };
        clear_up_port(p1, &mut n1.outgoing);

        let (n2, p2) = match edge.node_ports[1] {
            NodePort { node, port } => (
                self.nodes.get_mut(node.index()).expect("Node not found."),
                port,
            ),
        };

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

    pub fn node_weight(&self, a: NodeIndex<Ix>) -> Option<&N> {
        self.nodes.get(a.index())?.weight.as_ref()
    }

    pub fn edge_weight(&self, e: EdgeIndex<Ix>) -> Option<&E> {
        self.edges.get(e.index())?.weight.as_ref()
    }

    pub fn edge_endpoints(&self, e: EdgeIndex<Ix>) -> Option<(NodePort<Ix>, NodePort<Ix>)> {
        let [a, b] = self.edges.get(e.index())?.node_ports;
        Some((a, b))
    }

    pub fn incoming_edges(&self, n: NodeIndex<Ix>) -> impl Iterator<Item = &EdgeIndex<Ix>> {
        let node = self.nodes.get(n.index()).expect("Node not found");
        node.incoming.iter()
    }

    pub fn outgoing_edges(&self, n: NodeIndex<Ix>) -> impl Iterator<Item = &EdgeIndex<Ix>> {
        let node = self.nodes.get(n.index()).expect("Node not found");
        node.outgoing.iter()
    }

    pub fn nodes(&self) -> impl Iterator<Item = NodeIndex<Ix>> + '_ {
        self.nodes
            .iter()
            .enumerate()
            .filter_map(|(i, n)| n.weight.as_ref().map(|_| NodeIndex::new(i)))
    }
}
