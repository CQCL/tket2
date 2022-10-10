#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
use std::collections::BTreeMap;
use std::fmt::{self, Debug};
use std::hash::Hash;
use std::iter::FusedIterator;
use std::{iter, slice};
use thiserror::Error;

/// The default integer type for graph indices.
/// `u32` is the default to reduce the size of the graph's data and improve
/// performance in the common case.
///
/// Used for node and edge indices in `Graph` and `StableGraph`, used
/// for node indices in `Csr`.
pub type DefaultIx = u32;

/// Node identifier.
#[cfg_attr(feature = "pyo3", derive(FromPyObject))]
#[derive(Copy, Clone, PartialEq, PartialOrd, Eq, Ord, Hash, Debug)]
pub struct NodeIndex(DefaultIx);

impl NodeIndex {
    #[inline]
    pub fn new(x: usize) -> Self {
        NodeIndex(x as DefaultIx)
    }

    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }

    #[inline]
    pub fn end() -> Self {
        NodeIndex(DefaultIx::MAX)
    }

    fn _into_edge(self) -> EdgeIndex {
        EdgeIndex(self.0)
    }

    fn valid(self) -> Option<Self> {
        if self == Self::end() {
            None
        } else {
            Some(self)
        }
    }
}

impl Default for NodeIndex {
    fn default() -> Self {
        Self::end()
    }
}

impl From<DefaultIx> for NodeIndex {
    fn from(ix: DefaultIx) -> Self {
        NodeIndex(ix)
    }
}

/// Edge identifier.
#[cfg_attr(feature = "pyo3", derive(FromPyObject))]
#[derive(Copy, Clone, PartialEq, PartialOrd, Eq, Ord, Hash, Debug)]
pub struct EdgeIndex(DefaultIx);

impl EdgeIndex {
    #[inline]
    pub fn new(x: usize) -> Self {
        EdgeIndex(x as DefaultIx)
    }

    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }

    /// An invalid `EdgeIndex` used to denote absence of an edge, for example
    /// to end an adjacency list.
    #[inline]
    pub fn end() -> Self {
        EdgeIndex(DefaultIx::MAX)
    }

    fn _into_node(self) -> NodeIndex {
        NodeIndex(self.0)
    }

    fn valid(self) -> Option<Self> {
        if self == Self::end() {
            None
        } else {
            Some(self)
        }
    }
}

impl Default for EdgeIndex {
    fn default() -> Self {
        Self::end()
    }
}

impl From<DefaultIx> for EdgeIndex {
    fn from(ix: DefaultIx) -> Self {
        EdgeIndex(ix)
    }
}

/// The graph's node type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Node<N> {
    /// Associated node data or `None` if the node is free.
    pub weight: Option<N>,

    /// The first incoming and outgoing edge.
    ///
    /// If the node is free, the first component is used as part of a linked list of free nodes.
    /// The types do not match up, but edge and node indices are both just integers so we repurpose
    /// the space.
    edges: [EdgeIndex; 2],
}

impl<N> Node<N> {
    fn relink(&mut self, edge_map: &EdgeMap) {
        for i in 0..=1 {
            self.edges[i] = edge_map.get(&self.edges[i]).copied().unwrap_or_default();
        }
    }
}

/// The graph's edge type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Edge<E> {
    /// Associated edge data or `None` if the edge is free.
    pub weight: Option<E>,

    /// The nodes that the edge is connected to or `NodeIndex::end()` if the edge is dangling.
    ///
    /// The first component is the target and the second component the source of the edge. This
    /// is so that the array can be indexed by `Direction`.
    nodes: [NodeIndex; 2],

    /// Intrusive linked lists that point to the next edge connected to the edge's endpoints.
    /// This is `EdgeIndex::end()` if the edge is dangling.
    ///
    /// If the edge is free, the first component is used as part of a linked list of free edges.
    next: [EdgeIndex; 2],
}

impl<E> Edge<E> {
    fn relink(&mut self, node_map: &NodeMap, edge_map: &EdgeMap) {
        for i in 0..=1 {
            self.next[i] = edge_map.get(&self.next[i]).copied().unwrap_or_default();
            self.nodes[i] = node_map.get(&self.nodes[i]).copied().unwrap_or_default();
        }
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

impl Direction {
    pub fn index(self) -> usize {
        self as usize
    }

    pub fn reverse(self) -> Direction {
        match self {
            Direction::Incoming => Direction::Outgoing,
            Direction::Outgoing => Direction::Incoming,
        }
    }
}

/// Incoming and outgoing.
pub const DIRECTIONS: [Direction; 2] = [Direction::Incoming, Direction::Outgoing];

type NodeMap = BTreeMap<NodeIndex, NodeIndex>;
type EdgeMap = BTreeMap<EdgeIndex, EdgeIndex>;

#[derive(Clone, PartialEq)]
pub struct Graph<N, E> {
    nodes: Vec<Node<N>>,
    edges: Vec<Edge<E>>,
    node_count: usize,
    edge_count: usize,
    free_node: NodeIndex,
    free_edge: EdgeIndex,
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

impl<N, E> Default for Graph<N, E> {
    fn default() -> Self {
        Self::new()
    }
}

impl<N, E> Graph<N, E> {
    /// Create a new empty graph.
    pub fn new() -> Self {
        Self::with_capacity(0, 0)
    }

    /// Create a new empty graph with preallocated capacities for nodes and edges.
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

    /// Add a node to the graph.
    pub fn add_node(&mut self, weight: N) -> NodeIndex {
        let node = Node {
            weight: Some(weight),
            edges: [EdgeIndex::default(); 2],
        };

        self.node_count += 1;

        match self.free_node.valid() {
            Some(index) => {
                self.free_node = self.nodes[index.index()].edges[0]._into_node();
                self.nodes[index.index()] = node;
                index
            }
            None => {
                let index = self.nodes.len();
                self.nodes.push(node);
                NodeIndex::new(index)
            }
        }
    }

    /// Add a node to the graph with specified incoming and outgoing edges.
    ///
    /// # Example
    ///
    /// ```
    /// # use tket2::graph::graph::{Graph, Direction};
    /// let mut graph = Graph::<i8, i8>::new();
    ///
    /// let e1 = graph.add_edge(-1);
    /// let e2 = graph.add_edge(-2);
    /// let e3 = graph.add_edge(-3);
    /// let n0 = graph.add_node_with_edges(0, [e1, e2], [e3]).unwrap();
    ///
    /// assert!(graph.node_edges(n0, Direction::Incoming).eq([e1, e2]));
    /// assert!(graph.node_edges(n0, Direction::Outgoing).eq([e3]));
    /// ```
    pub fn add_node_with_edges(
        &mut self,
        weight: N,
        incoming: impl IntoIterator<Item = EdgeIndex>,
        outgoing: impl IntoIterator<Item = EdgeIndex>,
    ) -> Result<NodeIndex, ConnectError> {
        let node = self.add_node(weight);
        self.connect_many(node, incoming, Direction::Incoming, None)?;
        self.connect_many(node, outgoing, Direction::Outgoing, None)?;
        Ok(node)
    }

    /// Add an edge to the graph.
    pub fn add_edge(&mut self, weight: E) -> EdgeIndex {
        let edge = Edge {
            weight: Some(weight),
            next: [EdgeIndex::default(); 2],
            nodes: [NodeIndex::default(); 2],
        };

        self.edge_count += 1;

        match self.free_edge.valid() {
            Some(index) => {
                self.free_edge = self.edges[index.index()].next[0];
                self.edges[index.index()] = edge;
                index
            }
            None => {
                let index = self.edges.len();
                self.edges.push(edge);
                EdgeIndex::new(index)
            }
        }
    }

    /// Remove a node from the graph.
    ///
    /// The edges connected to the node will remain in the graph but will become dangling.
    ///
    /// Returns the node's weight if it existed or `None` otherwise.
    ///
    /// # Example
    ///
    /// ```
    /// # use tket2::graph::graph::{Graph, Direction};
    /// let mut graph = Graph::<i8, i8>::new();
    ///
    /// let e1 = graph.add_edge(-1);
    /// let n0 = graph.add_node_with_edges(0, [e1], []).unwrap();
    ///
    /// assert_eq!(graph.remove_node(n0), Some(0));
    /// assert_eq!(graph.remove_node(n0), None);
    /// assert!(graph.has_edge(e1));
    /// assert_eq!(graph.edge_endpoint(e1, Direction::Incoming), None);
    /// ```
    pub fn remove_node(&mut self, node_index: NodeIndex) -> Option<N> {
        let node = self.nodes.get_mut(node_index.index())?;
        let weight = std::mem::take(&mut node.weight)?;

        for direction in DIRECTIONS {
            let mut edge_index = node.edges[direction.index()];

            while edge_index != EdgeIndex::end() {
                let edge = &mut self.edges[edge_index.index()];
                edge.nodes[direction.index()] = NodeIndex::end();
                edge_index = std::mem::take(&mut edge.next[direction.index()]);
            }
        }

        self.nodes[node_index.index()].edges[0] = self.free_node._into_edge();
        self.free_node = node_index;

        self.node_count -= 1;

        Some(weight)
    }

    /// Remove an edge from the graph.
    ///
    /// The nodes that the edge is connected to will remain the graph but will no longer refer to
    /// the deleted edge. This changes the indices of the edges adjacent to a node.
    ///
    /// Returns the edge's weight if it existed or `None` otherwise.
    ///
    /// # Example
    ///
    /// ```
    /// # use tket2::graph::graph::{Graph, Direction};
    /// let mut graph = Graph::<i8, i8>::new();
    ///
    /// let e1 = graph.add_edge(-1);
    /// let e2 = graph.add_edge(-2);
    /// let e3 = graph.add_edge(-3);
    /// let n0 = graph.add_node_with_edges(0, [e1, e2, e3], []).unwrap();
    ///
    /// assert_eq!(graph.remove_edge(e2), Some(-2));
    /// assert_eq!(graph.remove_edge(e2), None);
    /// assert!(graph.node_edges(n0, Direction::Incoming).eq([e1, e3]));
    /// ```
    pub fn remove_edge(&mut self, e: EdgeIndex) -> Option<E> {
        self.disconnect(e, Direction::Incoming);
        self.disconnect(e, Direction::Outgoing);

        let edge = self.edges.get_mut(e.index())?;
        let weight = std::mem::take(&mut edge.weight)?;

        self.edges[e.index()].next[0] = self.free_edge;
        self.free_edge = e;

        self.edge_count -= 1;

        Some(weight)
    }

    /// Check whether the graph has a node with a given index.
    pub fn has_node(&self, n: NodeIndex) -> bool {
        if let Some(node) = self.nodes.get(n.index()) {
            node.weight.is_some()
        } else {
            false
        }
    }

    /// Check whether the graph has an edge with a given index.
    pub fn has_edge(&self, e: EdgeIndex) -> bool {
        if let Some(edge) = self.edges.get(e.index()) {
            edge.weight.is_some()
        } else {
            false
        }
    }

    /// Connect an edge to an incoming or outgoing port of a node.
    ///
    /// The edge will be connected after the edge with index `edge_prev`.
    ///
    /// # Example
    ///
    /// ```
    /// # use tket2::graph::graph::{Graph, Direction};
    /// let mut graph = Graph::<i8, i8>::new();
    ///
    /// let e1 = graph.add_edge(-1);
    /// let e2 = graph.add_edge(-2);
    /// let e3 = graph.add_edge(-3);
    /// let n0 = graph.add_node_with_edges(0, [e1, e3], []).unwrap();
    ///
    /// graph.connect_after(n0, e2, Direction::Incoming, e1).unwrap();
    ///
    /// assert!(graph.node_edges(n0, Direction::Incoming).eq([e1, e2, e3]));
    /// ```
    pub fn connect_after(
        &mut self,
        node: NodeIndex,
        edge: EdgeIndex,
        direction: Direction,
        edge_prev: EdgeIndex,
    ) -> Result<(), ConnectError> {
        if !self.has_node(node) {
            return Err(ConnectError::UnknownNode);
        } else if !self.has_edge(edge) {
            return Err(ConnectError::UnknownEdge);
        } else if !self.has_edge(edge_prev) {
            return Err(ConnectError::UnknownEdge);
        } else if self.edges[edge_prev.index()].nodes[direction.index()] != node {
            return Err(ConnectError::NodeMismatch);
        } else if self.edges[edge.index()].nodes[direction.index()] != NodeIndex::end() {
            return Err(ConnectError::AlreadyConnected);
        }

        self.edges[edge.index()].nodes[direction.index()] = node;
        self.edges[edge.index()].next[direction.index()] =
            self.edges[edge_prev.index()].next[direction.index()];
        self.edges[edge_prev.index()].next[direction.index()] = edge;

        Ok(())
    }

    /// Connect an edge to an incoming or outgoing port of a node.
    ///
    /// The edge will be connected before all other edges adjacent to the node.
    ///
    /// # Example
    ///
    /// ```
    /// # use tket2::graph::graph::{Graph, Direction};
    /// let mut graph = Graph::<i8, i8>::new();
    ///
    /// let e1 = graph.add_edge(-1);
    /// let e2 = graph.add_edge(-2);
    /// let n0 = graph.add_node(0);
    ///
    /// graph.connect_first(n0, e2, Direction::Incoming).unwrap();
    /// graph.connect_first(n0, e1, Direction::Incoming).unwrap();
    ///
    /// assert!(graph.node_edges(n0, Direction::Incoming).eq([e1, e2]));
    /// ```
    pub fn connect_first(
        &mut self,
        node: NodeIndex,
        edge: EdgeIndex,
        direction: Direction,
    ) -> Result<(), ConnectError> {
        if !self.has_node(node) {
            return Err(ConnectError::UnknownNode);
        } else if !self.has_edge(edge) {
            return Err(ConnectError::UnknownEdge);
        } else if self.edges[edge.index()].nodes[direction.index()] != NodeIndex::end() {
            return Err(ConnectError::AlreadyConnected);
        }

        self.edges[edge.index()].nodes[direction.index()] = node;
        self.edges[edge.index()].next[direction.index()] =
            self.nodes[node.index()].edges[direction.index()];
        self.nodes[node.index()].edges[direction.index()] = edge;

        Ok(())
    }

    /// Connect an edge to an incoming or outgoing port of a node.
    pub fn connect(
        &mut self,
        node: NodeIndex,
        edge: EdgeIndex,
        direction: Direction,
        edge_prev: Option<EdgeIndex>,
    ) -> Result<(), ConnectError> {
        match edge_prev {
            Some(edge_prev) => self.connect_after(node, edge, direction, edge_prev),
            None => self.connect_first(node, edge, direction),
        }
    }

    /// Connect a collection of edges to incoming or outgoing ports of a node.
    pub fn connect_many(
        &mut self,
        node: NodeIndex,
        edges: impl IntoIterator<Item = EdgeIndex>,
        direction: Direction,
        mut edge_prev: Option<EdgeIndex>,
    ) -> Result<(), ConnectError> {
        for edge in edges {
            self.connect(node, edge, direction, edge_prev)?;
            edge_prev = Some(edge);
        }

        Ok(())
    }

    fn edge_prev(&self, edge_index: EdgeIndex, direction: Direction) -> Option<EdgeIndex> {
        let node_index = self.edge_endpoint(edge_index, direction)?;

        self.node_edges(node_index, direction)
            .skip(1)
            .zip(self.node_edges(node_index, direction))
            .find(|(item, _)| *item == edge_index)
            .map(|(_, prev)| prev)
    }

    /// Disconnect an edge endpoint from a node.
    ///
    /// This operation takes time linear in the number of edges that precede the edge to be
    /// disconnected at the node.
    ///
    /// # Example
    ///
    /// ```
    /// # use tket2::graph::graph::{Graph, Direction};
    /// let mut graph = Graph::<i8, i8>::new();
    ///
    /// let e1 = graph.add_edge(-1);
    /// let e2 = graph.add_edge(-2);
    /// let e3 = graph.add_edge(-3);
    /// let n0 = graph.add_node_with_edges(0, [e1, e2, e3], []).unwrap();
    ///
    /// graph.disconnect(e2, Direction::Incoming);
    /// assert!(graph.node_edges(n0, Direction::Incoming).eq([e1, e3]));
    ///
    /// graph.disconnect(e1, Direction::Incoming);
    /// assert!(graph.node_edges(n0, Direction::Incoming).eq([e3]));
    /// ```
    pub fn disconnect(&mut self, edge_index: EdgeIndex, direction: Direction) {
        if !self.has_edge(edge_index) {
            return;
        }

        let prev = self.edge_prev(edge_index, direction);

        let edge = &mut self.edges[edge_index.index()];
        let node = std::mem::take(&mut edge.nodes[direction.index()]);
        let next = std::mem::take(&mut edge.next[direction.index()]);

        if node == NodeIndex::end() {
            return;
        }

        match prev {
            Some(prev) => self.edges[prev.index()].next[direction.index()] = next,
            None => self.nodes[node.index()].edges[direction.index()] = next,
        }
    }

    /// A reference to the weight of the node with a given index.
    pub fn node_weight(&self, a: NodeIndex) -> Option<&N> {
        self.nodes.get(a.index())?.weight.as_ref()
    }

    /// A mutable reference to the weight of the node with a given index.
    pub fn node_weight_mut(&mut self, a: NodeIndex) -> Option<&mut N> {
        self.nodes.get_mut(a.index())?.weight.as_mut()
    }

    /// A reference to the weight of the edge with a given index.
    pub fn edge_weight(&self, e: EdgeIndex) -> Option<&E> {
        self.edges.get(e.index())?.weight.as_ref()
    }

    /// A mutable reference to the weight of the edge with a given index.
    pub fn edge_weight_mut(&mut self, e: EdgeIndex) -> Option<&mut E> {
        self.edges.get_mut(e.index())?.weight.as_mut()
    }

    /// The endpoint of an edge in a given direction.
    ///
    /// Returns `None` if the edge does not exist or has no endpoint in that direction.
    pub fn edge_endpoint(&self, e: EdgeIndex, direction: Direction) -> Option<NodeIndex> {
        self.edges.get(e.index())?.nodes[direction.index()].valid()
    }

    /// Iterator over the edges that are connected to a node.
    pub fn node_edges(&self, n: NodeIndex, direction: Direction) -> NodeEdges<'_, N, E> {
        let next = self
            .nodes
            .get(n.index())
            .map(|node| node.edges[direction.index()])
            .unwrap_or_default();

        NodeEdges {
            graph: self,
            direction,
            next,
        }
    }

    /// Iterator over the node indices of the graph.
    ///
    /// # Example
    ///
    /// ```
    /// # use tket2::graph::graph::{Graph, Direction};
    /// let mut graph = Graph::<i8, i8>::new();
    ///
    /// let n0 = graph.add_node(0);
    /// let n1 = graph.add_node(1);
    /// let n2 = graph.add_node(2);
    ///
    /// graph.remove_node(n1);
    ///
    /// assert!(graph.node_indices().eq([n0, n2]));
    /// ```
    pub fn node_indices(&self) -> NodeIndices<N> {
        NodeIndices {
            len: self.node_count(),
            iter: self.nodes.iter().enumerate(),
        }
    }

    /// Iterator over the edge indices of the graph.
    ///
    /// # Example
    ///
    /// ```
    /// # use tket2::graph::graph::{Graph, Direction};
    /// let mut graph = Graph::<i8, i8>::new();
    ///
    /// let e1 = graph.add_edge(-1);
    /// let e2 = graph.add_edge(-2);
    /// let e3 = graph.add_edge(-3);
    ///
    /// graph.remove_edge(e2);
    ///
    /// assert!(graph.edge_indices().eq([e1, e3]));
    /// ```
    pub fn edge_indices(&self) -> EdgeIndices<E> {
        EdgeIndices {
            len: self.edge_count(),
            iter: self.edges.iter().enumerate(),
        }
    }

    /// Iterator over the node weights of the graph.
    pub fn node_weights(&self) -> impl Iterator<Item = &N> + '_ {
        self.nodes.iter().filter_map(|n| n.weight.as_ref())
    }

    /// Iterator over the edge weights of the graph.
    pub fn edge_weights(&self) -> impl Iterator<Item = &E> + '_ {
        self.edges.iter().filter_map(|n| n.weight.as_ref())
    }

    /// Number of nodes in the graph.
    #[inline]
    pub fn node_count(&self) -> usize {
        self.node_count
    }

    /// Number of edges in the graph.
    #[inline]
    pub fn edge_count(&self) -> usize {
        self.edge_count
    }

    /// Whether the graph has neither nodes nor edges.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.node_count == 0 && self.edge_count == 0
    }

    /// Insert a graph into this graph.
    ///
    /// Returns maps from the node and edge indices in the original graph to the
    /// new indices which were allocated in this graph.
    ///
    /// [Graph::merge_edges] can be used along dangling edges to connect the inserted
    /// subgraph with the rest of the graph
    pub fn insert_graph(&mut self, other: Self) -> (NodeMap, EdgeMap) {
        let node_map: NodeMap = other
            .nodes
            .into_iter()
            .enumerate()
            .filter_map(|(index, node)| {
                let new_index = self.add_node(node.weight?);
                let old_index = NodeIndex::new(index);
                Some((old_index, new_index))
            })
            .collect();

        let edge_map: EdgeMap = other
            .edges
            .into_iter()
            .enumerate()
            .filter_map(|(index, edge)| {
                let new_index = self.add_edge(edge.weight?);
                let old_index = EdgeIndex::new(index);
                Some((old_index, new_index))
            })
            .collect();

        for node_index in node_map.values() {
            self.nodes[node_index.index()].relink(&edge_map);
        }

        for edge_index in edge_map.values() {
            self.edges[edge_index.index()].relink(&node_map, &edge_map);
        }

        (node_map, edge_map)
    }

    /// Reindex the nodes and edges to be contiguous.
    ///
    /// Returns maps from the previous node and edge indices to their new indices.
    ///
    /// Preserves the order of nodes and edges.
    ///
    /// This method does not release the unused capacity of the graph's storage after
    /// compacting as it might be needed immediately for new insertions. To reduce the
    /// graph's memory allocation call [Graph::shrink_to_fit] after compacting.
    ///
    /// # Example
    ///
    /// ```
    /// # use tket2::graph::graph::{Graph, Direction};
    /// # use std::collections::BTreeMap;
    /// let mut graph = Graph::<i8, i8>::new();
    ///
    /// let e1 = graph.add_edge(-1);
    /// let e2 = graph.add_edge(-2);
    /// let n0 = graph.add_node_with_edges(0, [e2], [e1]).unwrap();
    /// let n1 = graph.add_node_with_edges(1, [e1], [e2]).unwrap();
    ///
    /// graph.remove_node(n0);
    /// graph.remove_edge(e1);
    ///
    /// let (node_map, edge_map) = graph.compact();
    ///
    /// assert_eq!(node_map, BTreeMap::from_iter([(n1, n0)]));
    /// assert_eq!(edge_map, BTreeMap::from_iter([(e2, e1)]));
    /// assert!(graph.node_edges(n0, Direction::Outgoing).eq([e1]));
    /// ```
    pub fn compact(&mut self) -> (NodeMap, EdgeMap) {
        let node_map: NodeMap = self
            .node_indices()
            .enumerate()
            .map(|(new_index, index)| (index, NodeIndex::new(new_index)))
            .collect();

        let edge_map: EdgeMap = self
            .edge_indices()
            .enumerate()
            .map(|(new_index, index)| (index, EdgeIndex::new(new_index)))
            .collect();

        self.nodes.retain_mut(|node| {
            if node.weight.is_some() {
                node.relink(&edge_map);
                true
            } else {
                false
            }
        });

        self.edges.retain_mut(|edge| {
            if edge.weight.is_some() {
                edge.relink(&node_map, &edge_map);
                true
            } else {
                false
            }
        });

        (node_map, edge_map)
    }

    /// Shrinks the graph's data store as much as possible.
    ///
    /// When there are a lot of empty slots, call [Graph::compact] before to make indices contiguous.
    pub fn shrink_to_fit(&mut self) {
        self.edges.shrink_to_fit();
        self.nodes.shrink_to_fit();
    }

    /// Merges two edges together:
    /// The edge with index `from` will be removed and its weight returned. The edge with
    /// index `into` will be connected to the node ports that `from` was connected to.
    ///
    /// This method is useful to connect subgraphs inserted via [Graph::insert_graph]
    /// to the rest of the graph.
    ///
    /// # Errors
    ///
    /// Attempting to merge edges which both are already connected to nodes in the same direction
    /// will result in an error.
    ///
    /// # Example
    ///
    /// ```
    /// # use tket2::graph::graph::{Graph, Direction};
    /// let mut graph = Graph::<i8, i8>::new();
    ///
    /// let e1 = graph.add_edge(-1);
    /// let e2 = graph.add_edge(-2);
    /// let n0 = graph.add_node_with_edges(0, [], [e1]).unwrap();
    /// let n1 = graph.add_node_with_edges(1, [e2], []).unwrap();
    ///
    /// assert_eq!(graph.merge_edges(e2, e1).unwrap(), -2);
    /// assert!(!graph.has_edge(e2));
    /// assert!(graph.node_edges(n0, Direction::Outgoing).eq([e1]));
    /// assert!(graph.node_edges(n1, Direction::Incoming).eq([e1]));
    /// ```
    pub fn merge_edges(&mut self, from: EdgeIndex, into: EdgeIndex) -> Result<E, MergeEdgesError> {
        if !self.has_edge(from) {
            return Err(MergeEdgesError::UnknownEdge);
        } else if !self.has_edge(into) {
            return Err(MergeEdgesError::UnknownEdge);
        }

        for direction in DIRECTIONS {
            let from_node = self.edges[from.index()].nodes[direction.index()];
            let into_node = self.edges[into.index()].nodes[direction.index()];

            if from_node != NodeIndex::end() && into_node != NodeIndex::end() {
                return Err(MergeEdgesError::AlreadyConnected);
            }
        }

        for direction in DIRECTIONS {
            let from_prev = self.edge_prev(from, direction);
            let from_edge = &mut self.edges[from.index()];
            let from_next = std::mem::take(&mut from_edge.next[direction.index()]);
            let from_node = std::mem::take(&mut from_edge.nodes[direction.index()]);

            if from_node == NodeIndex::end() {
                continue;
            }

            self.edges[into.index()].next[direction.index()] = from_next;

            match from_prev {
                Some(prev) => self.edges[prev.index()].next[direction.index()] = into,
                None => self.nodes[from_node.index()].edges[direction.index()] = into,
            }
        }

        Ok(self.remove_edge(from).unwrap())
    }
}

/// Error returned by [Graph::connect] and similar methods.
#[derive(Debug, Error)]
pub enum ConnectError {
    #[error("unknown node")]
    UnknownNode,
    #[error("unknown edge")]
    UnknownEdge,
    #[error("node mismatch")]
    NodeMismatch,
    #[error("edge is already connected")]
    AlreadyConnected,
}

/// Error returned by [Graph::merge_edges].
#[derive(Debug, Error)]
pub enum MergeEdgesError {
    #[error("unknown edge")]
    UnknownEdge,
    #[error("edge is already connected")]
    AlreadyConnected,
}

/// Iterator created by [Graph::node_edges].
#[derive(Clone)]
pub struct NodeEdges<'a, N, E> {
    graph: &'a Graph<N, E>,
    direction: Direction,
    next: EdgeIndex,
}

impl<'a, N, E> Iterator for NodeEdges<'a, N, E> {
    type Item = EdgeIndex;

    fn next(&mut self) -> Option<Self::Item> {
        let next = self.graph.edges[self.next.valid()?.index()].next[self.direction.index()];
        Some(std::mem::replace(&mut self.next, next))
    }
}

impl<'a, N, E> FusedIterator for NodeEdges<'a, N, E> {}

/// Iterator created by [Graph::node_indices].
#[derive(Debug, Clone)]
pub struct NodeIndices<'a, N: 'a> {
    iter: iter::Enumerate<slice::Iter<'a, Node<N>>>,
    len: usize,
}

impl<'a, N> Iterator for NodeIndices<'a, N> {
    type Item = NodeIndex;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.find_map(|(i, node)| {
            if node.weight.is_some() {
                self.len -= 1;
                Some(NodeIndex::new(i))
            } else {
                None
            }
        })
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }
}

impl<'a, N> ExactSizeIterator for NodeIndices<'a, N> {}
impl<'a, N> FusedIterator for NodeIndices<'a, N> {}

/// Iterator created by [Graph::edge_indices].
#[derive(Debug, Clone)]
pub struct EdgeIndices<'a, N: 'a> {
    iter: iter::Enumerate<slice::Iter<'a, Edge<N>>>,
    len: usize,
}

impl<'a, N> Iterator for EdgeIndices<'a, N> {
    type Item = EdgeIndex;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.find_map(|(i, edge)| {
            if edge.weight.is_some() {
                self.len -= 1;
                Some(EdgeIndex::new(i))
            } else {
                None
            }
        })
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }
}

impl<'a, N> ExactSizeIterator for EdgeIndices<'a, N> {}
impl<'a, N> FusedIterator for EdgeIndices<'a, N> {}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    pub fn merge_with_multiple_edges() {
        let mut graph = Graph::<i8, i8>::new();

        let e1 = graph.add_edge(-1);
        let e2 = graph.add_edge(-2);
        let e3 = graph.add_edge(-3);
        let e4 = graph.add_edge(-4);

        let n0 = graph.add_node_with_edges(0, [], [e1, e2, e3]).unwrap();
        let n1 = graph.add_node_with_edges(1, [e1, e4, e3], []).unwrap();

        assert_eq!(graph.merge_edges(e4, e2).unwrap(), -4);
        assert!(graph.node_edges(n0, Direction::Outgoing).eq([e1, e2, e3]));
        assert!(graph.node_edges(n1, Direction::Incoming).eq([e1, e2, e3]));
    }

    #[test]
    pub fn merge_edges_error() {
        let mut graph = Graph::<i8, i8>::new();

        let e1 = graph.add_edge(-1);
        let e2 = graph.add_edge(-2);
        let _ = graph.add_node_with_edges(0, [e1], []).unwrap();
        let _ = graph.add_node_with_edges(1, [e2], []).unwrap();

        assert!(graph.merge_edges(e2, e1).is_err());
    }
}
