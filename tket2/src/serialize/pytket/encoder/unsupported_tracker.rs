//! Tracking of subgraphs of unsupported nodes in the hugr.

use std::collections::HashMap;

use hugr::core::HugrNode;
use hugr::HugrView;
use petgraph::unionfind::UnionFind;

use crate::Circuit;

/// A structure for tracking nodes in the hugr that cannot be encoded as TKET1
/// operations.
///
/// The nodes are accumulated in connected components of the hugr. When they
/// cannot be grown further, each component is encoded as a single TKET1 barrier
/// containing the unsupported operations as metadata.
#[derive(Debug, Clone)]
pub struct UnsupportedTracker<N> {
    /// Unsupported nodes in the hugr.
    ///
    /// Stores the index each node in [`Self::components`].
    nodes: HashMap<N, UnsupportedNode>,
    /// A UnionFind structure for tracking connected components of `Self::nodes`.
    components: UnionFind<usize>,
}

/// The connected component ID of a node.
///
/// Multiple IDs may be merged together via the union-find structure in
/// [`UnsupportedTracker::components`].
type ComponentId = usize;

#[derive(Debug, Clone, Copy, Default)]
struct UnsupportedNode {
    /// The index of the node in [`UnsupportedTracker::components`].
    component: ComponentId,
    /// Whether the node's component has been extracted already.
    ///
    /// Once that happens, we can no longer add new elements to the component
    /// and must instead extract a separate set of operations.
    extracted: bool,
}

impl<N: HugrNode> UnsupportedTracker<N> {
    /// Create a new [`UnsupportedTracker`].
    pub fn new(_circ: &Circuit<impl HugrView>) -> Self {
        Self {
            nodes: HashMap::new(),
            components: UnionFind::new_empty(),
        }
    }

    /// Record an unsupported node in the hugr.
    pub fn record_node(&mut self, node: N, circ: &Circuit<impl HugrView<Node = N>>) {
        let node_data = UnsupportedNode {
            component: self.components.new_set(),
            extracted: false,
        };
        self.nodes.insert(node, node_data);

        // Take the union of the component with any currently tracked incoming
        // neighbour.
        for neighbour in circ.hugr().input_neighbours(node) {
            if let Some(neigh_data) = self.nodes.get(&neighbour) {
                // Once a node's component has been extracted, we cannot add new
                // nodes to it.
                if neigh_data.extracted {
                    continue;
                }
                self.components
                    .union(neigh_data.component, node_data.component);
            }
        }
    }

    /// Returns the connected component of a node, and marks its elements as
    /// extracted.
    ///
    /// Once a component has been extracted, no new nodes can be added to it and
    /// calling [`UnsupportedTracker::record_node`] will use a new component
    /// instead.
    pub fn extract_component(&mut self, node: N) -> Vec<N> {
        let node_data = self.nodes.get_mut(&node).unwrap();
        node_data.extracted = true;
        let component = node_data.component;

        // Compute the nodes in the component, and mark them as extracted.
        //
        // TODO: Implement efficient iteration over the nodes in a component on petgraph,
        // and use it here. For now we do a BFS.
        let mut nodes = vec![];
        let mut queue = vec![node];
        while let Some(node) = queue.pop() {
            self.nodes.get_mut(&node).unwrap().extracted = true;
            nodes.push(node);
            for neighbour in self.nodes.keys() {
                if self
                    .components
                    .equiv(self.nodes[neighbour].component, component)
                {
                    queue.push(*neighbour);
                }
            }
        }

        nodes
    }
}

impl<N> Default for UnsupportedTracker<N> {
    fn default() -> Self {
        Self {
            nodes: HashMap::new(),
            components: UnionFind::new_empty(),
        }
    }
}
