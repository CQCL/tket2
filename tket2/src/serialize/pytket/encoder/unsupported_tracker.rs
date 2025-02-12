//! Tracking of subgraphs of unsupported nodes in the hugr.

use std::collections::HashMap;

use hugr::{HugrView, Node};
use petgraph::unionfind::UnionFind;

use crate::Circuit;

/// A structure for tracking nodes in the hugr that cannot be encoded as TKET1
/// operations.
///
/// The nodes are accumulated in connected components of the hugr. When they
/// cannot be grown further, each component is encoded as a single TKET1 barrier
/// containing the unsupported operations as metadata.
#[derive(Debug, Clone)]
pub struct UnsupportedTracker {
    /// Unsupported nodes in the hugr.
    ///
    /// Stores the index each node in [`Self::components`].
    nodes: HashMap<Node, UnsupportedNode>,
    /// A UnionFind structure for tracking connected components of `Self::nodes`.
    components: UnionFind<usize>,
    /// The number of components currently tracked by the union-find structure.
    ///
    /// This is temporarily required since we are pre-allocating the union-find
    /// with a fixed size. (See TODO).
    component_count: usize,
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
    extracted: bool,
}

impl UnsupportedTracker {
    /// Create a new [`UnsupportedTracker`].
    pub fn new(circ: &Circuit<impl HugrView>) -> Self {
        Self {
            nodes: HashMap::new(),
            // TODO: Dynamically add elements to avoid allocating too much space.
            // Requires <https://github.com/petgraph/petgraph/issues/729>
            components: UnionFind::new(circ.hugr().node_count()),
            component_count: 0,
        }
    }

    /// Record an unsupported node in the hugr.
    pub fn record_node(&mut self, node: Node, circ: &Circuit<impl HugrView>) {
        let node_data = UnsupportedNode {
            component: self.component_count as ComponentId,
            extracted: false,
        };
        self.component_count += 1;
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
    pub fn extract_component(&mut self, node: Node) -> Vec<Node> {
        let node_data = self.nodes.get_mut(&node).unwrap();
        node_data.extracted = true;
        let component = node_data.component;

        self.nodes
            .iter()
            .filter_map(
                |(n, data)| match self.components.equiv(data.component, component) {
                    true => Some(*n),
                    false => None,
                },
            )
            .collect()
    }
}

impl Default for UnsupportedTracker {
    fn default() -> Self {
        Self {
            nodes: HashMap::new(),
            components: UnionFind::new(0),
            component_count: 0,
        }
    }
}
